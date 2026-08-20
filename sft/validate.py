"""Held-out validation for SFT, decoded from the LIVE student.

At each epoch boundary the held-out records (`SFT_VAL_SPLIT`) are answered by the
model that is being trained — `model.generate` on the in-memory weights — and
graded against ground truth with the same OpenRouter grader the pipeline uses.

This deliberately does NOT go through vLLM. Serving would mean merging the
adapter, writing a full checkpoint, and standing an engine up on a card of its
own every epoch, purely to ask the same weights that are already resident in this
process a few dozen questions. Decoding in-process instead frees that whole GPU
for training (the student shards over both training cards — see
`pipeline.utils.configure_train_gpus`) and removes the serve/tear-down cycle, at
the cost of HF's slower sampling loop over a capped, batched validation set.

What is kept identical to the serving path, because validation is only meaningful
if it measures what a served checkpoint would do:

  * the prompt — `pipeline.prompts.serving_prompt_ids`, the byte-identical
    conditioning vLLM builds for a chat request (and what `sft.dataset` trains on);
  * the stop tokens — harmony's, so a turn ends where the server would end it;
  * the parse — the LAST harmony message is the content vLLM would return, then
    `inference_script._parse_output` pulls `<answer>` out of it;
  * the grader — `pipeline.eval_script.run_eval`.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

import torch

from pipeline import config as pc
from pipeline.eval_script import run_eval
# The repo's single implementation of the <reasoning>/<answer> split, shared with
# the vLLM path so both score the same text the same way (private only in the
# sense that the serving flow reaches it through run_inference).
from pipeline.inference_script import _parse_output
from pipeline.prompts import harmony_stop_token_ids, serving_prompt_ids
from pipeline.utils import logger

from . import config

# `<|channel|>final<|message|>` (or `analysis`, or a channel carrying a
# `to=…` recipient before the message opens). The LAST match in a completion is
# the message the model finished on.
_CHANNEL_RE = re.compile(r"<\|channel\|>(\w+)[^<]*<\|message\|>")
# Anything that closes a harmony message; generation stops on these, but the
# decoded text still carries the token that ended it.
_STOP_MARKERS = ("<|return|>", "<|end|>", "<|call|>", "<|endoftext|>")


def validate(model, tokenizer, val_records: list[dict], epoch: int) -> float:
    """Answer the held-out problems from the live weights and grade them.

    Returns accuracy in [0, 1]. Generation runs in-process (batched, greedy unless
    TEMPERATURE > 0); grading fans out over OpenRouter, which is where the wall
    time would otherwise go serially.
    """
    logger.info("[val] epoch %d: decoding %d held-out problem(s) from the live "
                "student …", epoch, len(val_records))
    completions = _generate(model, tokenizer, [r["problem"] for r in val_records])

    answers, n_unfinished = [], 0
    for raw in completions:
        reasoning, answer = _parse_completion(raw)
        answers.append(answer)
        n_unfinished += not answer and bool(reasoning)
    if n_unfinished:
        # The model spent its whole budget in the analysis channel and never opened
        # a final message — the same shape vLLM reports as content=None. Those score
        # wrong (no answer was produced), but they mean "still thinking", not
        # "answered incorrectly", so they are worth separating in the log.
        logger.warning(
            "[val] epoch %d: %d/%d completion(s) never reached a final message "
            "(hit the %d-token budget mid-reasoning); they grade as wrong.",
            epoch, n_unfinished, len(completions), config.VAL_MAX_NEW_TOKENS,
        )

    correct = graded = 0
    with ThreadPoolExecutor(max_workers=max(1, config.GEN_CONCURRENCY)) as pool:
        futures = {
            pool.submit(run_eval, answer, rec.get("answer", "")): rec
            for answer, rec in zip(answers, val_records)
        }
        for fut in as_completed(futures):
            try:
                correct += bool(fut.result())
                graded += 1
            except Exception as exc:  # a single ungradeable item shouldn't sink it
                logger.warning("[val] problem %s failed to grade: %s",
                               futures[fut].get("number"), exc)
    acc = correct / graded if graded else 0.0
    if not graded:
        # Every grader call raised — an OpenRouter outage or a bad key, not a
        # student that got everything wrong. Say which, so 0.000 isn't misread.
        logger.error(
            "[val] epoch %d: all %d held-out item(s) FAILED to grade — this is a "
            "grader/API failure, not a score of 0. Check the warnings above.",
            epoch, len(val_records),
        )
    logger.info("[val] epoch %d accuracy: %.3f (%d/%d graded).",
                epoch, acc, correct, graded)
    return acc


def _generate(model, tokenizer, problems: list[str]) -> list[str]:
    """Decode one completion per problem, batched, and return the raw texts.

    Prompts are sorted by length so a batch pads to roughly its own longest
    member rather than to the longest in the set; the results are unsorted back to
    the caller's order before returning.
    """
    prompts = [
        serving_prompt_ids(config.SYSTEM_PROMPT, problem, tokenizer,
                           model_path=config.STUDENT_MODEL)
        for problem in problems
    ]
    # The first shard holds the embeddings, so that is where `generate` wants its
    # inputs; accelerate's hooks move activations across the remaining shards.
    device = model.get_input_embeddings().weight.device
    stop_ids = (harmony_stop_token_ids(tokenizer, model_path=config.STUDENT_MODEL)
                or tokenizer.eos_token_id)
    kwargs = dict(
        max_new_tokens=config.VAL_MAX_NEW_TOKENS,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=stop_ids,
    )
    if pc.TEMPERATURE > 0:
        # Sampling follows the pipeline's inference temperature; `set_global_seed`
        # (called once at run start) makes the draw reproducible across epochs.
        kwargs.update(do_sample=True, temperature=pc.TEMPERATURE)
    else:
        kwargs["do_sample"] = False

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    texts: list[str] = [""] * len(prompts)
    batch_size = max(1, config.VAL_BATCH_SIZE)
    with _decoding(model):
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            input_ids, attention_mask = _left_pad(
                [prompts[i] for i in rows], tokenizer.pad_token_id, device
            )
            out = model.generate(input_ids=input_ids,
                                 attention_mask=attention_mask, **kwargs)
            for row, i in enumerate(rows):
                # Special tokens are KEPT: the channel headers are what identify
                # the final message (see _parse_completion).
                texts[i] = tokenizer.decode(out[row, input_ids.shape[1]:],
                                            skip_special_tokens=False)
            logger.info("[val] decoded %d/%d.",
                        min(start + len(rows), len(order)), len(order))
    return texts


@contextmanager
def _decoding(model):
    """Put a mid-training model into a state that can generate, then restore it.

    Three things have to move and move back, because `Trainer` will keep stepping
    this same object afterwards:

      * eval mode — LoRA dropout off, so validation measures the weights;
      * gradient checkpointing OFF — it is incompatible with a KV cache
        (transformers silently drops `use_cache` under it, which would make every
        step re-run the whole prefix);
      * `use_cache` back ON — `sft.model` loads with it off for training.
    """
    was_training = model.training
    was_checkpointing = bool(getattr(model, "is_gradient_checkpointing", False))
    model.eval()
    if was_checkpointing:
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    try:
        with torch.no_grad():
            yield
    finally:
        model.config.use_cache = False
        if was_checkpointing:
            # Same kwargs sft.train passes through TrainingArguments; gpt-oss's MoE
            # router needs the reentrant implementation.
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": True}
            )
        if was_training:
            model.train()


def _left_pad(sequences: list[list[int]], pad_token_id: int, device):
    """Stack prompts into a batch, padded on the LEFT.

    Decoder-only generation continues from the last position of every row, so the
    padding has to sit in front of the prompt (right padding would have the model
    continue from pad tokens). The attention mask zeroes those positions, and
    `generate` derives position_ids from it, so the shorter rows are conditioned
    exactly as they would be alone.
    """
    width = max(len(seq) for seq in sequences)
    input_ids = torch.full((len(sequences), width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, seq in enumerate(sequences):
        input_ids[row, width - len(seq):] = torch.tensor(seq, dtype=torch.long)
        attention_mask[row, width - len(seq):] = 1
    return input_ids.to(device), attention_mask.to(device)


def _parse_completion(raw: str) -> tuple[str, str]:
    """Split a decoded completion into (reasoning, answer).

    A harmony completion is a sequence of channelled messages —

        <|channel|>analysis<|message|>…<|end|><|start|>assistant<|channel|>final<|message|>…<|return|>

    — of which vLLM returns the FINAL one as `content` and the rest as
    reasoning. The last message is therefore the one to parse, and only if it is on
    the `final` channel is there an answer at all: a completion that ran out of
    budget mid-analysis has produced reasoning and nothing else, which is exactly
    what the serving path reports as `content=None`.

    A model whose tokenizer doesn't speak harmony emits no channel headers; then
    the whole completion is the message.
    """
    channels = list(_CHANNEL_RE.finditer(raw))
    if not channels:
        return _parse_output(_strip_stop_markers(raw))
    last = channels[-1]
    message = _strip_stop_markers(raw[last.end():])
    if last.group(1) != "final":
        return message.strip(), ""
    return _parse_output(message)


def _strip_stop_markers(text: str) -> str:
    """Cut a message at the first token that closes it."""
    cuts = [i for i in (text.find(marker) for marker in _STOP_MARKERS) if i >= 0]
    return text[: min(cuts)] if cuts else text
