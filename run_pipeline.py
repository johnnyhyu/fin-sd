#!/usr/bin/env python3
"""
run_pipeline.py — main training loop orchestrator.

For each item in data/trainingset.json:
  1. Run inference on the problem (vLLM)
  2. Evaluate the answer (OpenRouter)
  3. If correct → skip
  4. Generate a hint (OpenRouter)
  5. Truncate reasoning at the mistake (OpenRouter)
  6. Extract logits under two conditions (HuggingFace model)
  7. Compute KL divergence loss
  8. Backpropagate and update weights

Running out of the MAX_NEW_TOKENS budget short-circuits steps 4-7: there is no
located mistake to distil from a rollout that simply never finished, so the item
trains the token-budget term instead (config.LENGTH_PENALTY, flush_length_batch).
That term also attaches to items which merely came close to the budget, including
correct ones that step 3 skips.

Required environment variables:
    OPENROUTER_API_KEY   — for eval / hint / state-finder calls
    VLLM_BASE_URL        — e.g. http://localhost:8000/v1
    VLLM_MODEL           — model id served by vLLM
    HF_MODEL_PATH        — local path or HuggingFace hub id for the same model
                           (used for probing + gradient computation)

Optional:
    HF_DEVICE, HF_DTYPE, LEARNING_RATE, MAX_NEW_TOKENS, TEMPERATURE, …
    (see pipeline/config.py for full list)

To run: CUDA_VISIBLE_DEVICES=4,5,6,7 python run_pipeline.py
"""
import os

# Configure the CUDA caching allocator BEFORE torch is imported (it is pulled in
# transitively by the pipeline.* imports below). expandable_segments lets the
# allocator grow/shrink segments instead of stranding the wide batch-N student-
# sampling blocks as reserved memory through the batch-1 grad forwards — the
# dominant reserved-vs-allocated gap on the HF GPUs. setdefault so an explicit
# override from the environment still wins. This var is inherited by child procs;
# vLLM manages its own allocator and breaks under expandable_segments, so
# vllm_server.start() strips it from the vLLM subprocess env.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import logging
import math
import random
import shutil
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import wandb

# ── Bootstrap ──────────────────────────────────────────────────────────────
from pipeline.utils import setup_logging, set_global_seed, logger, post_with_retry, FILE_LOG_LOCK
from pipeline import (
    inference_script,
    eval_script,
    hint_script,
    state_finder_script,
    probing_script,
    loss_script,
    optimization_script,
    vllm_server,
    utils,
    config,
)

ROOT = Path(__file__).parent
TRAININGSET_PATH = ROOT / "data" / "trainingset.json"
# Each run gets its own timestamped subdirectory so repeated runs don't clobber
# one another's per-epoch/final checkpoints (the paths below are keyed only by
# epoch, not by run). On Modal the base dir is the persistent checkpoints Volume.
_CHECKPOINT_BASE = Path(os.getenv("CHECKPOINT_DIR", str(ROOT / "checkpoints")))
# RUN_ID keys the per-run checkpoint dir. On Modal it is passed in via the
# environment (see modal_app.py) so every automatic retry of a preempted run
# shares one CHECKPOINT_DIR and can resume from its committed epochs; locally it
# defaults to a fresh timestamp per run.
RUN_ID = os.getenv("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
CHECKPOINT_DIR = _CHECKPOINT_BASE / RUN_ID
_REASONING_LOG = ROOT / "data" / "reasoning_log.jsonl"
DIVIDER = "─" * 70

# Epoch-level mean-entropy accumulator (config.LOG_ENTROPY). process_item runs
# inside a worker thread (width-1 on the full-vocab path, but a worker all the
# same), so the running sums are guarded by a lock. Student entropy is collected
# on every trained item; teacher entropy only on the full-vocab path, where the
# teacher forward is full-vocab — hence separate counts. Reset at each epoch start
# by _reset_epoch_entropy(); read out into the epoch-end wandb.log.
_ENTROPY_LOCK = threading.Lock()
_entropy_acc = {"student_sum": 0.0, "student_n": 0, "teacher_sum": 0.0, "teacher_n": 0}


# Epoch-level rollout-length accumulator (config.LENGTH_PENALTY). Read out at the
# epoch boundary next to val accuracy: the token-budget penalty's one real failure
# mode is the model learning to terminate early with a guessed answer, and that
# shows up as mean completion tokens and accuracy falling together. Same threading
# story as _entropy_acc above.
_LENGTH_LOCK = threading.Lock()
_length_acc = {"tokens_sum": 0, "n": 0, "truncated": 0, "near_budget": 0}


def _reset_epoch_lengths() -> None:
    """Zero the epoch rollout-length accumulator (called at the start of each epoch)."""
    with _LENGTH_LOCK:
        _length_acc.update(tokens_sum=0, n=0, truncated=0, near_budget=0)


def _accumulate_length(n_tokens: int, truncated: bool, near_budget: bool) -> None:
    """Record one rollout's length against the epoch's token-budget statistics."""
    with _LENGTH_LOCK:
        _length_acc["tokens_sum"] += n_tokens
        _length_acc["n"] += 1
        _length_acc["truncated"] += int(truncated)
        _length_acc["near_budget"] += int(near_budget)


def _epoch_length_means() -> dict:
    """Snapshot the epoch's rollout-length statistics as wandb.log fields."""
    with _LENGTH_LOCK:
        n = _length_acc["n"]
        if not n:
            return {}
        return {
            "epoch_mean_completion_tokens": _length_acc["tokens_sum"] / n,
            "epoch_truncated_frac": _length_acc["truncated"] / n,
            "epoch_near_budget_frac": _length_acc["near_budget"] / n,
        }


def _reset_epoch_entropy() -> None:
    """Zero the epoch entropy accumulator (called at the start of each epoch)."""
    with _ENTROPY_LOCK:
        _entropy_acc.update(student_sum=0.0, student_n=0, teacher_sum=0.0, teacher_n=0)


def _accumulate_entropy(student: float, teacher: Optional[float] = None) -> None:
    """Add one trained item's mean per-token entropy to the epoch accumulator."""
    with _ENTROPY_LOCK:
        _entropy_acc["student_sum"] += student
        _entropy_acc["student_n"] += 1
        if teacher is not None:
            _entropy_acc["teacher_sum"] += teacher
            _entropy_acc["teacher_n"] += 1


def _epoch_entropy_means() -> dict:
    """Snapshot the epoch's mean student/teacher entropy as wandb.log fields."""
    with _ENTROPY_LOCK:
        out = {}
        if _entropy_acc["student_n"]:
            out["epoch_entropy"] = _entropy_acc["student_sum"] / _entropy_acc["student_n"]
        if _entropy_acc["teacher_n"]:
            out["epoch_teacher_entropy"] = _entropy_acc["teacher_sum"] / _entropy_acc["teacher_n"]
        return out


def _teacher_rollout_full_loss(probe: dict, epoch: int):
    """Select the loss for one full-vocab teacher-rollout probe (run_probing).

    Default: forward KL(teacher ‖ student). With config.TEACHER_ROLLOUT_REVERSE the
    reverse KL(student ‖ teacher) objective is applied to the same teacher-decoded
    tokens — full-vocab (run_reverse_loss) or biased over the teacher's top-k
    (run_reverse_topk_loss) when TEACHER_ROLLOUT_BIASED is set. With
    config.ADAPTIVE_KL (which overrides the above) the loss is the epoch-annealed
    forward+reverse blend, top-k when TEACHER_ROLLOUT_BIASED is set.
    """
    if config.ADAPTIVE_KL:
        fw = config.adaptive_kl_forward_weight(epoch)
        if config.TEACHER_ROLLOUT_BIASED:
            return loss_script.run_adaptive_topk_loss(
                probe["logits_with_hint"], probe["logits_without_hint"],
                config.TEACHER_ROLLOUT_BIASED_K, fw,
            )
        return loss_script.run_adaptive_loss(
            probe["logits_with_hint"], probe["logits_without_hint"], fw
        )
    if config.TEACHER_ROLLOUT_REVERSE:
        if config.TEACHER_ROLLOUT_BIASED:
            return loss_script.run_reverse_topk_loss(
                probe["logits_with_hint"], probe["logits_without_hint"],
                config.TEACHER_ROLLOUT_BIASED_K,
            )
        return loss_script.run_reverse_loss(
            probe["logits_with_hint"], probe["logits_without_hint"]
        )
    return loss_script.run_loss(
        probe["logits_with_hint"], probe["logits_without_hint"]
    )


def _teacher_rollout_biased_loss(p: dict):
    """Select the loss for one buffered top-k teacher-rollout probe (vLLM path).

    Default: forward top-k KL(teacher ‖ student) (run_topk_loss). With
    config.TEACHER_ROLLOUT_REVERSE the biased reverse KL(student ‖ teacher) over the
    teacher's renormalised top-k (run_reverse_topk_loss_from_teacher) — full-vocab
    reverse KL is unavailable on this path (no full teacher distribution), which
    main() enforces by requiring TEACHER_ROLLOUT_BIASED. With config.ADAPTIVE_KL the
    loss is instead the epoch-annealed forward+reverse blend over the teacher's
    renormalised top-k (run_adaptive_topk_loss_from_teacher).
    """
    if config.ADAPTIVE_KL:
        fw = config.adaptive_kl_forward_weight(p["_meta"]["epoch"])
        return loss_script.run_adaptive_topk_loss_from_teacher(
            p["teacher_topk_ids"], p["teacher_topk_logprobs"],
            p["student_logits"], fw, p.get("teacher_topk_mask"),
        )
    if config.TEACHER_ROLLOUT_REVERSE:
        return loss_script.run_reverse_topk_loss_from_teacher(
            p["teacher_topk_ids"], p["teacher_topk_logprobs"],
            p["student_logits"], p.get("teacher_topk_mask"),
        )
    return loss_script.run_topk_loss(
        p["teacher_topk_ids"], p["teacher_topk_logprobs"],
        p["student_logits"], p.get("teacher_topk_mask"),
    )


def _build_length_probe(label: str, epoch: int, step: int, problem: str, inf: dict):
    """Prepare this item's token-budget probe, or None if it doesn't warrant one.

    An item warrants one when its rollout was cut off at the MAX_NEW_TOKENS budget
    (finish_reason == "length") or merely came within config.LENGTH_SOFT_FRAC of
    it. Both cases used to yield nothing: a truncated rollout died in the
    hint/state-finder path (there is no "sentence where the mistake happened" when
    the failure is not finishing), and a near-budget rollout that answered
    correctly was skipped outright. See config.LENGTH_PENALTY.

    Never raises: a missing length signal degrades this item to the pre-existing
    behaviour, which is not worth failing an otherwise fine training item over.
    """
    if not config.LENGTH_PENALTY:
        return None
    truncated = inf.get("finish_reason") == "length"
    budget_frac = float(inf.get("budget_frac") or 0.0)
    if not truncated and budget_frac < config.LENGTH_SOFT_FRAC:
        return None

    token_ids = inf.get("completion_tokens") or []
    if not token_ids:
        logger.warning(
            "%s: rollout used %.0f%% of the token budget but no token ids came "
            "back from vLLM; no length penalty for this item.", label, budget_frac * 100,
        )
        return None
    try:
        probe = probing_script.prepare_length_probe(problem, token_ids)
    except Exception as exc:
        logger.warning("%s: could not prepare the token-budget probe — %s.", label, exc)
        return None
    if probe is None:
        return None

    probe["_meta"] = {
        "label": label, "epoch": epoch, "step": step,
        "budget_frac": budget_frac, "truncated": truncated,
    }
    logger.info(
        "%s: rollout used %d tokens (%.0f%% of budget%s); buffering a token-budget "
        "probe over its last %d position(s).",
        label, probe["n_generated"], budget_frac * 100,
        ", TRUNCATED" if truncated else "", probe["_n_completion"],
    )
    return probe


def _log_correct_reasoning(reasoning: str) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "correct": True,
        "full_reasoning": reasoning,
        "full_len": len(reasoning),
    }
    try:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        _REASONING_LOG.parent.mkdir(parents=True, exist_ok=True)
        # Lock so concurrent worker threads can't interleave lines.
        with FILE_LOG_LOCK, _REASONING_LOG.open("a", encoding="utf-8") as fh:
            fh.write(line)
    except Exception as exc:
        logger.warning("Failed to write correct reasoning to log: %s", exc)


def _guard_env() -> None:
    missing = []
    if not config.OPENROUTER_API_KEY:
        missing.append("OPENROUTER_API_KEY")
    if missing:
        sys.exit(f"Error: missing required environment variable(s): {', '.join(missing)}")
    # Validated here rather than at first use: run_hint raises on an unknown mode,
    # and since that happens per item, a typo'd HINT_MODE would otherwise error its
    # way through the entire dataset — one failed item at a time, after paying for
    # every inference call — instead of failing before the run starts.
    if config.HINT_MODE not in hint_script.VALID_CONFIG_HINT_MODES:
        sys.exit(
            f"Error: HINT_MODE={config.HINT_MODE!r} is not a valid mode; expected "
            f"one of {', '.join(hint_script.VALID_CONFIG_HINT_MODES)}."
        )


def _preflight_vllm_topk() -> None:
    """Verify the vLLM server supports the top-k probing path before training.

    Exercises the EXACT request shape the probing paths use — /v1/completions with
    a raw token-id prompt, `logprobs: K` (the count of alternatives; the chat
    endpoint's logprobs+top_logprobs pair does not exist here), stop_token_ids, and
    return_tokens_as_token_ids — rather than a chat request that merely resembles
    it. The two endpoints validate and shape their logprobs differently, so a
    server that answers one can still reject the other, and a preflight against the
    wrong one passes and then fails on training item 1, deep into the run.

    Checks that the server:
      • accepts a token-id prompt and stop_token_ids,
      • returns K alternatives per position (requires --max-logprobs >= K), and
      • honors return_tokens_as_token_ids in BOTH `tokens` and the `top_logprobs`
        keys — the pipeline parses ids out of both.
    """
    k = config.PROBE_TOPK_K
    url = f"{config.VLLM_BASE_URL}/completions"
    headers = {
        "Authorization": f"Bearer {config.VLLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.VLLM_MODEL,
        # A token-id prompt, like the probe sends. Content is irrelevant (we ask
        # for one token); the point is that the server accepts the ids form.
        "prompt": [1, 2, 3],
        "temperature": 0.0,
        "max_tokens": 1,
        "logprobs": k,
        "stop_token_ids": [],
        "return_tokens_as_token_ids": True,
    }
    logger.info("Preflight: verifying vLLM top-k probing support (k=%d) at %s …", k, url)
    hint = (
        f"The top-k probing path needs the server reachable at "
        f"{config.VLLM_BASE_URL}, started with --max-logprobs >= {k} "
        f"(PROBE_TOPK_K), and able to serve /v1/completions with token-id prompts. "
        f"Start it per setup.sh, or set PROBE_TOPK_VIA_VLLM=0 to use the full-vocab "
        f"HF probing path."
    )
    try:
        resp = post_with_retry(url, headers=headers, payload=payload, timeout=60,
                               label="vLLM-preflight")
        lp = resp.json()["choices"][0]["logprobs"]
        token = lp["tokens"][0]
        alts = (lp.get("top_logprobs") or [{}])[0] or {}
    except RuntimeError as exc:
        sys.exit(f"vLLM preflight failed: {exc}\n{hint}")
    except (KeyError, IndexError, TypeError) as exc:
        sys.exit(
            f"vLLM preflight got an unexpected /v1/completions response shape "
            f"({exc}); the server did not return logprobs.tokens. {hint}"
        )
    bad = [t for t in [token, *alts.keys()]
           if not (isinstance(t, str) and t.startswith("token_id:"))]
    if bad:
        sys.exit(
            f"vLLM preflight: logprob token {bad[0]!r} is not in 'token_id:N' form, "
            f"so return_tokens_as_token_ids is not in effect. The probe cannot align "
            f"the teacher's top-k with the HF vocabulary without it — upgrade/"
            f"configure the vLLM server, or set PROBE_TOPK_VIA_VLLM=0."
        )
    if len(alts) < k:
        # /v1/completions reports k (+1 for the sampled token) alternatives; fewer
        # means the server silently clamped to its own --max-logprobs.
        logger.warning(
            "Preflight: asked for %d teacher alternatives but the server returned "
            "%d — it is likely clamping to --max-logprobs. The probe will train on "
            "a narrower teacher distribution than PROBE_TOPK_K asks for.",
            k, len(alts),
        )
    logger.info("Preflight OK: vLLM returns token-id top-%d logprobs on /v1/completions.", k)


def _preflight_length_penalty() -> None:
    """Verify the CHAT endpoint returns token-id logprobs before training.

    The token-budget penalty teacher-forces a rollout's tail on the HF model, so
    it needs the ids vLLM generated — which only come back from
    /v1/chat/completions with logprobs + return_tokens_as_token_ids. That is a
    different endpoint from the one _preflight_vllm_topk exercises, and the two
    validate their logprobs arguments separately, so a server that satisfies one
    can still reject the other.

    Checked up front rather than per item because the failure modes are opposite
    and both are bad late: a server that REJECTS the flags fails every inference
    call (i.e. the whole run), and one that IGNORES them silently drops the
    objective for the entire run while looking healthy.
    """
    url = f"{config.VLLM_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.VLLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": inference_script.get_active_model() or config.VLLM_MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 0,
        "return_tokens_as_token_ids": True,
    }
    hint = (
        "The token-budget penalty (LENGTH_PENALTY=1) needs /v1/chat/completions to "
        "accept logprobs + return_tokens_as_token_ids and report tokens as "
        "'token_id:N'. Upgrade the vLLM server, or set LENGTH_PENALTY=0 to train "
        "without the length term."
    )
    logger.info("Preflight: verifying chat-endpoint token-id logprobs at %s …", url)
    try:
        resp = post_with_retry(url, headers=headers, payload=payload, timeout=60,
                               label="vLLM-length-preflight")
        content = ((resp.json()["choices"][0].get("logprobs") or {}).get("content")) or []
    except RuntimeError as exc:
        sys.exit(f"Token-budget preflight failed: {exc}\n{hint}")
    except (KeyError, IndexError, TypeError) as exc:
        sys.exit(f"Token-budget preflight got an unexpected response shape ({exc}). {hint}")
    token = (content[0] or {}).get("token") if content else None
    if not (isinstance(token, str) and token.startswith("token_id:")):
        sys.exit(
            f"Token-budget preflight: chat logprob token {token!r} is not in "
            f"'token_id:N' form, so the rollout's tokens cannot be aligned with the "
            f"HF vocabulary. {hint}"
        )
    logger.info("Preflight OK: chat endpoint returns token-id logprobs.")


def process_item(
    epoch: int, idx: int, total: int, item: dict
) -> tuple[str, Optional[dict], Optional[dict]]:
    """Process one training item.

    Returns (outcome, probe, length_probe). outcome is one of 'skipped' (answer
    correct), 'trained', 'arithmetic' (answer wrong but the mistake is a
    calculation/rounding slip, so training is skipped), 'overlong' (the rollout
    ran into the token budget, so there is no located mistake to distil — only the
    length term applies), 'error', or 'buffered'.

    This function does the per-item network work only (inference, eval, hint,
    state-finder, and the top-k teacher decode) and touches no shared training
    state, so it is safe to run concurrently across items. On the default top-k
    path the GPU-heavy student forward is deferred: the prepared probe is
    returned alongside the 'buffered' outcome, and the caller (on the main
    thread) buffers it for flush_training_batch() to run the batched forward +
    backward, at which point the item is counted as 'trained'. The full-vocab
    path trains the item immediately (per-item forward/backward) and returns
    ('trained', None, …).

    length_probe is deferred the same way (flush_length_batch) and is independent
    of the outcome — a correct-but-near-budget item returns ('skipped', None,
    probe) and still trains its length term. See config.LENGTH_PENALTY.
    """
    label = f"Epoch {epoch} | Item {idx}/{total} (#{item.get('number', idx)})"
    step = (epoch - 1) * total + idx

    def _log_error(stage: str) -> None:
        wandb.log({"outcome": "error", "failed_stage": stage, "epoch": epoch, "step": step})

    problem = item.get("problem")
    ground_truth = item.get("answer")
    if not problem or ground_truth is None:
        logger.error("%s: Malformed item (missing 'problem' or 'answer'). Skipping.", label)
        _log_error("validation")
        return "error", None, None
    problem, ground_truth = str(problem), str(ground_truth)

    # ── Step 1: Inference ─────────────────────────────────────────────────
    logger.info("%s: Running inference …", label)
    try:
        inf = inference_script.run_inference(problem)
    except Exception as exc:
        logger.error("%s: Inference failed — %s. Skipping item.", label, exc, exc_info=True)
        _log_error("inference")
        return "error", None, None

    reasoning: str = inf["reasoning"]
    generated_answer: str = inf["generated_answer"]
    logger.info("%s: Generated answer → %s", label, generated_answer[:120])

    # ── Step 1b: Token budget ─────────────────────────────────────────────
    # Prepared before the grader runs, because it applies whether or not the
    # answer was right — running the budget down is its own failure.
    truncated_by_budget: bool = inf.get("finish_reason") == "length"
    budget_frac = float(inf.get("budget_frac") or 0.0)
    _accumulate_length(
        int(inf.get("n_completion_tokens") or 0),
        truncated_by_budget,
        budget_frac >= config.LENGTH_SOFT_FRAC,
    )
    len_probe = _build_length_probe(label, epoch, step, problem, inf)

    # ── Step 2: Evaluation ────────────────────────────────────────────────
    logger.info("%s: Evaluating answer …", label)
    try:
        is_correct: bool = eval_script.run_eval(generated_answer, ground_truth)
    except Exception as exc:
        logger.error("%s: Evaluation failed — %s. Skipping item.", label, exc, exc_info=True)
        _log_error("evaluation")
        return "error", None, len_probe

    if is_correct:
        logger.info("%s: Answer CORRECT. No training needed.", label)
        _log_correct_reasoning(reasoning)
        return "skipped", None, len_probe

    logger.info("%s: Answer INCORRECT. Beginning training steps …", label)

    # A rollout cut off at the budget has no located mistake to distil: it was not
    # reasoned wrongly, it never finished. Sending it down the hint → state-finder
    # path asks the hint model to quote "the sentence where the mistake happened"
    # from a trace that does not contain one, and the unanchored quote is then
    # rejected downstream (or, worse, accepted at a meaningless cut point). Route
    # it to the length term alone.
    if truncated_by_budget:
        logger.info(
            "%s: Rollout was truncated at the %d-token budget; training the "
            "token-budget term only (no hint/state-finder).",
            label, config.MAX_NEW_TOKENS,
        )
        wandb.log({"outcome": "overlong", "epoch": epoch, "step": step,
                   "budget_frac": budget_frac})
        return "overlong", None, len_probe

    # ── Step 3: Hint generation ───────────────────────────────────────────
    # Use the full reasoning (including native channel) so the hint model has
    # maximum context about the mistake.
    logger.info("%s: Generating hint …", label)
    hint_mode = hint_script.resolve_hint_mode(
        config.HINT_MODE, epoch, config.NUM_EPOCHS
    )
    try:
        hint, quoted_sentence, error_type = hint_script.run_hint(
            problem, reasoning, generated_answer, ground_truth, mode=hint_mode
        )
    except Exception as exc:
        logger.error("%s: Hint generation failed — %s. Skipping item.", label, exc, exc_info=True)
        _log_error("hint")
        return "error", None, len_probe
    logger.info("%s: Hint → %s", label, hint[:200])

    # Arithmetic/rounding slips have correct reasoning, so there is no conceptual
    # lesson to backpropagate. Skip training on them (still counted as incorrect),
    # but record that we saw one so the per-epoch tally can be logged.
    if config.SKIP_ARITHMETIC_ERRORS and error_type == hint_script.ARITHMETIC_ERROR_TYPE:
        logger.info(
            "%s: Mistake classified as arithmetic/rounding. Skipping training (still counted wrong).",
            label,
        )
        wandb.log({"outcome": "arithmetic", "epoch": epoch, "step": step})
        return "arithmetic", None, len_probe

    # ── Step 4: State finding (truncate reasoning at mistake) ─────────────
    logger.info("%s: Locating mistake in reasoning chain …", label)
    try:
        truncated_reasoning: str = state_finder_script.run_state_finder(
            reasoning, hint, quoted_sentence
        )
    except Exception as exc:
        logger.error("%s: State finding failed — %s. Skipping item.", label, exc, exc_info=True)
        _log_error("state_finder")
        return "error", None, len_probe
    logger.info(
        "%s: Truncated reasoning: %d → %d chars.",
        label, len(reasoning), len(truncated_reasoning),
    )

    # ── Step 5a: On-policy reverse-KL student-rollout mode ────────────────
    # Sample NUM_STUDENT_ROLLOUTS completions from the student's own (without-hint)
    # prefix, then minimise reverse KL(student ‖ teacher) over each. Each rollout is
    # one gradient-accumulation micro-batch. Runs the full-vocab HF forwards inline
    # (so this path stays serial — see the width logic in main()), bypassing both
    # the top-k buffered path and the default full-vocab forward-KL path below.
    if config.STUDENT_ROLLOUT:
        _srmode = (
            "adaptive-KL" if config.ADAPTIVE_KL
            else "forward-KL" if config.STUDENT_ROLLOUT_FORWARD
            else "reverse-KL"
        )
        logger.info("%s: Sampling %d student rollouts (%s mode) …",
                    label, config.NUM_STUDENT_ROLLOUTS, _srmode)
        try:
            # Per-item (and per-epoch) seed: reproducible across runs, yet distinct
            # per item/epoch so rollouts stay diverse rather than collapsing to one
            # fixed sample. None when seeding is disabled (config.SEED < 0).
            rollout_seed = (
                None if config.SEED < 0
                else config.SEED + epoch * 1_000_003 + idx
            )
            sampled = probing_script.sample_student_rollouts(
                problem, truncated_reasoning, hint, seed=rollout_seed
            )
        except Exception as exc:
            logger.error("%s: Student-rollout probing failed — %s. Skipping item.",
                         label, exc, exc_info=True)
            _log_error("probing")
            return "error", None, len_probe

        completions = sampled["completions"]
        teacher_logits = sampled["teacher_logits"]
        reference_logits = sampled.get("reference_logits")
        n_rollouts = len(completions)
        try:
            # One problem is one accumulation micro-batch: its gradient is the
            # mean of its rollouts' (per-token-mean) gradients. Each rollout is
            # backpropped with scale=1/n_rollouts so the N gradients average, and
            # the micro-batch is committed once after the loop — so GRAD_ACCUM_STEPS
            # counts problems, not rollouts, and the effective batch is
            # GRAD_ACCUM_STEPS problems regardless of how many rollouts each yields.
            rollout_losses = []
            for r_idx, (completion_ids, t_logits) in enumerate(
                zip(completions, teacher_logits), 1
            ):
                # Student-forward one rollout at a time and backprop it before the
                # next forward, so only a single student autograd graph is ever
                # resident on the GPU (the on-policy OOM guard — see
                # probing_script.student_rollout_logits). The teacher forwards were
                # already batched up front in sample_student_rollouts.
                rollout = probing_script.student_rollout_logits(
                    sampled, completion_ids, t_logits
                )
                n_tokens = len(rollout["completion_tokens"])
                if config.EXOPD:
                    kl_loss = loss_script.run_exopd_loss(
                        rollout["logits_with_hint"], rollout["logits_without_hint"],
                        reference_logits[r_idx - 1], config.EXOPD_LAMBDA,
                    )
                elif config.ADAPTIVE_KL:
                    fw = config.adaptive_kl_forward_weight(epoch)
                    if config.STUDENT_ROLLOUT_BIASED:
                        kl_loss = loss_script.run_adaptive_topk_loss(
                            rollout["logits_with_hint"], rollout["logits_without_hint"],
                            config.STUDENT_ROLLOUT_BIASED_K, fw,
                        )
                    else:
                        kl_loss = loss_script.run_adaptive_loss(
                            rollout["logits_with_hint"], rollout["logits_without_hint"], fw,
                        )
                elif config.STUDENT_ROLLOUT_FORWARD:
                    # Forward KL(teacher ‖ student) on the student-sampled tokens.
                    # Full-vocab (the teacher's full with-hint distribution is
                    # already materialised per rollout), so STUDENT_ROLLOUT_BIASED is
                    # ignored here — see config.STUDENT_ROLLOUT_FORWARD.
                    kl_loss = loss_script.run_loss(
                        rollout["logits_with_hint"], rollout["logits_without_hint"]
                    )
                elif config.STUDENT_ROLLOUT_BIASED:
                    kl_loss = loss_script.run_reverse_topk_loss(
                        rollout["logits_with_hint"], rollout["logits_without_hint"],
                        config.STUDENT_ROLLOUT_BIASED_K,
                    )
                else:
                    kl_loss = loss_script.run_reverse_loss(
                        rollout["logits_with_hint"], rollout["logits_without_hint"]
                    )
                loss_val = optimization_script.accumulate_loss(
                    kl_loss, scale=1.0 / n_rollouts
                )
                rollout_losses.append(loss_val)
                # Drop references to this rollout's (N, V) tensors / freed graph so
                # the memory is reclaimable before the next rollout's forward.
                del rollout, kl_loss
                logger.info(
                    "%s: Rollout %d/%d %s = %.6f over %d tokens.",
                    label, r_idx, n_rollouts, _srmode, loss_val, n_tokens,
                )

            # Commit the problem as one micro-batch; steps if the window is full.
            grad_norm = optimization_script.commit_micro_batch()
            mean_loss = sum(rollout_losses) / len(rollout_losses)
            if grad_norm is not None:
                logger.info(
                    "%s: Problem mean %s = %.6f over %d rollouts. "
                    "Weights updated (|grad|=%.4f).",
                    label, _srmode, mean_loss, n_rollouts, grad_norm,
                )
            metrics = {
                "loss": mean_loss,
                "n_rollouts": n_rollouts,
                "epoch": epoch,
                "step": step,
            }
            if grad_norm is not None:
                metrics["grad_norm"] = grad_norm
            wandb.log(metrics)
        except Exception as exc:
            logger.error("%s: Reverse-KL optimisation failed — %s. Skipping item.",
                         label, exc, exc_info=True)
            _log_error("optimization")
            return "error", None, len_probe
        return "trained", None, len_probe

    # ── Step 5: Probing (extract aligned logits) ──────────────────────────
    # Two modes: the default top-k path (teacher decoded + top-k logprobs from
    # vLLM, student forward deferred so several items batch into one HF forward),
    # or the full-vocab HF path (teacher + student forwards on HF, per item).
    logger.info("%s: Running probing (extracting logits) …", label)
    try:
        if config.PROBE_TOPK_VIA_VLLM:
            # Defer the student forward: hand the prepared probe back so the main
            # thread can buffer a batch that shares one HF forward + backward in
            # flush_training_batch().
            probe = probing_script.prepare_probing_topk(problem, truncated_reasoning, hint)
            probe["_meta"] = {
                "label": label, "epoch": epoch, "step": step,
                "n_tokens": len(probe["completion_tokens"]),
            }
            logger.info(
                "%s: Probe prepared over %d completion tokens; buffering for batch.",
                label, probe["_meta"]["n_tokens"],
            )
            return "buffered", probe, len_probe
        probe = probing_script.run_probing(problem, truncated_reasoning, hint)
    except Exception as exc:
        logger.error("%s: Probing failed — %s. Skipping item.", label, exc, exc_info=True)
        _log_error("probing")
        return "error", None, len_probe

    n_tokens = len(probe["completion_tokens"])
    logger.info("%s: Logits extracted over %d completion tokens.", label, n_tokens)

    # ── Step 6: KL divergence loss (full-vocab path) ──────────────────────
    # Forward KL(teacher ‖ student) by default; reverse / biased-reverse
    # KL(student ‖ teacher) over the same teacher-decoded tokens when
    # config.TEACHER_ROLLOUT_REVERSE is set (see _teacher_rollout_full_loss).
    logger.info("%s: Computing KL divergence loss …", label)
    try:
        kl_loss = _teacher_rollout_full_loss(probe, epoch)
    except Exception as exc:
        logger.error("%s: Loss computation failed — %s. Skipping item.", label, exc, exc_info=True)
        _log_error("loss")
        return "error", None, len_probe
    logger.info("%s: KL loss = %.6f", label, kl_loss.item())

    # ── Step 7: Optimisation (full-vocab path) ────────────────────────────
    logger.info("%s: Backpropagating loss …", label)
    try:
        loss_val, grad_norm = optimization_script.run_optimization(kl_loss)
    except Exception as exc:
        logger.error("%s: Optimisation failed — %s. Skipping item.", label, exc, exc_info=True)
        _log_error("optimization")
        return "error", None, len_probe

    if grad_norm is not None:
        logger.info(
            "%s: Loss calculated: %.6f. Weights updated successfully.", label, loss_val
        )
    else:
        logger.info(
            "%s: Loss calculated: %.6f. Gradient accumulated.", label, loss_val
        )
    metrics = {
        "loss": loss_val,
        "kl_tokens": n_tokens,
        "epoch": epoch,
        "step": step,
    }
    if config.LOG_ENTROPY:
        student_h = loss_script.token_entropy(probe["logits_without_hint"])
        teacher_h = loss_script.token_entropy(probe["logits_with_hint"])
        metrics["entropy"] = student_h
        metrics["teacher_entropy"] = teacher_h
        _accumulate_entropy(student_h, teacher_h)
    if grad_norm is not None:
        metrics["grad_norm"] = grad_norm
    wandb.log(metrics)
    return "trained", None, len_probe


def flush_training_batch(pending: list) -> tuple[int, int]:
    """Run the batched student forward + loss + backward for buffered probes.

    Consumes (and clears) `pending` — the items accumulated by process_item on
    the top-k path — running ONE batched HF student forward, the top-k KL per
    item, and a single backward over the whole batch.

    Returns (n_trained, n_failed): the whole batch is one unit, so a failure means
    every item in it went untrained. They are reported as errors rather than
    silently vanishing — a dropped batch used to leave those items counted in
    neither `trained` nor `error`, so an epoch that lost batches still looked like
    a clean run in the tally and in W&B.
    """
    if not pending:
        return 0, 0

    batch = list(pending)
    pending.clear()
    n = len(batch)
    labels = ", ".join(p["_meta"]["label"] for p in batch)

    try:
        probing_script.student_forward_topk_batch(batch)
        kl_losses = [_teacher_rollout_biased_loss(p) for p in batch]
        for p, kl in zip(batch, kl_losses):
            logger.info("%s: KL loss = %.6f", p["_meta"]["label"], kl.item())
        mean_loss, grad_norm = optimization_script.run_optimization_batch(kl_losses)
    except Exception as exc:
        logger.error(
            "Batched optimisation failed for %d item(s) [%s] — %s. Dropping batch; "
            "counting them as errors.",
            n, labels, exc, exc_info=True,
        )
        for p in batch:
            m = p["_meta"]
            wandb.log({"outcome": "error", "failed_stage": "batched_optimization",
                       "epoch": m["epoch"], "step": m["step"]})
        return 0, n

    if grad_norm is not None:
        logger.info(
            "Batch of %d item(s) trained; mean loss=%.6f. Weights updated (|grad|=%.4f).",
            n, mean_loss, grad_norm,
        )
    else:
        logger.info(
            "Batch of %d item(s) trained; mean loss=%.6f. Gradient accumulated.",
            n, mean_loss,
        )

    # Per-item loss metrics (each item carries its own step), plus one grad_norm
    # point for the step this batch may have triggered.
    for p, kl in zip(batch, kl_losses):
        m = p["_meta"]
        item_metrics = {
            "loss": float(kl.item()),
            "kl_tokens": m["n_tokens"],
            "epoch": m["epoch"],
            "step": m["step"],
        }
        if config.LOG_ENTROPY:
            # Only the student forward is full-vocab on this path; the teacher is
            # top-k logprobs only, so no teacher entropy here.
            student_h = loss_script.token_entropy(p["student_logits"])
            item_metrics["entropy"] = student_h
            _accumulate_entropy(student_h)
        wandb.log(item_metrics)
    if grad_norm is not None:
        wandb.log({"grad_norm": grad_norm, "lr": optimization_script.current_lr()})
    return n, 0


def flush_length_batch(pending: list) -> tuple[int, int]:
    """Run the student forward + token-budget penalty for buffered length probes.

    The length-term analogue of flush_training_batch. Consumes (and clears)
    `pending`, running one batched forward over the buffered rollouts, the
    per-item penalty (loss_script.run_length_penalty), and a single backward
    weighted by config.LENGTH_PENALTY_WEIGHT.

    Accumulation accounting: this counts as ONE micro-batch, exactly like a KL
    batch, so an item that produces both a KL probe and a length probe contributes
    two micro-batches. GRAD_ACCUM_STEPS therefore counts micro-batches rather than
    problems once the length term is on — the per-step gradient stays a mean over
    equally-weighted micro-batches either way, which is what governs step size.

    Returns (n_trained, n_failed); like the KL flush, a failed batch takes every
    item in it down rather than letting them vanish from the tally.
    """
    if not pending:
        return 0, 0

    batch = list(pending)
    pending.clear()
    n = len(batch)
    labels = ", ".join(p["_meta"]["label"] for p in batch)

    try:
        probing_script.student_forward_batch(batch)
        losses = [
            loss_script.run_length_penalty(
                p["student_logits"], p["end_token_ids"], p["position_weights"]
            )
            for p in batch
        ]
        for p, l in zip(batch, losses):
            logger.info(
                "%s: token-budget loss = %.6f over %d position(s) (%.0f%% of budget).",
                p["_meta"]["label"], l.item(), p["_n_completion"],
                p["_meta"]["budget_frac"] * 100,
            )
        # Weight applied here, not inside the loss, so the raw −log p(end) stays
        # comparable across runs in the logs while only the gradient is scaled.
        _, grad_norm = optimization_script.run_optimization_batch(
            [l * config.LENGTH_PENALTY_WEIGHT for l in losses]
        )
    except Exception as exc:
        logger.error(
            "Token-budget optimisation failed for %d item(s) [%s] — %s. Dropping batch.",
            n, labels, exc, exc_info=True,
        )
        return 0, n

    for p, l in zip(batch, losses):
        m = p["_meta"]
        wandb.log({
            "length_loss": float(l.item()),
            "length_positions": p["_n_completion"],
            "budget_frac": m["budget_frac"],
            "epoch": m["epoch"],
            "step": m["step"],
        })
    if grad_norm is not None:
        wandb.log({"grad_norm": grad_norm, "lr": optimization_script.current_lr()})
    return n, 0


def validate_item(epoch: int, val_idx: int, n_val: int, item: dict) -> str:
    """Run one validation item (inference + eval); return 'correct'/'wrong'/'error'.

    Validation does no GPU training (only the vLLM rollout and OpenRouter eval),
    so it is safe to run concurrently across items — the backend concurrency caps
    in utils still bound the load on each server.
    """
    problem = item.get("problem")
    ground_truth = item.get("answer")
    if not problem or ground_truth is None:
        return "error"
    try:
        inf = inference_script.run_inference(str(problem))
        is_correct = eval_script.run_eval(inf["generated_answer"], str(ground_truth))
        return "correct" if is_correct else "wrong"
    except Exception as exc:
        logger.warning("Epoch %d | Val item %d/%d: failed — %s", epoch, val_idx, n_val, exc)
        return "error"


def _find_resume_epoch(base: Path) -> Optional[int]:
    """Return the highest fully-committed epoch in `base`, or None if fresh.

    Trusts only epoch dirs that carry all of: the `DONE` sentinel (written last,
    so a half-written/uncommitted epoch is never selected — see main()), the
    saved optimizer state, and the adapter config. Ignores the `*_merged` dirs
    (those are matched to their epoch by name at restore time).
    """
    if not base.exists():
        return None
    done: list[int] = []
    for d in base.glob("epoch_*"):
        if d.name.endswith("_merged"):
            continue
        try:
            n = int(d.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if (d / "DONE").exists() and (d / "optim_state.pt").exists() \
                and (d / "adapter_config.json").exists():
            done.append(n)
    return max(done) if done else None


def _log_resolved_arm() -> None:
    """Print the knobs that define which arm this run is, before training starts.

    Config comes from three layers (shell env, configs/ preset, .env), so the only
    reliable record of what a run actually was is the resolved values. Logging them
    here means a run's own output proves its arm rather than the operator's memory
    of which preset they passed.
    """
    hint_writer = "self (frozen teacher)" if config.HINT_VIA_VLLM else config.HINT_MODEL
    if config.STUDENT_ROLLOUT:
        direction = "forward KL" if config.STUDENT_ROLLOUT_FORWARD else "reverse KL"
        source = "student rollouts"
    else:
        direction = "reverse KL" if config.TEACHER_ROLLOUT_REVERSE else "forward KL"
        source = "teacher rollouts"
    logger.info(
        "Resolved arm | hint=%s writer=%s | %s over %s | L=%d | "
        "warmup=%d accum=%d(eff batch %d) lr=%g decay_floor=%g | epochs=%d seed=%d | "
        "length_penalty=%s",
        config.HINT_MODE, hint_writer, direction, source, config.MAX_PROBE_TOKENS,
        config.NUM_WARMUP_STEPS, config.GRAD_ACCUM_STEPS,
        config.TRAIN_BATCH_SIZE * config.GRAD_ACCUM_STEPS,
        config.LEARNING_RATE, config.LR_DECAY_FLOOR, config.NUM_EPOCHS, config.SEED,
        "on" if config.LENGTH_PENALTY else "off",
    )
    if config.HINT_MODE == "vague":
        logger.info("  ^ HINT_MODE=vague is the PLACEBO arm (paper §5.3), not Fin-SD.")


def main(on_checkpoint: Callable[[], None] = lambda: None) -> None:
    # `on_checkpoint` is called after each epoch's (and the final) adapter/merged
    # checkpoint is written. Defaults to a no-op for local runs; modal_app.py
    # passes the checkpoints Volume's `.commit` so per-epoch artifacts are made
    # durable mid-run rather than only at container exit.
    setup_logging()
    set_global_seed(config.SEED)
    _guard_env()
    logger.info("Run %s: checkpoints → %s", RUN_ID, CHECKPOINT_DIR)

    # Resume detection: on an automatic Modal retry the same RUN_ID yields the
    # same CHECKPOINT_DIR, so pick up from the highest fully-committed epoch.
    # vLLM's served weights are restored here; the HF model + optimizer are
    # restored below — so both sides of the train/serve split reflect the resumed
    # epoch before any item runs (else rollouts/validation would run on base
    # weights while the student carries the trained adapter).
    resume_from = _find_resume_epoch(CHECKPOINT_DIR)
    resume_dir = CHECKPOINT_DIR / f"epoch_{resume_from}" if resume_from else None

    # The previous epoch's merged checkpoint / hot-loaded adapter name, kept so
    # the next epoch boundary can clean up (merged) or unload (adapter) the prior
    # one. Seeded from the resumed epoch so post-resume cleanup matches steady state.
    prev_merged_dir: Optional[str] = None
    prev_adapter_name: Optional[str] = None

    # Bring up the pipeline-owned vLLM server before anything talks to it. Fresh
    # run → base model; resume → the resumed epoch's trained weights (expert LoRA
    # restarts on the merged checkpoint; hot-load LoRA serves base + the epoch's
    # hot-loaded adapter).
    # HF_MODEL_PATH (what we train) and VLLM_MODEL (what we serve) are set
    # independently, and VLLM_MODEL keeps its 20B default unless overridden — so
    # pointing HF_MODEL_PATH at the 120B and forgetting the other silently trains
    # one model while serving another. On the hot-load path that ends in a shape
    # error when the adapter meets the base; on the merge path the served *name*
    # merely lies about what is running. Either way the run is misconfigured.
    if not utils.same_model_family(config.HF_MODEL_PATH, config.VLLM_MODEL):
        logger.warning(
            "HF_MODEL_PATH is %r (%s) but VLLM_MODEL is %r (%s) — training and "
            "serving are pointed at different model sizes. Set VLLM_MODEL to the "
            "release matching HF_MODEL_PATH.",
            config.HF_MODEL_PATH, utils.model_family(config.HF_MODEL_PATH),
            config.VLLM_MODEL, utils.model_family(config.VLLM_MODEL),
        )
    # Tear the server down on EVERY exit path — normal return, Ctrl-C, and the
    # SIGTERM/SIGHUP that would otherwise kill us without running atexit and
    # orphan the vLLM subprocess (it lives in its own session and never gets the
    # signal). Shared with the serve tools so all three teardown paths agree.
    vllm_server.install_shutdown_handlers()
    if resume_from and config.LORA_TARGET_EXPERTS:
        prev_merged_dir = str(CHECKPOINT_DIR / f"epoch_{resume_from}_merged")
        vllm_server.start(prev_merged_dir)
    else:
        vllm_server.start(config.VLLM_MODEL)
        if resume_from and not config.LORA_TARGET_EXPERTS:
            prev_adapter_name = f"epoch_{resume_from}"
            vllm_server.load_adapter(prev_adapter_name, str(resume_dir))
            inference_script.set_active_model(prev_adapter_name)
    # Both the top-k teacher path and the (vLLM-sampled) student-rollout path read
    # token-id logprobs from vLLM, so preflight the server whenever either is on.
    # (Student rollouts only need return_tokens_as_token_ids, not max-logprobs>=K,
    # but the shared preflight is harmless given setup.sh starts vLLM with both.)
    if config.PROBE_TOPK_VIA_VLLM or config.STUDENT_ROLLOUT:
        _preflight_vllm_topk()
    # The token-budget penalty reads its ids off the CHAT endpoint instead, which
    # validates logprobs arguments separately — hence its own preflight.
    if config.LENGTH_PENALTY:
        _preflight_length_penalty()

    # Reverse-KL on teacher rollouts: validate the objective is reachable on the
    # active probing path before training (see config.TEACHER_ROLLOUT_REVERSE).
    if config.TEACHER_ROLLOUT_REVERSE:
        if config.STUDENT_ROLLOUT:
            logger.warning(
                "TEACHER_ROLLOUT_REVERSE is set but STUDENT_ROLLOUT is on; the "
                "student-rollout path runs instead, so TEACHER_ROLLOUT_REVERSE has "
                "no effect. Set STUDENT_ROLLOUT=0 to train on teacher rollouts."
            )
        elif config.PROBE_TOPK_VIA_VLLM and not config.TEACHER_ROLLOUT_BIASED:
            sys.exit(
                "TEACHER_ROLLOUT_REVERSE needs TEACHER_ROLLOUT_BIASED=1 on the default "
                "vLLM top-k probing path: vLLM returns only the teacher's top-k, not "
                "its full distribution, so full-vocab reverse KL is unavailable. Set "
                "TEACHER_ROLLOUT_BIASED=1 (biased reverse KL), or PROBE_TOPK_VIA_VLLM=0 "
                "for the full-vocab HF path (which supports both)."
            )

    # Forward KL on student rollouts: a fixed-direction student-rollout objective.
    # It only runs on the student-rollout path, is full-vocab (so STUDENT_ROLLOUT_BIASED
    # does not apply), and is overridden by EXOPD / ADAPTIVE_KL (their own objectives).
    if config.STUDENT_ROLLOUT_FORWARD:
        if not config.STUDENT_ROLLOUT:
            logger.warning(
                "STUDENT_ROLLOUT_FORWARD is set but STUDENT_ROLLOUT is off; it only "
                "affects the student-rollout path and has no effect here. Set "
                "STUDENT_ROLLOUT=1 to train forward KL on student rollouts."
            )
        elif config.EXOPD or config.ADAPTIVE_KL:
            logger.warning(
                "STUDENT_ROLLOUT_FORWARD is ignored: %s is on and is its own "
                "objective (STUDENT_ROLLOUT_FORWARD only selects a fixed KL "
                "direction).", "EXOPD" if config.EXOPD else "ADAPTIVE_KL",
            )
        elif config.STUDENT_ROLLOUT_BIASED:
            logger.warning(
                "STUDENT_ROLLOUT_FORWARD is on and full-vocab; STUDENT_ROLLOUT_BIASED "
                "is ignored (no top-k forward-KL variant on student rollouts)."
            )

    # EXOPD is defined over student rollouts (y~π_θ), so it only runs on the
    # student-rollout path; the full-vocab reference forward it adds also has no
    # top-k variant, so STUDENT_ROLLOUT_BIASED is ignored under EXOPD.
    if config.EXOPD:
        if not config.STUDENT_ROLLOUT:
            sys.exit(
                "EXOPD requires STUDENT_ROLLOUT=1: the objective is defined over "
                "student rollouts (y~π_θ). Set STUDENT_ROLLOUT=1."
            )
        if config.STUDENT_ROLLOUT_BIASED:
            logger.warning(
                "EXOPD is on and full-vocab; STUDENT_ROLLOUT_BIASED is ignored "
                "(no top-k EXOPD variant)."
            )

    # Adaptive-KL blends both directions on a schedule, so it is its own objective:
    # it overrides the fixed-direction TEACHER_ROLLOUT_REVERSE and is exclusive with
    # EXOPD. It works on every probing path (the vLLM top-k path uses the teacher's
    # top-k representation), so no path-specific guard is needed here.
    if config.ADAPTIVE_KL:
        if config.EXOPD:
            sys.exit(
                "ADAPTIVE_KL and EXOPD are mutually exclusive objectives; enable "
                "only one."
            )
        if config.TEACHER_ROLLOUT_REVERSE:
            logger.warning(
                "ADAPTIVE_KL is on; TEACHER_ROLLOUT_REVERSE is ignored (adaptive "
                "blends forward and reverse KL on an epoch schedule)."
            )

    # Restore the HF student + optimizer for a resumed run. Order is load-bearing:
    # materialise the model (fresh LoRA), overwrite with the epoch's adapter
    # weights, THEN build AdamW (binds to those params) and restore its moments +
    # the LR-scheduler step. Skips completed epochs via start_epoch below.
    start_epoch = 1
    if resume_from:
        probing_script.get_model_and_tokenizer()
        optimization_script.load_adapter_weights(resume_dir)
        optimization_script.get_optimizer()
        optimization_script.load_optimizer_state(resume_dir)
        start_epoch = resume_from + 1
        logger.info("Resumed from epoch %d; continuing at epoch %d.",
                    resume_from, start_epoch)

    if not TRAININGSET_PATH.exists():
        sys.exit(f"Training set not found: {TRAININGSET_PATH}")

    with open(TRAININGSET_PATH) as f:
        trainingset: list[dict] = json.load(f)

    _log_resolved_arm()

    wandb.init(
        entity=config.WANDB_ENTITY or None,
        project=config.WANDB_PROJECT,
        name=config.WANDB_RUN_NAME or None,
        # Key the W&B run to RUN_ID and allow resume so an automatic Modal retry
        # continues one timeline instead of spawning a fresh run per restart.
        id=RUN_ID,
        resume="allow",
        # Full config snapshot, section-ordered to mirror pipeline/config.py.
        # Excludes the two API keys (secrets), the WANDB_* values (already
        # passed to init above as run metadata, not experiment knobs), and the
        # INFERENCE_SYSTEM_PROMPT.
        config={
            # §0 Experiment knobs
            "hint_via_vllm": config.HINT_VIA_VLLM,
            "hint_mode": config.HINT_MODE,
            "student_rollout": config.STUDENT_ROLLOUT,
            "student_rollout_forward": config.STUDENT_ROLLOUT_FORWARD,
            "teacher_rollout_reverse": config.TEACHER_ROLLOUT_REVERSE,
            "seed": config.SEED,
            # §2 Training hyperparameters
            "learning_rate": config.LEARNING_RATE,
            "num_warmup_steps": config.NUM_WARMUP_STEPS,
            "lr_decay_floor": config.LR_DECAY_FLOOR,
            "adam_beta1": config.ADAM_BETA1,
            "adam_beta2": config.ADAM_BETA2,
            "weight_decay": config.WEIGHT_DECAY,
            "grad_accum_steps": config.GRAD_ACCUM_STEPS,
            "train_batch_size": config.TRAIN_BATCH_SIZE,
            "num_epochs": config.NUM_EPOCHS,
            "skip_arithmetic_errors": config.SKIP_ARITHMETIC_ERRORS,
            "train_val_split": config.TRAIN_VAL_SPLIT,
            "max_new_tokens": config.MAX_NEW_TOKENS,
            "max_probe_tokens": config.MAX_PROBE_TOKENS,
            "temperature": config.TEMPERATURE,
            # §3 Compute backend
            "use_modal": config.USE_MODAL,
            "modal_gpu": config.MODAL_GPU,
            # §4 Models & serving
            "eval_model": config.EVAL_MODEL,
            "hint_model": config.HINT_MODEL,
            "state_model": config.STATE_MODEL,
            "openrouter_url": config.OPENROUTER_URL,
            "vllm_base_url": config.VLLM_BASE_URL,
            "model": config.VLLM_MODEL,
            "vllm_gpus": config.VLLM_GPUS,
            "vllm_tensor_parallel": config.VLLM_TENSOR_PARALLEL,
            "vllm_gpu_mem_util": config.VLLM_GPU_MEM_UTIL,
            "vllm_max_model_len": config.VLLM_MAX_MODEL_LEN,
            "vllm_max_logprobs": config.VLLM_MAX_LOGPROBS,
            "vllm_serve_extra_args": config.VLLM_SERVE_EXTRA_ARGS,
            "vllm_startup_timeout": config.VLLM_STARTUP_TIMEOUT,
            "hf_model_path": config.HF_MODEL_PATH,
            "hf_device": config.HF_DEVICE,
            "hf_dtype": config.HF_DTYPE,
            # Empty unless the operator pinned the placement; the ceilings a run
            # actually loaded under are derived per card and logged at load time.
            "hf_max_memory": config.HF_MAX_MEMORY,
            "hf_min_headroom_gib": config.HF_MIN_HEADROOM_GIB,
            # §5 LoRA
            "lora_rank": config.LORA_RANK,
            "lora_alpha": config.LORA_ALPHA,
            "lora_target_experts": config.LORA_TARGET_EXPERTS,
            "lora_expert_rank": config.LORA_EXPERT_RANK,
            "lora_merge_for_gen": config.LORA_MERGE_FOR_GEN,
            # §6 Runtime
            "max_concurrent_items": config.MAX_CONCURRENT_ITEMS,
            "vllm_max_concurrency": config.VLLM_MAX_CONCURRENCY,
            "openrouter_max_concurrency": config.OPENROUTER_MAX_CONCURRENCY,
            "max_retries": config.MAX_RETRIES,
            "retry_base_delay": config.RETRY_BASE_DELAY,
            # §7 Distillation objective
            "probe_topk_via_vllm": config.PROBE_TOPK_VIA_VLLM,
            "probe_topk_k": config.PROBE_TOPK_K,
            "frozen_teacher": config.FROZEN_TEACHER,
            "num_student_rollouts": config.NUM_STUDENT_ROLLOUTS,
            "student_rollout_gen_batch": config.STUDENT_ROLLOUT_GEN_BATCH,
            "student_rollout_temperature": config.STUDENT_ROLLOUT_TEMPERATURE,
            "student_rollout_top_p": config.STUDENT_ROLLOUT_TOP_P,
            "student_rollout_biased": config.STUDENT_ROLLOUT_BIASED,
            "student_rollout_biased_k": config.STUDENT_ROLLOUT_BIASED_K,
            "teacher_rollout_biased": config.TEACHER_ROLLOUT_BIASED,
            "teacher_rollout_biased_k": config.TEACHER_ROLLOUT_BIASED_K,
            "exopd": config.EXOPD,
            "exopd_lambda": config.EXOPD_LAMBDA,
            "adaptive_kl": config.ADAPTIVE_KL,
            "length_penalty": config.LENGTH_PENALTY,
            "length_penalty_weight": config.LENGTH_PENALTY_WEIGHT,
            "length_soft_frac": config.LENGTH_SOFT_FRAC,
            "length_penalty_power": config.LENGTH_PENALTY_POWER,
            "length_penalty_window": config.LENGTH_PENALTY_WINDOW,
            "length_penalty_batch": config.LENGTH_PENALTY_BATCH,
            # Derived / dataset
            "total_items": len(trainingset),
        },
    )

    all_items = trainingset
    total_all = len(all_items)

    # Deliberately hardcoded and independent of config.SEED: the split defines
    # *what* we measure, so it must stay fixed while SEED is swept across runs.
    # Varying it would make each seed's val accuracy a different quantity,
    # confounding training stochasticity with val-set difficulty. Not seeded by
    # config.SEED even when seeding is disabled — a nondeterministic run still
    # has to be comparable to the seeded ones.
    rng = random.Random(42)
    indices = list(range(total_all))
    rng.shuffle(indices)
    n_train = math.ceil(total_all * config.TRAIN_VAL_SPLIT)
    train_items = [all_items[i] for i in indices[:n_train]]
    val_items = [all_items[i] for i in indices[n_train:]]
    total = len(train_items)
    n_val = len(val_items)

    logger.info(
        "Dataset split: %d train / %d val (%.0f/%d split) from %s",
        total, n_val, config.TRAIN_VAL_SPLIT * 100, 100 - round(config.TRAIN_VAL_SPLIT * 100),
        TRAININGSET_PATH,
    )
    logger.info(DIVIDER)

    counts = {"skipped": 0, "trained": 0, "arithmetic": 0, "overlong": 0, "error": 0}
    n_length_trained = 0
    # prev_merged_dir / prev_adapter_name are declared and seeded (from the resumed
    # epoch, if any) near the top of main() so the vLLM resync and per-epoch cleanup
    # share them.

    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        logger.info(DIVIDER)
        logger.info("Epoch %d/%d", epoch, config.NUM_EPOCHS)
        counts = {"skipped": 0, "trained": 0, "arithmetic": 0, "overlong": 0, "error": 0}
        n_length_trained = 0
        # Point the LR decay at this epoch before any step is taken (the warmup
        # factor stays keyed to the optimizer step index). Covers resumed runs too,
        # since start_epoch is the resumed epoch + 1.
        optimization_script.set_epoch(epoch)
        _reset_epoch_entropy()
        _reset_epoch_lengths()
        pending: list = []      # buffered top-k probes awaiting a batched forward
        pending_len: list = []  # buffered token-budget probes (config.LENGTH_PENALTY)

        # Items within an epoch are independent (the vLLM-served weights are
        # frozen until epoch end), so each window's per-item network work runs
        # concurrently. The deferred student forward/backward stays on this
        # (main) thread via flush_training_batch, which only runs after a
        # window's pool has joined — so GPU training never overlaps worker GPU
        # ops. The full-vocab path does its forward/backward inline inside
        # process_item, so it must stay serial: force a width-1 pool there.
        # The student-rollout mode does its HF generate + forward/backward inline
        # (like the full-vocab path), so it must stay serial too.
        concurrent = config.PROBE_TOPK_VIA_VLLM and not config.STUDENT_ROLLOUT
        width = config.MAX_CONCURRENT_ITEMS if concurrent else 1
        width = max(1, width)

        for start in range(0, total, width):
            window = list(enumerate(train_items[start:start + width], start + 1))
            logger.info(DIVIDER)
            if width > 1:
                logger.info(
                    "Epoch %d | Processing items %d-%d (up to %d concurrent) …",
                    epoch, start + 1, start + len(window), len(window),
                )
            # map_interruptible, not `with ThreadPoolExecutor(...) ... ex.map`: the
            # latter parks this thread in Future.result() with no timeout and then
            # joins every in-flight worker on the way out, so a Ctrl-C mid-window
            # was absorbed for the length of the window with vLLM holding its VRAM.
            # Submission order is preserved either way, so batch composition and
            # accumulation-window boundaries stay deterministic regardless of
            # completion order.
            results = utils.map_interruptible(
                lambda p: process_item(epoch, p[0], total, p[1]),
                window, max_workers=width,
            )

            for outcome, probe, len_probe in results:
                if len_probe is not None:
                    pending_len.append(len_probe)
                if outcome == "buffered":
                    pending.append(probe)
                    # Counted as "trained" only once the batched forward succeeds;
                    # a failed batch counts its items as errors instead.
                    if len(pending) >= config.TRAIN_BATCH_SIZE:
                        n_ok, n_bad = flush_training_batch(pending)
                        counts["trained"] += n_ok
                        counts["error"] += n_bad
                else:
                    counts[outcome] += 1
                # Length probes carry sequences up to MAX_NEW_TOKENS long, so they
                # are flushed in their own (small) batches rather than joining the
                # KL batch — see config.LENGTH_PENALTY_BATCH.
                len_batch = max(1, config.LENGTH_PENALTY_BATCH)
                while len(pending_len) >= len_batch:
                    head, pending_len = pending_len[:len_batch], pending_len[len_batch:]
                    n_length_trained += flush_length_batch(head)[0]

        # Train any partial batches left over at the epoch boundary.
        n_ok, n_bad = flush_training_batch(pending)
        counts["trained"] += n_ok
        counts["error"] += n_bad
        n_length_trained += flush_length_batch(pending_len)[0]

        train_correct = counts["skipped"]
        train_accuracy = train_correct / total if total > 0 else 0.0
        logger.info(DIVIDER)
        logger.info(
            "Epoch %d/%d training complete. Total=%d | Correct/skipped=%d | Trained=%d | "
            "Arithmetic-skipped=%d | Over-budget=%d | Length-term items=%d | "
            "Errors=%d | Train accuracy=%.4f",
            epoch, config.NUM_EPOCHS,
            total, counts["skipped"], counts["trained"], counts["arithmetic"],
            counts["overlong"], n_length_trained, counts["error"], train_accuracy,
        )
        # Peak VRAM for the epoch just finished, then reset the high-water mark so
        # each epoch reports its own. Logged before the merge/restart below, which
        # would otherwise fold the merge's transient into the training figure.
        if probing_script.is_loaded():
            logger.info("Epoch %d peak VRAM — %s", epoch, utils.vram_peak_report())

        # ── Checkpoint + resync vLLM BEFORE validation, so validation (and the
        # next epoch's rollouts, and the skip/train decisions derived from them)
        # all reflect the weights just trained this epoch — not last epoch's.
        # Two refresh strategies (see pipeline/vllm_server.py):
        #   • expert LoRA — merge the adapter into a full checkpoint and restart
        #     vLLM on it (fused-MoE experts can't take a hot-swapped LoRA);
        #   • non-expert LoRA — hot-load the saved adapter into the live server.
        if probing_script.is_loaded():
            optimization_script.flush_gradients()
            epoch_ckpt_dir = CHECKPOINT_DIR / f"epoch_{epoch}"
            adapter_dir = optimization_script.save_adapter(epoch_ckpt_dir)
            # Persist optimizer moments + LR-scheduler step alongside the adapter so
            # a resumed run continues the optimizer in-place (not cold).
            optimization_script.save_optimizer_state(epoch_ckpt_dir)
            if config.LORA_TARGET_EXPERTS:
                merged_dir = optimization_script.save_merged_model(
                    CHECKPOINT_DIR / f"epoch_{epoch}_merged"
                )
                vllm_server.restart(merged_dir)
                # Restart succeeded and vLLM now serves `merged_dir`; the prior
                # epoch's merged checkpoint is dead weight — drop it so disk
                # doesn't grow by a full model every epoch (#2).
                if prev_merged_dir is not None:
                    shutil.rmtree(prev_merged_dir, ignore_errors=True)
                prev_merged_dir = merged_dir
            else:
                # Hot-load the new adapter, then point requests at it, then unload
                # the previous one — loading first means a failure never strands us
                # without a live adapter.
                adapter_name = f"epoch_{epoch}"
                vllm_server.load_adapter(adapter_name, adapter_dir)
                inference_script.set_active_model(adapter_name)
                if prev_adapter_name is not None:
                    vllm_server.unload_adapter(prev_adapter_name)
                prev_adapter_name = adapter_name
            # Completion sentinel, written LAST so _find_resume_epoch only ever
            # selects a fully-written epoch: the adapter, optimizer state, and (on
            # the expert path) the merged checkpoint all exist by now. A kill before
            # the commit below leaves this epoch uncommitted, so resume falls back
            # to the prior epoch (which has its own DONE).
            (epoch_ckpt_dir / "DONE").write_text(RUN_ID)
            # Persist this epoch's freshly-written checkpoint to durable storage
            # (a no-op locally; on Modal this commits the checkpoints Volume so
            # the adapter survives a hard container kill before the run finishes).
            on_checkpoint()
        else:
            logger.info(
                "Epoch %d: no training occurred (model never loaded); skipping checkpoint.",
                epoch,
            )

        # ── Validation (on the freshly restarted weights) ──────────────────────
        val_correct = 0
        val_errors = 0
        if n_val > 0:
            logger.info(DIVIDER)
            logger.info("Epoch %d/%d: Running validation on %d items …", epoch, config.NUM_EPOCHS, n_val)
            val_width = max(1, config.MAX_CONCURRENT_ITEMS)
            val_results = utils.map_interruptible(
                lambda p: validate_item(epoch, p[0], n_val, p[1]),
                list(enumerate(val_items, 1)), max_workers=val_width,
            )
            for outcome in val_results:
                if outcome == "correct":
                    val_correct += 1
                elif outcome == "error":
                    val_errors += 1
            val_accuracy = val_correct / n_val if n_val > 0 else 0.0
            logger.info(
                "Epoch %d/%d validation complete. Correct=%d/%d | Errors=%d | Val accuracy=%.4f",
                epoch, config.NUM_EPOCHS, val_correct, n_val, val_errors, val_accuracy,
            )
        else:
            val_accuracy = 0.0

        epoch_metrics = {
            "epoch": epoch,
            "epoch_skipped": counts["skipped"],
            "epoch_trained": counts["trained"],
            "epoch_arithmetic_skipped": counts["arithmetic"],
            "epoch_overlong": counts["overlong"],
            "epoch_length_trained": n_length_trained,
            "epoch_errors": counts["error"],
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "val_correct": val_correct,
            "lr": optimization_script.current_lr(),
        }
        if config.LOG_ENTROPY:
            # Mean over this epoch's trained items (empty if nothing trained).
            epoch_metrics.update(_epoch_entropy_means())
        # Logged next to val_accuracy on purpose: the token-budget penalty is
        # working if mean tokens fall while accuracy holds, and over-tuned if they
        # fall together (see config.LENGTH_PENALTY).
        epoch_metrics.update(_epoch_length_means())
        wandb.log(epoch_metrics)

    logger.info(DIVIDER)
    if probing_script.is_loaded():
        final_dir = optimization_script.save_adapter(CHECKPOINT_DIR / "final")
        final_merged = optimization_script.save_merged_model(CHECKPOINT_DIR / "final_merged")
        # final_merged supersedes the last epoch's merged checkpoint; drop it.
        if prev_merged_dir is not None:
            shutil.rmtree(prev_merged_dir, ignore_errors=True)
        logger.info("Final LoRA adapter saved to %s (merged checkpoint at %s)",
                    final_dir, final_merged)
        on_checkpoint()
        wandb.summary.update({"final_adapter": final_dir, "final_merged": final_merged})
    else:
        logger.info("No training occurred in any epoch; no adapter to save.")
    logger.info("Pipeline complete (%d epochs).", config.NUM_EPOCHS)
    wandb.summary.update({
        "total_items": len(trainingset),
        "train_items": total,
        "val_items": n_val,
        "epochs": config.NUM_EPOCHS,
        "final_epoch_skipped": counts["skipped"],
        "final_epoch_trained": counts["trained"],
        "final_epoch_arithmetic_skipped": counts["arithmetic"],
        "final_epoch_overlong": counts["overlong"],
        "final_epoch_length_trained": n_length_trained,
        "final_epoch_errors": counts["error"],
    })
    wandb.finish()


if __name__ == "__main__":
    main()
