# SFT distillation

> **Baseline.** This is the SFT comparison in §5.2 and Table 2 of the paper, not
> the paper's method — it underperforms the base model there. Fin-SD itself is
> `run_pipeline.py`. Run this arm with
> `python -m sft.run --config configs/sft.env`.

A standard supervised-fine-tuning pipeline that distils the teacher
(`pipeline.config.HINT_MODEL` — the hint model, served via OpenRouter) into a
student model, using the problems in [`data/trainingset.json`](../data/trainingset.json).

Two stages:

1. **Generate** — the teacher solves every problem with the shared inference
   system prompt, producing `<reasoning>…</reasoning><answer>…</answer>` traces.
   Reasoning-model teachers return their chain-of-thought on the separate
   `reasoning`/`reasoning_content` channel rather than inside the content, so both
   halves are read back and re-emitted as one canonical tagged trace (also stored
   raw in the record's `reasoning` field); traces that come back with no reasoning
   are resampled and dropped, since the point is to supervise the reasoning and
   not just the answer. With rejection sampling on (default), a trace is kept only
   if its final answer matches the ground-truth answer (graded by
   `pipeline.eval_script.run_eval`).
   Output: `sft/data/distill.jsonl` (resumable — re-run to fill gaps).
2. **Train** — Hugging Face `Trainer` fine-tunes the student on those pairs with
   a plain next-token loss over the full prompt+completion sequence
   (`SFT_MASK_PROMPT=1` supervises only the completion). LoRA on by default.
   Output: one checkpoint dir per epoch under `sft/checkpoints/<run-id>/` (see
   [Checkpoints](#checkpoints)).

   The supervised target is the assistant turn **in harmony form** —
   `<|channel|>final<|message|>` + the tagged trace + `<|return|>` — because that
   is what gpt-oss actually generates and what vLLM's harmony parser expects. The
   stock `apply_chat_template` render of a [system, user, assistant] conversation
   omits the channel header (the gpt-oss template only emits it for a message with
   a separate `thinking` field), and a student trained on that emits `<|message|>`
   with no channel, which comes back from vLLM as `content: null` — an unusable
   checkpoint. `dataset._render` builds the turn explicitly instead.

   The prompt half comes from [`pipeline.prompts`](../pipeline/prompts.py), which
   renders through the same `openai_harmony` encoding vLLM serves with (verified
   token-identical against the engine's own `prompt_token_ids`). Rendering with
   the chat template instead adds 17 tokens the server never sends, so the student
   would be trained on a prompt its checkpoint is never served under.

## Quick start

```bash
# Both stages (teacher decode needs OPENROUTER_API_KEY; training needs GPUs):
OPENROUTER_API_KEY=... python -m sft.run

# Or run the stages separately:
python -m sft.generate_data          # build sft/data/distill.jsonl
python -m sft.train                  # SFT on it

# Smoke test on 20 problems:
python -m sft.run --limit 20
```

## Checkpoints

Same layout as the main pipeline, one subdir per run under `SFT_OUTPUT_DIR`
(`SFT_RUN_ID`, else `RUN_ID`, else a UTC timestamp), with a fresh pair written
every epoch:

```
sft/checkpoints/20260730-120000/
├── epoch_1/           adapter (or, without LoRA, the full model) + DONE sentinel
├── epoch_1_merged/    the same weights merged into a standalone full checkpoint
├── epoch_2/  epoch_2_merged/  …
├── final/             the trained adapter at end of training
└── final_merged/      merged copy of the final weights (SFT_MERGE_AFTER_TRAIN=1)
```

Each `epoch_<N>_merged` is a complete checkpoint (weights + config +
`generation_config.json` + tokenizer) that vLLM serves as-is, so any epoch of any
run can be brought up for benchmarking exactly like a pipeline epoch:

```bash
python tools/serve_checkpoint.py --source sft --run-id 20260730-120000 --epoch 2
```

Every epoch's merged copy is kept by default (`SFT_KEEP_EPOCH_MERGED=0` prunes
the previous one as the pipeline does, leaving only the newest). Keeping them
matters here: `serve_checkpoint`'s re-merge rebuilds the PEFT tree from
`pipeline.config`, whose expert rank need not match an SFT run's, so a bare SFT
adapter does not always re-merge after the fact. Budget ~39 GiB per epoch for the
20B (~240 GiB for the 120B).

Held-out validation (`SFT_VAL_SPLIT`) does **not** use that merged checkpoint.
The held-out problems are answered by the live in-memory student
(`model.generate`, see [`validate.py`](validate.py)) and graded with the same
OpenRouter grader — no merge, no vLLM, no extra GPU. What it keeps identical to
the serving path, so the number still means what a served checkpoint would score:
the prompt (`serving_prompt_ids`), the harmony stop tokens, the `<answer>` parse,
and the grader. A completion that spends its whole `SFT_VAL_MAX_NEW_TOKENS`
budget in the analysis channel never opens a final message; those grade as wrong
(no answer was produced) but are counted and logged separately, because they mean
"still thinking", not "answered incorrectly".

## GPU placement

Training picks its own cards, the same way `launch.py` does — no
`CUDA_VISIBLE_DEVICES` prefix needed, and two runs on one box won't collide
(each reservation is held under a cross-process lock for the run's lifetime):

* **20B student** (the default): the next idle cards (<4 GiB used) — **two** for
  the HF trainer, which shards the student across both (`device_map="auto"`
  splits the ~39 GiB bf16 base ~20/20, and the per-card activation headroom is
  what lets a long example train), plus one more for vLLM only when the run
  actually serves (`--serve` / `SFT_SERVE_AFTER_TRAIN`; validation no longer
  needs one). No per-card cap is pinned for that pair: the ceilings are each
  card's live free VRAM minus `HF_MIN_HEADROOM_GIB`, read off the driver when the
  student loads. `SFT_HF_MAX_MEMORY` overrides them with absolute GiB caps.
* **anything else** (e.g. the 120B): the fixed `4,5,6,7` trainer block, with
  `VLLM_GPUS` (`0,1,2,3`) left for serving.

An explicit `CUDA_VISIBLE_DEVICES` always wins; the run then only reserves those
cards. Selection happens after the generate stage (which is OpenRouter-only), so
a long distillation never sits on idle GPUs. See [`gpus.py`](gpus.py).

## Attention kernel

The student is loaded with an explicitly chosen attention implementation, because
the default is a memory trap for gpt-oss. gpt-oss sets `_supports_sdpa = False`
(its attention sinks are an extra per-head logit that SDPA's fused kernels cannot
express), so transformers' autoselect skips SDPA; with flash-attn absent — it is
not in `requirements.txt`, since the wheel is CUDA-version specific — the
fallback is `eager`, which materialises the `[batch, heads, seq, seq]` score
matrix. For one 8875-token example on the 20B that is 9.4 GiB *per layer's
attention*, on top of weights and optimizer state; measured end-to-end, a single
forward at that length peaks at **28.5 GiB under eager vs 0.09 GiB under
FlexAttention**. That, not the model size, is what OOMed the 20B mid-epoch, and
it happens silently because eager is a fine choice for short sequences.

[`model.py`](model.py) therefore picks `flash_attention_2` if installed, else
`flex_attention`, and logs which one the model *actually* loaded with.
`SFT_ATTN_IMPLEMENTATION` pins it (`eager` is allowed, with a warning; an
unavailable value is an error rather than a silent downgrade).

Keeping FlexAttention on its fused path then takes two more things, both of which
fail silently and only once training is under way:

* **Dynamo's recompile ceiling.** transformers compiles `flex_attention` into one
  process-wide singleton shared by every layer and shard, and SFT trains
  variable-length examples at batch size 1, so each new length bucket costs a
  cache entry — the 20B burned all 8 of the default budget within 14 steps. On
  overflow dynamo does not raise; it runs the kernel unfused, i.e. straight back
  to eager's memory profile. The ceiling is raised to 128
  (`SFT_DYNAMO_CACHE_LIMIT`), not disabled, so a real recompile storm still shows.
* **A per-shard `BlockMask`.** The mask is built once per forward on the *inputs'*
  device, which under `device_map="auto"` is the first shard only. Moving it is
  not just `BlockMask.to()`: `mask_mod` is a Python closure that captures tensors
  bound to the original card (`q_offset`, and the padding mask when the batch is
  padded), so the traced `q_idx + q_offset` straddles two GPUs and aborts the
  compile for every layer that isn't on card 0. Both the index tensors and the
  closure are re-homed per shard, and cached on the mask for the forward's life.

## Configuration

Everything is set in [`config.py`](config.py) and overridable via environment
variables (and `.env`). It reuses the main pipeline's teacher, OpenRouter
credentials, system prompt, and student/LoRA defaults. Common knobs:

| Env var | Default | Meaning |
| --- | --- | --- |
| `SFT_TEACHER_MODEL` | `HINT_MODEL` | Teacher used to generate solutions. |
| `SFT_STUDENT_MODEL` | `HF_MODEL_PATH` | Model being fine-tuned. |
| `SFT_REJECT_SAMPLING` | `1` | Keep only teacher traces with a correct answer. |
| `SFT_REJECT_MAX_ATTEMPTS` | `3` | Retries per problem before dropping it. |
| `SFT_REQUIRE_REASONING` | `1` | Drop traces that came back with no reasoning. |
| `SFT_USE_LORA` | `1` | LoRA vs full fine-tuning. |
| `SFT_NUM_EPOCHS` | `3` | Training epochs. |
| `SFT_LEARNING_RATE` | `1e-4` | LR. |
| `SFT_BATCH_SIZE` / `SFT_GRAD_ACCUM_STEPS` | `1` / `8` | Effective batch = product. |
| `SFT_MAX_SEQ_LEN` | `2 × MAX_NEW_TOKENS` (16384) | Sequences longer than this are truncated. |
| `SFT_MASK_PROMPT` | `0` | Supervise only the assistant turn (`1`) vs the whole sequence. |
| `SFT_VAL_SPLIT` / `SFT_VAL_MAX_EXAMPLES` | `0.1` / `64` | Held-out fraction, and its cap, graded from the live weights each epoch (`0` disables). |
| `SFT_VAL_BATCH_SIZE` / `SFT_VAL_MAX_NEW_TOKENS` | `64` / `8192` | Validation decode: prompts per `generate` call, and the per-problem token budget. |
| `SFT_ATTN_IMPLEMENTATION` | *(autoselect)* | Pin the attention kernel. Empty prefers `flash_attention_2`, else `flex_attention`, and never `eager` — see below. |
| `SFT_RUN_ID` | UTC timestamp | Names this run's checkpoint subdir. |
| `SFT_KEEP_EPOCH_MERGED` | `1` | Keep every epoch's merged checkpoint (`0` keeps only the newest). |
| `SFT_MERGE_AFTER_TRAIN` | `0` | Also write `final_merged/` (a copy of the last epoch's merged weights). |
| `SFT_SERVE_AFTER_TRAIN` | `0` | Leave vLLM up on the final checkpoint (same as `sft.train --serve`). |
| `SFT_WANDB` | `1` | Log to Weights & Biases. |

## Files

| File | Purpose |
| --- | --- |
| `generate_data.py` | Teacher → distillation JSONL (with optional rejection sampling). |
| `dataset.py` | Torch `Dataset` + padding collator. |
| `model.py` | Student + LoRA loader (mirrors the pipeline's HF/PEFT conventions). |
| `train.py` | `Trainer`-based SFT loop; writes the per-epoch adapter + merged checkpoints. |
| `run.py` | Orchestrates generate → train. |
| `gpus.py` | Picks + reserves the training/serving GPUs (see above). |
| `config.py` | All settings. |
