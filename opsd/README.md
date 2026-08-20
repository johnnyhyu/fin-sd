# opsd — on-policy context distillation from reference solutions

> **Baseline.** This is the OPSD comparison in §5.2 and Table 2 of the paper,
> not the paper's method. Fin-SD itself is `run_pipeline.py`. Run this arm with
> `python -m opsd.run --config configs/opsd.env`.

`opsd` trains the student to solve problems from scratch by distilling from a
**teacher that has seen a reference solution**. Unlike the main `pipeline/` (which
injects a targeted *hint at the mistake point*), the teacher/student difference
here is pure **context**: the teacher's prompt contains a full reference solution,
the student's does not.

## Two stages

### 1. Reference build (`opsd.generate_data`)
The hint model (`OPSD_HINT_MODEL`, default `pipeline.config.HINT_MODEL`, via
OpenRouter) writes a full reference solution for **every** problem in
`data/trainingset.json`. On an answer mismatch the problem is retried with the
ground-truth answer appended to the prompt (so the model can write a solution that
reaches it); the first answer-matching trace is kept, else the last attempt is kept
so every problem yields a reference. Output is a resumable JSONL at
`opsd/data/reference.jsonl`.

### 2. Training (`opsd.train`)
Each epoch, over the training split (served on vLLM):

1. Run inference on the problem; grade it. **Correct → skip** (no update).
2. **Wrong →** sample a student rollout (default 1, capped at 1024 tokens) from
   the **bare problem** prompt, then compute **forward KL(teacher ‖ student)** over
   those tokens, where:
   - **student** prompt = `[system, user: problem]`
   - **teacher** prompt = `[system, user: problem + "Here is a reference solution:" +
     reference + "After understanding the reference solution, please try to solve
     this problem using your own approach below:"]`

   The teacher forward uses the **live student weights** (adapter on), so the
   target co-evolves with the student. Both prefixes open a fresh assistant turn,
   so the rollout tokens align to both — and both are rendered by
   [`pipeline.prompts`](../pipeline/prompts.py) through the same `openai_harmony`
   encoding vLLM serves with, so the context the rollouts are *sampled* under and
   the context they are *scored* under are token-identical (the chat template
   would add 17 tokens the server never sends).
3. AdamW step (grad accumulation), then at the epoch boundary the trained weights
   are merged + vLLM is restarted on them (expert-LoRA path) or the adapter is
   hot-loaded, and the held-out validation split is graded for answer accuracy.

## Run

```bash
# build references then train
python -m opsd.run

# stages independently
python -m opsd.generate_data --limit 20     # smoke-test the reference build
python -m opsd.train                        # train on an existing reference.jsonl
```

Training picks and reserves its own GPUs (`pipeline.utils.configure_train_gpus`,
the same placement `launch.py` gives the main pipeline, shared with `sft`): for
the 20B student, the next idle card for the HF trainer plus one for vLLM, kept
disjoint; for anything larger, the fixed `4,5,6,7` trainer block with
`VLLM_GPUS` (`0,1,2,3`) left for serving. An explicit `CUDA_VISIBLE_DEVICES` still
wins and is simply reserved so concurrent runs skip those cards. Selection
happens inside `train()`, after the OpenRouter-only reference build, so a long
generate stage never sits on idle GPUs.

## Checkpoints

Same layout as the main pipeline, one subdir per run under `OPSD_OUTPUT_DIR`
(`OPSD_RUN_ID`, else `RUN_ID`, else a UTC timestamp): `epoch_<N>/` holds the
epoch's adapter + optimizer state, `epoch_<N>_merged/` the same weights merged
into a standalone full checkpoint (expert-LoRA path), and `final/` +
`final_merged/` the end-of-training pair. Any epoch can be brought back up for
benchmarking:

```bash
python tools/serve_checkpoint.py --source opsd --run-id 20260730-120000 --epoch 2
```

That re-merges the epoch's adapter when its merged copy is gone — the previous
epoch's is dropped once vLLM is serving the next one, since a full model per
epoch is ~39 GiB (20B) / ~240 GiB (120B). Set `OPSD_KEEP_EPOCH_MERGED=1` to keep
them all and skip the re-merge.

## Key knobs (all `OPSD_*`, env-overridable)

| Knob | Default | Meaning |
| --- | --- | --- |
| `OPSD_HINT_MODEL` | `pipeline.config.HINT_MODEL` | Model that writes reference solutions |
| `OPSD_REFERENCE_MAX_ATTEMPTS` | 4 | Answer-appended retries in the reference build |
| `OPSD_NUM_ROLLOUTS` | 1 | Student rollouts per wrong problem |
| `OPSD_MAX_ROLLOUT_TOKENS` | 1024 | Rollout length cap (the forward-KL span) |
| `OPSD_ROLLOUT_TEMPERATURE` / `OPSD_ROLLOUT_TOP_P` | 1.0 / 0.9 | Rollout sampling |
| `OPSD_NUM_EPOCHS` | `pipeline.config.NUM_EPOCHS` | Training epochs |
| `OPSD_TRAIN_VAL_SPLIT` | `pipeline.config.TRAIN_VAL_SPLIT` | Train fraction (fixed 42-seeded split) |
| `OPSD_KEEP_EPOCH_MERGED` | 0 | Keep every epoch's merged checkpoint instead of only the newest |
| `OPSD_LEARNING_RATE`, `OPSD_GRAD_ACCUM_STEPS`, `OPSD_NUM_WARMUP_STEPS`, … | pipeline defaults | Optimizer schedule (written back onto `pipeline.config`) |

The student model, LoRA, HF/vLLM serving, OpenRouter credentials, grader, and
system prompt are all reused from `pipeline.config` — an opsd run serves and
trains the same gpt-oss student the same way `run_pipeline.py` does.

> Reuses `pipeline/`: `probing_script` (HF singleton + logit extraction),
> `optimization_script` (AdamW + checkpointing), `vllm_server` (serving lifecycle),
> `inference_script`, `eval_script`, `loss_script.run_loss` (forward KL), `utils`.
> `opsd/modal_app.py` (cloud dispatch, mirroring `sft/modal_app.py`) is not yet
> included — a stretch item for remote runs.
