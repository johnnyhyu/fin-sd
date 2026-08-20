# Architecture

How serving, training and the epoch boundary fit together. Read this before
changing anything about GPU placement or the vLLM setup — several of the
constraints below are not obvious from the code and are expensive to rediscover.

## The pipeline owns the vLLM process

**Do not run `vllm serve` yourself.** The pipeline starts the server, and
restarts it on a freshly merged checkpoint at each epoch boundary. A
hand-started server collides on the port and the run fails at startup.

`pipeline/vllm_server.py` manages that lifecycle. Configure it through `.env`:

| Setting | Default | |
|---|---|---|
| `VLLM_GPUS` | `0,1,2,3` | Cards for serving. Must stay disjoint from the trainer's. Four, because the merged 120B checkpoint vLLM restarts on each epoch is ~218 GiB. |
| `VLLM_TENSOR_PARALLEL` | `4` | Always follows the `VLLM_GPUS` card count; a mismatch aborts vLLM at startup. |
| `VLLM_GPU_MEM_UTIL` | `0.9` | |
| `VLLM_MAX_MODEL_LEN` | `16384` | |
| `VLLM_MAX_LOGPROBS` | `20` | Must be at least `PROBE_TOPK_K` for the top-k probing path. |
| `VLLM_STARTUP_TIMEOUT` | `1800` | |
| `VLLM_SERVE_EXTRA_ARGS` | `''` | Raw extra flags. |

## Why the teacher is served by merge-and-restart, not adapter hot-swap

gpt-oss packs its MoE experts as fused 3-D parameters. There is no per-expert
module for vLLM to attach a LoRA adapter to, so when `LORA_TARGET_EXPERTS=1` —
the default, and what the paper uses — the trained weights cannot be hot-loaded.
Instead, at each epoch boundary the adapter is merged into a full standalone
checkpoint and vLLM is restarted on it.

Consequences worth knowing:

- A merged checkpoint is roughly 39 GiB for the 20B student, 240 GiB for the
  120B, so the previous epoch's merged copy is dropped once vLLM is serving the
  next. `tools/serve_checkpoint.py` re-merges an epoch's adapter on demand when
  its merged copy is gone. The two baselines expose this as a knob of their own
  (`OPSD_KEEP_EPOCH_MERGED`, default off; `SFT_KEEP_EPOCH_MERGED`, default on —
  an SFT adapter cannot be re-merged after the fact).
- Expert targeting requires **BF16** weights for both the HF training model and
  `VLLM_MODEL`, since the merged checkpoint vLLM serves is full precision. Use
  `unsloth/gpt-oss-20b-BF16`, not the quantized release.
- The teacher tracks training on this path. `FROZEN_TEACHER=1` only guarantees a
  genuinely frozen target when combined with `PROBE_TOPK_VIA_VLLM=0`, because
  otherwise vLLM is serving merged — that is, trained — weights.

With `LORA_TARGET_EXPERTS=0` the adapter is hot-loaded into the live server
instead, and `--enable-lora --max-loras 2` is passed so a load failure never
destroys the still-serving adapter.

## GPU placement

Serving and training GPUs must be **disjoint**. `launch.py` arranges this before
anything touches CUDA:

- **20B BF16** (the default): the next two idle cards, under 4 GiB used. First
  serves, second trains, tensor-parallel 1. Reserved with a lock, so two
  simultaneous launches cannot both claim the same "idle" card — a freshly
  selected card shows no VRAM until its model loads, which makes a plain
  `nvidia-smi` scan race.
- **Anything larger**: the fixed trainer block `4,5,6,7`, leaving `VLLM_GPUS`
  (`0,1,2,3`) for serving — four cards each way, which is what serving the merged
  120B checkpoint requires.

An explicit `CUDA_VISIBLE_DEVICES` always wins and is simply reserved so
concurrent runs skip those cards. Placement is keyed on the model *family*, not
an exact repo id, because the same 20B weights are spelled several ways.

## Trainer VRAM

Per-GPU ceilings are **derived at load time** from each visible card's live free
VRAM, minus `HF_MIN_HEADROOM_GIB` (default 6) held back for one step's
activations, LoRA factors and optimizer moments. A layout therefore sizes itself
to the cards it landed on rather than to a number written down by hand.

`HF_MAX_MEMORY` is empty by default. Set it to absolute per-card GiB caps
(`62,62,62,62`) only to pin a placement deliberately — it overrides the derived
ceilings, and it also fixes the width of the merge block (see
`utils.hf_merge_gpu_count`).

Two gpt-oss-specific memory traps are handled in `pipeline/`, and are worth
knowing about before changing either:

- **Attention kernel.** gpt-oss declares `_supports_sdpa = False` — its attention
  sinks are an extra logit per head that SDPA's fused kernels cannot express — so
  transformers autoselects `eager`, which builds the full
  `[batch, heads, seq, seq]` score matrix. That is ~30 GiB of transient VRAM per
  layer for the 120B at 8192 tokens, and it happens silently.
  `pipeline/attention.py` makes the choice in one place;
  `HF_ATTN_IMPLEMENTATION` pins it.
- **Expert LoRA.** PEFT's `ParamWrapper` builds the full delta for each fused MoE
  tensor. `pipeline/expert_lora.py` adapts them in activation space instead —
  0.69 GiB transient per layer against PEFT's 18.1 GiB, with frozen experts
  costing no parameters, gradients or optimizer moments.
  `EXPERT_LORA_ACTIVATION_SPACE=0` falls back to stock PEFT.

## Prompt encoding

Training contexts and serving contexts must be token-identical, or the student
is trained under a prompt its checkpoint is never served under.
`pipeline/prompts.py` renders through `openai_harmony` — the same library vLLM
serves with — rather than through `apply_chat_template`, which adds 17 tokens
the server never sends. `opsd/` and `sft/` render through the same module for
the same reason.

## The epoch boundary

1. Mine failures: sample the student on each training problem, grade against
   ground truth, keep what it got wrong.
2. For each failure, localize the first faulty step and write a hint; drop pure
   arithmetic slips without a gradient.
3. Truncate before the error, roll out `L` tokens, accumulate the KL gradient.
4. Step AdamW every `GRAD_ACCUM_STEPS` contributing examples, flushing any
   partial batch at the end.
5. Merge the adapter, restart vLLM on it, re-freeze the teacher, grade the
   held-out validation split.

The failure set is recomputed every epoch, so training tracks the student's
current weaknesses rather than a set fixed at initialization.

## Running on Modal

`USE_MODAL=1` dispatches the whole pipeline — vLLM and the HF trainer — into one
Modal GPU container. The base model downloads once into a Volume and is reused;
adapters and checkpoints persist to a second Volume. One-time setup:

```bash
pip install modal && modal token new
modal secret create openrouter OPENROUTER_API_KEY=sk-or-...
```

The dispatch runs `--detach`, so a network blip cannot tear down a multi-hour
job. See `modal_app.py`.
