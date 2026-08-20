# Configuration

Every knob lives in `pipeline/config.py` and is set by environment variable.
There are 83 of them; about fifteen describe the paper. This page separates the
three groups so the rest is documented without being in your way.

**Precedence:** shell environment > `configs/` preset > `.env` > code default.
All three file layers use `setdefault`, so whichever writes a key first wins.

Presets may declare `EXTENDS=<file>`, resolved recursively against `configs/`.
The child's values win. A cycle raises rather than hanging.

---

## Paper knobs

The defaults below are Table 1. `configs/finsd.env` pins the same values
explicitly, so the table in the paper and that file can be diffed by eye.

### Method

| Knob | Default | |
|---|---|---|
| `HINT_MODE` | `full` | How much the critique reveals. `full` is Fin-SD's, whose prompt carries the Figure 2 example verbatim. **`vague` emits exactly "Your mistake is conceptual." — that is the §5.3 placebo, not the method.** |
| `HINT_VIA_VLLM` | `1` | Serve the hint from the live policy (π_H = self). `0` routes to `OPENROUTER_HINT_MODEL` for the §6.1 ablation. |
| `OPENROUTER_HINT_MODEL` | `openai/gpt-oss-120b` | The external hint writer, used only when `HINT_VIA_VLLM=0`. |
| `STUDENT_ROLLOUT` | `1` | Sample rollouts from the student and score reverse KL over them (§3.4). |
| `STUDENT_ROLLOUT_FORWARD` | `0` | Flip to forward KL on the same positions (§6.2). |
| `MAX_PROBE_TOKENS` | `100` | The rollout length **L** (§3.4, §6.3). |
| `SKIP_ARITHMETIC_ERRORS` | `1` | Localize arithmetic slips but withhold their gradient (§3.2). They still count as incorrect in accuracy. |

### Optimization

| Knob | Default | |
|---|---|---|
| `LEARNING_RATE` | `2e-5` | |
| `NUM_WARMUP_STEPS` | `15` | See "post-paper findings" below. |
| `LR_DECAY_FLOOR` | `1.0` | Floor of the post-warmup cosine decay, as a fraction of peak LR. **1.0 disables decay** — warmup then constant, which is what the paper ran. |
| `ADAM_BETA1` / `ADAM_BETA2` | `0.9` / `0.95` | β₂ gives a ~20-step horizon, matched to a run of tens of steps rather than thousands. |
| `WEIGHT_DECAY` | `0.0` | |
| `TRAIN_BATCH_SIZE` | `1` | Items per forward/backward. |
| `GRAD_ACCUM_STEPS` | `16` | Effective batch = the product of these two = 16. |
| `NUM_EPOCHS` | `10` | |
| `TRAIN_VAL_SPLIT` | `0.8` | The split itself is fixed, not seeded, so every seed scores the same validation set. |
| `MAX_NEW_TOKENS` | `8192` | Reasoning rollout cap. |
| `TEMPERATURE` | `0.0` | Greedy. Run-to-run spread is therefore not sampling variance (§4.3). |
| `SEED` | `7` | Seeds Python, NumPy, torch and vLLM. `-1` disables. The paper sweeps 7, 42, 72, 2, 27. |

Gradient clipping is fixed at `max_norm=1.0` in `pipeline/optimization_script.py`
and is not configurable.

### The student model

**The repository default is the 120B, not the paper's student.** `HF_MODEL_PATH`
defaults to `unsloth/gpt-oss-120b-BF16` and `VLLM_MODEL` to
`unsloth/gpt-oss-120b`, with `VLLM_GPUS=0,1,2,3` and tensor-parallel 4 sized for
serving it.

`configs/finsd.env` pins **both** back to the 20B, which is the paper's student.
Both entries have to be named: pinning only the trainer would train a 20B while
serving a 120B. Every preset extends `finsd.env`, so every paper arm gets the
20B; a bare `python launch.py` with no `--config` gets the 120B and needs eight
cards.

### LoRA

| Knob | Default | |
|---|---|---|
| `LORA_RANK` / `LORA_ALPHA` | `64` / `128` | Attention adapters. |
| `LORA_EXPERT_RANK` | `8` | Expert adapters. Empty resolves to `rank // num_experts`. |
| `LORA_TARGET_EXPERTS` | `1` | Adapt the MoE experts as well as attention. Requires BF16 weights — see [ARCHITECTURE.md](ARCHITECTURE.md). |
| `PROFILE_EXPERT_ACTIVATIONS` | `1` | Profile router activations to choose which experts to adapt. `launch.py` refreshes this cache when it is missing or was built for a different model. |

---

## Infrastructure

Machine- and deployment-specific; nothing here changes the method.

| Group | Knobs |
|---|---|
| Serving | `VLLM_BASE_URL`, `VLLM_MODEL`, `VLLM_GPUS`, `VLLM_TENSOR_PARALLEL`, `VLLM_GPU_MEM_UTIL`, `VLLM_MAX_MODEL_LEN`, `VLLM_MAX_LOGPROBS`, `VLLM_STARTUP_TIMEOUT`, `VLLM_SERVE_EXTRA_ARGS`, `VLLM_API_KEY`, `VLLM_MAX_CONCURRENCY` |
| Training host | `HF_MODEL_PATH`, `HF_DEVICE`, `HF_DTYPE`, `HF_MAX_MEMORY`, `HF_MIN_HEADROOM_GIB`, `HF_ATTN_IMPLEMENTATION` |
| External models | `OPENROUTER_API_KEY`, `OPENROUTER_EVAL_MODEL`, `OPENROUTER_STATE_MODEL`, `OPENROUTER_MAX_CONCURRENCY` |
| Concurrency and retries | `MAX_CONCURRENT_ITEMS`, `MAX_RETRIES`, `RETRY_BASE_DELAY` |
| Probing path | `PROBE_TOPK_VIA_VLLM`, `PROBE_TOPK_K`, `LORA_MERGE_FOR_GEN` |
| Expert LoRA | `EXPERT_LORA_ACTIVATION_SPACE` |
| Cloud | `USE_MODAL`, `MODAL_GPU` |
| Logging | `WANDB_MODE`, `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_RUN_NAME`, `LOG_ENTROPY` |

`HF_MAX_MEMORY` is empty by default; per-card ceilings are derived at load time
from live free VRAM minus `HF_MIN_HEADROOM_GIB` (6). Set it only to pin a
placement by hand. `HF_ATTN_IMPLEMENTATION` and `EXPERT_LORA_ACTIVATION_SPACE`
both guard large, silent memory costs specific to gpt-oss — see
[ARCHITECTURE.md](ARCHITECTURE.md) before changing either.

`WANDB_MODE` defaults to `disabled`, which makes `wandb.init` and every
`wandb.log` a no-op — no login, no network — so a fresh clone trains without a
Weights & Biases account. Set it to `online` to log.

---

## Experimental surface

**None of the following is used by any result in the paper.** These are
alternative objectives and paths explored during development, kept because they
work and are occasionally worth rerunning, and documented here so
`pipeline/config.py` does not have to carry the argument for each one inline.

| Knob | Default | |
|---|---|---|
| `EXOPD`, `EXOPD_LAMBDA` | `0`, `1.25` | Generalized off-policy distillation: maximize `λ·log(π*/π_ref) − KL(π_θ‖π_ref)` against the adapter-disabled base as reference. Student-rollout only. |
| `ADAPTIVE_KL` | `0` | Blend forward and reverse KL over the same tokens, annealing from pure forward at epoch 1 to pure reverse at the last. Overrides the fixed-direction flags. |
| `TEACHER_ROLLOUT_REVERSE` | `0` | Apply reverse KL to *teacher*-decoded tokens instead of student rollouts. Ignored while `STUDENT_ROLLOUT=1`. |
| `TEACHER_ROLLOUT_BIASED`, `..._K` | `0`, `20` | Restrict that sum to the teacher's top-k. Required on the vLLM path, which only has top-k. |
| `STUDENT_ROLLOUT_BIASED`, `..._K` | `0`, `20` | The same top-k restriction on the student-rollout path. |
| `NUM_STUDENT_ROLLOUTS` | `1` | Rollouts sampled per failure. |
| `STUDENT_ROLLOUT_TEMPERATURE`, `..._TOP_P` | `1.0`, `0.9` | Must stay stochastic or the rollouts are identical. |
| `FROZEN_TEACHER` | `0` | Target the original base weights rather than the co-evolving self-teacher. Only genuinely frozen with `PROBE_TOPK_VIA_VLLM=0` on the expert path. |
| `LENGTH_PENALTY` and `LENGTH_PENALTY_*` | off | See below. |
| `STUDENT_ROLLOUT_GEN_BATCH` | `1` | **Deprecated and unused.** Rollout generation moved to vLLM, which batches internally. Retained so existing `.env` files do not break. |

`HINT_MODE` also accepts three modes the paper does not report — `partial`
(names the misused concept only), `concept` (surfaces the needed concept without
diagnosing), and `custom` (a self-contained prompt) — plus a `curriculum`
meta-mode that uses `partial` for the first half of training and `full` for the
second.

---

## Post-paper findings

Measured after the paper was written. They are presets rather than defaults, so
that `launch.py` with no `--config` always means the published method.

### `configs/tuned.env` — schedule

The paper's warmup occupied 15 of roughly 20 optimizer steps, which §7 flags as
having likely limited gains. Three changes measured better afterwards:

- `NUM_WARMUP_STEPS=2`. A run here is only tens of optimizer steps — roughly
  `NUM_EPOCHS × (wrong items per epoch) / (GRAD_ACCUM_STEPS × TRAIN_BATCH_SIZE)`
  — so warmup has to be counted against that, not against the thousands-of-steps
  budget the usual defaults assume. At 2 steps of a ~40-step run the ramp is ~5%:
  enough to keep AdamW's first updates off a cold second-moment estimate without
  spending the run below peak LR.
- `GRAD_ACCUM_STEPS=8`. At ~30 wrong items per epoch, 16 gave about 2 optimizer
  steps per epoch — too few for any LR schedule to act on, and it made the
  epoch-boundary remainder a large fraction of all steps. 8 roughly doubles the
  step count at the cost of a noisier per-step gradient, the right trade at this
  scale.
- `LR_DECAY_FLOOR=0.1`. The decay runs on *epoch* progress rather than step
  progress: the total step count is not known up front, since it depends on the
  per-epoch error rate, but `NUM_EPOCHS` is. The run then ends on small,
  converging steps instead of full-size ones.

### `configs/length-penalty.env` — length penalty

A soft penalty on rollouts running toward the token budget. A rollout that hit
`MAX_NEW_TOKENS` previously contributed nothing; this instead raises the
probability of the tokens that end the turn, weighted by

```
w_i = clamp((i/MAX_NEW_TOKENS − LENGTH_SOFT_FRAC) / (1 − LENGTH_SOFT_FRAC), 0, 1) ** LENGTH_PENALTY_POWER
```

so a normal-length rollout receives no penalty at all. `LENGTH_PENALTY_WEIGHT`
is the term to tune: `L_len` sits around 8–14 nats mid-reasoning while the KL is
0.5–2, so 0.05 puts them within a small factor of each other. Both are logged
separately.

**Watch for one reward hack:** terminating early with a guessed answer. Track
mean completion tokens against validation accuracy; if both fall, the weight is
too high.

---

## Benchmark harness configuration

Each vendored harness has its own `config/config.yaml` and is documented in its
own README. `benchmarks/FinanceReasoning/config/config.yaml` ships a trimmed
model catalog — the served checkpoint and the extraction judge. Upstream's wider
comparison lineup (Kimi, GLM, Nemotron, Qwen, o1/o3, Claude, Gemini) is not
included because no number in the paper depends on it; add entries back in the
same shape if you want them.
