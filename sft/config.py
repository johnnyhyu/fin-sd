"""SFT-pipeline configuration.

Reuses the main pipeline's config for the teacher (`HINT_MODEL`), the OpenRouter
credentials, the shared system prompt, and the student model / LoRA defaults, and
layers SFT-specific knobs (data paths, teacher-decoding, and training schedule)
on top. Everything is overridable via environment variables (and `.env`, which
`pipeline.config` already loads at import).
"""
import os
from datetime import datetime, timezone
from pathlib import Path

from pipeline import config as pc

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _flag(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() not in ("0", "false", "no", "")


# ── Data ────────────────────────────────────────────────────────────────────
# Problems to distil (the "trainingset"): a list of {number, problem, answer}.
TRAININGSET_PATH: str = os.getenv("SFT_TRAININGSET", str(_REPO_ROOT / "data" / "trainingset.json"))
# Teacher-generated (prompt, solution) pairs land here, one JSON object per line.
DISTILL_PATH: str = os.getenv("SFT_DISTILL_PATH", str(_REPO_ROOT / "sft" / "data" / "distill.jsonl"))
# Base dir holding the per-run checkpoint dirs (a RUN_ID subdir is added below).
OUTPUT_DIR: str = os.getenv("SFT_OUTPUT_DIR", str(_REPO_ROOT / "sft" / "checkpoints"))
# Keys this run's checkpoint subdir, `OUTPUT_DIR/<RUN_ID>/` — mirrors run_pipeline's
# layout so tools/serve_checkpoint.py can resolve an epoch by run (`--source sft
# --run-id … --epoch N`) and two runs sharing one OUTPUT_DIR can't overwrite each
# other's epochs. Falls back to the pipeline's RUN_ID so a launcher that sets one
# id for the whole job groups them together; else a UTC timestamp.
RUN_ID: str = (
    os.getenv("SFT_RUN_ID") or os.getenv("RUN_ID")
    or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
)

# Fraction of the distilled records held out for validation (0 → no validation).
# The held-out problems are never trained on; at each epoch boundary they are
# answered by the LIVE student (`model.generate`, see sft/validate.py) and graded
# for answer accuracy. No merge, no vLLM, no second GPU.
VAL_SPLIT: float = float(os.getenv("SFT_VAL_SPLIT", "0.1"))
# Cap the validation set size so per-epoch generation stays cheap (0 → uncapped).
# Records beyond the cap return to the training set rather than being dropped.
VAL_MAX_EXAMPLES: int = int(os.getenv("SFT_VAL_MAX_EXAMPLES", "64"))
# Prompts decoded per `generate` call. In-process validation runs under no_grad
# with the training activations freed, so the batch is bounded by the KV cache
# rather than by the backward pass — hence far wider than the training BATCH_SIZE.
# Lower it if a validation epoch OOMs on a long held-out prompt.
VAL_BATCH_SIZE: int = int(os.getenv("SFT_VAL_BATCH_SIZE", "64"))
# Generation budget per held-out problem. Matches the teacher's decode budget
# (SFT_TEACHER_MAX_TOKENS) so a student that reasons as long as its supervision
# is not scored as wrong for being cut off mid-analysis — that truncation is the
# `never reached a final message` warning sft/validate.py logs.
VAL_MAX_NEW_TOKENS: int = int(os.getenv("SFT_VAL_MAX_NEW_TOKENS", "8192"))

# ── Distillation (teacher decoding) ─────────────────────────────────────────
# The teacher is the pipeline's hint model, served via OpenRouter.
TEACHER_MODEL: str = os.getenv("SFT_TEACHER_MODEL", pc.HINT_MODEL)
TEACHER_TEMPERATURE: float = float(os.getenv("SFT_TEACHER_TEMPERATURE", "0.7"))
TEACHER_MAX_TOKENS: int = int(os.getenv("SFT_TEACHER_MAX_TOKENS", "8192"))
# Rejection sampling: only keep a teacher trace whose final answer matches the
# ground-truth answer (checked with pipeline.eval_script.run_eval). Retry a
# problem up to REJECT_MAX_ATTEMPTS times before giving up on it.
REJECT_SAMPLING: bool = _flag("SFT_REJECT_SAMPLING", True)
REJECT_MAX_ATTEMPTS: int = int(os.getenv("SFT_REJECT_MAX_ATTEMPTS", "3"))
# Drop traces whose reasoning came back empty (resampling up to
# REJECT_MAX_ATTEMPTS first). The dataset exists to supervise the reasoning, so
# an answer-only trace is worse than no example; set to 0 to keep them anyway.
REQUIRE_REASONING: bool = _flag("SFT_REQUIRE_REASONING", True)
# Thread-pool width for the teacher calls (openrouter_call self-gates to
# OPENROUTER_MAX_CONCURRENCY on top of this).
GEN_CONCURRENCY: int = int(os.getenv("SFT_GEN_CONCURRENCY", str(pc.OPENROUTER_MAX_CONCURRENCY)))

# ── Student model (reuses the pipeline's HF student + LoRA defaults) ─────────
STUDENT_MODEL: str = os.getenv("SFT_STUDENT_MODEL", pc.HF_MODEL_PATH)
HF_DEVICE: str = os.getenv("SFT_HF_DEVICE", pc.HF_DEVICE)
HF_DTYPE: str = os.getenv("SFT_HF_DTYPE", pc.HF_DTYPE)
# Empty (the default) derives per-card ceilings from live free VRAM at load time;
# set it to pin them by hand — see pipeline.config.HF_MAX_MEMORY / HF_MIN_HEADROOM_GIB.
HF_MAX_MEMORY: str = os.getenv("SFT_HF_MAX_MEMORY", pc.HF_MAX_MEMORY).strip()
# Attention kernel, forced onto the model config at load (see sft/model.py).
# Empty → autoselect: prefer flash_attention_2, else flex_attention, and never
# fall through to `eager`, whose materialized [batch, heads, seq, seq] score
# matrix is what OOMs a long example (9.4 GiB for a single 8875-token turn on the
# 20B). Set explicitly to pin one — "eager" is accepted, and warned about.
ATTN_IMPLEMENTATION: str = os.getenv("SFT_ATTN_IMPLEMENTATION", "").strip()
USE_LORA: bool = _flag("SFT_USE_LORA", True)
LORA_RANK: int = int(os.getenv("SFT_LORA_RANK", str(pc.LORA_RANK)))
LORA_ALPHA: int = int(os.getenv("SFT_LORA_ALPHA", str(pc.LORA_ALPHA)))
LORA_DROPOUT: float = float(os.getenv("SFT_LORA_DROPOUT", "0"))
LORA_TARGET_EXPERTS: bool = _flag("SFT_LORA_TARGET_EXPERTS", pc.LORA_TARGET_EXPERTS)
# Also write a `final_merged/` copy of the merged weights when training ends.
# Off by default because it duplicates the last epoch's `epoch_<N>_merged` (which
# already holds the final weights); turn it on to get the pipeline's exact
# `final` + `final_merged` pair. When on, the last epoch's merged copy is
# superseded and pruned unless KEEP_EPOCH_MERGED.
MERGE_AFTER_TRAIN: bool = _flag("SFT_MERGE_AFTER_TRAIN", False)
# Keep every epoch's merged full checkpoint (`epoch_<N>_merged`, or the epoch dir
# itself when not using LoRA) instead of pruning the previous one once the next is
# written. On by default — unlike the pipeline, an SFT adapter cannot reliably be
# re-merged after the fact (serve_checkpoint rebuilds the PEFT tree from
# `pipeline.config`, whose expert rank need not match the SFT run's), so retaining
# each epoch's merged copy is what makes every epoch servable/benchmarkable. Set
# to 0 to mirror the pipeline and keep only the newest (a full model per epoch is
# ~39 GiB for the 20B, ~240 GiB for the 120B).
KEEP_EPOCH_MERGED: bool = _flag("SFT_KEEP_EPOCH_MERGED", True)
# After training, spin up a local vLLM server on the final checkpoint (reuses
# pipeline.vllm_server; serves the merged full checkpoint under pc.VLLM_MODEL).
# Off by default; also togglable per-run with `--serve`.
SERVE_AFTER_TRAIN: bool = _flag("SFT_SERVE_AFTER_TRAIN", False)

# ── Training schedule ───────────────────────────────────────────────────────
NUM_EPOCHS: int = int(os.getenv("SFT_NUM_EPOCHS", "3"))
LEARNING_RATE: float = float(os.getenv("SFT_LEARNING_RATE", "1e-4"))
BATCH_SIZE: int = int(os.getenv("SFT_BATCH_SIZE", "1"))
GRAD_ACCUM_STEPS: int = int(os.getenv("SFT_GRAD_ACCUM_STEPS", "8"))
WARMUP_RATIO: float = float(os.getenv("SFT_WARMUP_RATIO", "0.03"))
WEIGHT_DECAY: float = float(os.getenv("SFT_WEIGHT_DECAY", "0.0"))
# Full-sequence cap (system + problem + reasoning + answer). Unlike the main
# pipeline's MAX_NEW_TOKENS (which bounds generation only, on top of the prompt),
# this bounds the whole tokenized turn, so it must leave room for the prompt too.
# Default to 2× the pipeline's generation budget so a full-length teacher trace
# plus its prompt never truncates — truncation here chops the tail (the <answer>
# block), which is exactly the span we need supervised.
MAX_SEQ_LEN: int = int(os.getenv("SFT_MAX_SEQ_LEN", str(2 * pc.MAX_NEW_TOKENS)))
# Supervise only the assistant turn (the teacher trace), ignoring the prompt
# tokens in the loss. Off by default — the loss stays over the whole sequence,
# as it has been — but turning it on is the conventional choice for a
# (problem → solution) distillation set: every example shares the same ~600-token
# system prompt, so with it off a large share of the gradient goes into
# reproducing that prompt and the problem statement rather than the solution.
MASK_PROMPT: bool = _flag("SFT_MASK_PROMPT", False)
LR_SCHEDULER: str = os.getenv("SFT_LR_SCHEDULER", "cosine")
LOGGING_STEPS: int = int(os.getenv("SFT_LOGGING_STEPS", "1"))
SAVE_STEPS: int = int(os.getenv("SFT_SAVE_STEPS", "0"))  # 0 → save once at the end
# Seeds the train/val split and HF `Trainer` (data order, dropout). Matches the
# pipeline's default so an SFT run and a pipeline run line up out of the box.
# LoRA init is seeded separately from pc.SEED — see sft.train.
SEED: int = int(os.getenv("SFT_SEED", "42"))
GRADIENT_CHECKPOINTING: bool = _flag("SFT_GRADIENT_CHECKPOINTING", True)

# ── Logging ─────────────────────────────────────────────────────────────────
USE_WANDB: bool = _flag("SFT_WANDB", True)
WANDB_PROJECT: str = os.getenv("SFT_WANDB_PROJECT", pc.WANDB_PROJECT)
WANDB_RUN_NAME: str = os.getenv("SFT_WANDB_RUN_NAME", "")

# The shared system prompt (same one the pipeline conditions inference on), so
# the distilled traces match the format the student is expected to produce.
SYSTEM_PROMPT: str = pc.INFERENCE_SYSTEM_PROMPT
