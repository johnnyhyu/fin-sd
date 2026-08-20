"""opsd configuration.

Layers opsd-specific knobs on top of the main pipeline's config (`pipeline.config`),
reusing the student model / LoRA / vLLM / OpenRouter / system-prompt settings so
opsd serves and trains the same gpt-oss student the same way `run_pipeline.py` does.
Everything is overridable via environment variables (and `.env`, which
`pipeline.config` already loads at import).

Two kinds of knob live here:

  1. opsd-owned knobs (`OPSD_*`) that only opsd reads — the reference-solution
     build, the on-policy rollout objective, epochs, and the train/val split.

  2. Optimizer knobs consumed *inside* the shared `pipeline.optimization_script`
     (LEARNING_RATE, GRAD_ACCUM_STEPS, warmup, betas, weight decay). opsd reuses
     that module, and it reads `pipeline.config`, so to make `OPSD_*` overrides
     for those actually take effect we write them back onto the pipeline config
     object at import (see the "Optimizer" section). Left unset, they inherit the
     pipeline defaults — identical to a `run_pipeline.py` run.
"""
import os
from pathlib import Path

from pipeline import config as pc

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _flag(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() not in ("0", "false", "no", "")


# ── Data ────────────────────────────────────────────────────────────────────
# Problems to train on (the "trainingset"): a list of {number, problem, answer}.
TRAININGSET_PATH: str = os.getenv("OPSD_TRAININGSET", str(_REPO_ROOT / "data" / "trainingset.json"))
# Hint-model reference solutions land here, one JSON object per line (resumable).
REFERENCE_PATH: str = os.getenv("OPSD_REFERENCE_PATH", str(_REPO_ROOT / "opsd" / "data" / "reference.jsonl"))
# Where trained adapters / merged checkpoints are written (per-run subdir added in train.py).
OUTPUT_DIR: str = os.getenv("OPSD_OUTPUT_DIR", str(_REPO_ROOT / "opsd" / "checkpoints"))
# Keep every epoch's merged full checkpoint (`epoch_<N>_merged`) instead of
# dropping the previous one once the next is served. Off by default, mirroring
# run_pipeline: a full model per epoch is ~39 GiB (20B) / ~240 GiB (120B), and an
# opsd adapter re-merges cleanly on demand because opsd trains the pipeline's own
# PEFT tree (`python tools/serve_checkpoint.py --source opsd --epoch N`). Set to 1
# to keep each epoch pre-merged and skip that re-merge.
KEEP_EPOCH_MERGED: bool = _flag("OPSD_KEEP_EPOCH_MERGED", False)

# Fraction of problems used for training; the rest are the fixed held-out val set
# graded for answer accuracy at each epoch boundary (mirrors run_pipeline's split).
TRAIN_VAL_SPLIT: float = float(os.getenv("OPSD_TRAIN_VAL_SPLIT", str(pc.TRAIN_VAL_SPLIT)))

# ── Reference-solution build (the "hint model" over every problem) ───────────
# The hint model that writes reference solutions, served via OpenRouter.
HINT_MODEL: str = os.getenv("OPSD_HINT_MODEL", pc.HINT_MODEL)
HINT_TEMPERATURE: float = float(os.getenv("OPSD_HINT_TEMPERATURE", "0.7"))
HINT_MAX_TOKENS: int = int(os.getenv("OPSD_HINT_MAX_TOKENS", "8192"))
# On an answer mismatch, retry with the ground-truth answer appended to the prompt
# (so the model can write a solution that reaches it). If none of the attempts
# match, keep the last (answer-appended) trace so every problem still gets a
# reference. Total attempts = 1 (bare) + this many answer-appended retries.
REFERENCE_MAX_ATTEMPTS: int = int(os.getenv("OPSD_REFERENCE_MAX_ATTEMPTS", "4"))
# Thread-pool width for the reference-build teacher calls (openrouter_call also
# self-gates to OPENROUTER_MAX_CONCURRENCY on top of this).
GEN_CONCURRENCY: int = int(os.getenv("OPSD_GEN_CONCURRENCY", str(pc.OPENROUTER_MAX_CONCURRENCY)))

# ── On-policy objective (forward KL over student rollouts) ───────────────────
# Rollouts sampled per wrong problem per epoch, each capped at MAX_ROLLOUT_TOKENS.
NUM_ROLLOUTS: int = int(os.getenv("OPSD_NUM_ROLLOUTS", "1"))
MAX_ROLLOUT_TOKENS: int = int(os.getenv("OPSD_MAX_ROLLOUT_TOKENS", "1024"))
ROLLOUT_TEMPERATURE: float = float(os.getenv("OPSD_ROLLOUT_TEMPERATURE", "1.0"))
ROLLOUT_TOP_P: float = float(os.getenv("OPSD_ROLLOUT_TOP_P", "0.9"))

# ── Training schedule ───────────────────────────────────────────────────────
# Written back onto pipeline.config as well: the shared LR schedule anneals over
# NUM_EPOCHS (optimization_script._decay_factor) and reads it from there, so an
# OPSD_NUM_EPOCHS override that stayed local would decay against the wrong horizon.
NUM_EPOCHS: int = int(os.getenv("OPSD_NUM_EPOCHS", str(pc.NUM_EPOCHS)))
pc.NUM_EPOCHS = NUM_EPOCHS

# ── Optimizer (consumed inside pipeline.optimization_script) ─────────────────
# Overriding here writes back onto pipeline.config so the shared optimizer picks
# them up. Defaults inherit the pipeline's values, so an opsd run with no OPSD_*
# optimizer overrides trains with exactly the same schedule as run_pipeline.py.
pc.LEARNING_RATE = float(os.getenv("OPSD_LEARNING_RATE", str(pc.LEARNING_RATE)))
pc.GRAD_ACCUM_STEPS = int(os.getenv("OPSD_GRAD_ACCUM_STEPS", str(pc.GRAD_ACCUM_STEPS)))
pc.NUM_WARMUP_STEPS = int(os.getenv("OPSD_NUM_WARMUP_STEPS", str(pc.NUM_WARMUP_STEPS)))
pc.LR_DECAY_FLOOR = float(os.getenv("OPSD_LR_DECAY_FLOOR", str(pc.LR_DECAY_FLOOR)))
pc.WEIGHT_DECAY = float(os.getenv("OPSD_WEIGHT_DECAY", str(pc.WEIGHT_DECAY)))
pc.ADAM_BETA1 = float(os.getenv("OPSD_ADAM_BETA1", str(pc.ADAM_BETA1)))
pc.ADAM_BETA2 = float(os.getenv("OPSD_ADAM_BETA2", str(pc.ADAM_BETA2)))
# Read-back copies for logging/convenience.
LEARNING_RATE: float = pc.LEARNING_RATE
GRAD_ACCUM_STEPS: int = pc.GRAD_ACCUM_STEPS

# ── Serving / concurrency (reused from the pipeline) ─────────────────────────
LORA_TARGET_EXPERTS: bool = pc.LORA_TARGET_EXPERTS
# Concurrency for the per-epoch inference sweep + validation (network-only work).
MAX_CONCURRENT_ITEMS: int = pc.MAX_CONCURRENT_ITEMS

# ── Seeding ──────────────────────────────────────────────────────────────────
SEED: int = int(os.getenv("OPSD_SEED", str(pc.SEED)))

# ── Logging ─────────────────────────────────────────────────────────────────
USE_WANDB: bool = _flag("OPSD_WANDB", True)
WANDB_ENTITY: str = os.getenv("OPSD_WANDB_ENTITY", pc.WANDB_ENTITY)
WANDB_PROJECT: str = os.getenv("OPSD_WANDB_PROJECT", pc.WANDB_PROJECT)
WANDB_RUN_NAME: str = os.getenv("OPSD_WANDB_RUN_NAME", "")

# ── Prompts ──────────────────────────────────────────────────────────────────
# Shared system prompt (same one the student is served/graded under), used for
# both the reference build and the student/teacher probing prefixes.
SYSTEM_PROMPT: str = pc.INFERENCE_SYSTEM_PROMPT

# Teacher user message: the bare problem, followed by the reference solution and
# an instruction to re-solve it independently. `{problem}` / `{reference_solution}`
# are filled per item. The student sees only `{problem}` (no reference).
TEACHER_USER_TEMPLATE: str = (
    "{problem}\n\n"
    "Here is a reference solution:\n{reference_solution}\n\n"
    "After understanding the reference solution, please try to solve this problem "
    "using your own approach below:"
)
