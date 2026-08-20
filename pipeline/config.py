"""Centralised configuration loaded from environment variables.

Sections (in order):
  0. Experiment knobs ..... the handful of flags toggled run-to-run
  1. Bootstrap ............ .env loading + env-parsing helpers
  2. Training ............. optimizer + schedule + batching hyperparameters
  3. Compute backend ...... local GPUs vs Modal
  4. Models & serving ..... OpenRouter, vLLM (serving + process mgmt), HuggingFace
  5. LoRA ................. adapter rank/targets/merge behaviour
  6. Runtime .............. concurrency + retry/rate-limit
  7. Distillation objective probing path, teacher source, KL direction, EXOPD
  8. Logging .............. Weights & Biases
  9. Prompts .............. shared system prompt
"""
import datetime as _dt
import os
import sys
from pathlib import Path
from typing import Optional

# ══════════════════════════════════════════════════════════════════════════════
# 1. Bootstrap
# ══════════════════════════════════════════════════════════════════════════════
# Presets (configs/*.env) resolve BEFORE .env, so a run's arm is selected by one
# file instead of a pile of shell exports. Handled here rather than in launch.py
# because every entrypoint — launch.py, run_pipeline.py, `python -m opsd.run`,
# `python -m sft.run` — imports this module, and several import it at module top
# before their own argument parsing runs.
#
# Precedence, strongest first: shell environment > preset > .env > code default.
# All three file layers use setdefault, so whoever writes a key first wins; the
# preset chain is therefore applied child-before-parent.
_CONFIG_DIR = Path(__file__).parent.parent / "configs"


def _read_env_file(path: Path) -> dict:
    """Parse a KEY=value file, tolerating comments, blanks and quoted values."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        out[k.strip()] = v
    return out


def _preset_path() -> Optional[Path]:
    """The preset named by `--config <path>` or $PIPELINE_CONFIG, if any."""
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if arg.startswith("--config="):
            return Path(arg.split("=", 1)[1])
    env = os.getenv("PIPELINE_CONFIG", "").strip()
    return Path(env) if env else None


def _load_preset_chain(start: Path) -> None:
    """Apply `start` and everything it EXTENDS, child first so children win.

    A bare filename in EXTENDS resolves against configs/, so `EXTENDS=finsd.env`
    works regardless of the working directory. Cycles raise rather than hang.
    """
    seen, path, applied = set(), start, []
    while path is not None:
        resolved = path if path.is_absolute() else Path.cwd() / path
        if not resolved.exists() and not path.is_absolute():
            resolved = _CONFIG_DIR / path.name
        if not resolved.exists():
            raise FileNotFoundError(f"preset not found: {path}")
        key = str(resolved.resolve())
        if key in seen:
            raise ValueError(f"EXTENDS cycle in preset chain at {resolved}")
        seen.add(key)
        values = _read_env_file(resolved)
        applied.append(resolved.name)
        parent = values.pop("EXTENDS", "").strip()
        for k, v in values.items():
            os.environ.setdefault(k, v)
        path = Path(parent) if parent else None
    print(f"[config] preset: {' <- '.join(applied)}", flush=True)


_preset = _preset_path()
if _preset is not None:
    _load_preset_chain(_preset)

_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            _v = _v.strip()
            # Strip matching surrounding quotes (KEY="value" / KEY='value')
            if len(_v) >= 2 and _v[0] == _v[-1] and _v[0] in ("'", '"'):
                _v = _v[1:-1]
            os.environ.setdefault(_k.strip(), _v)


def _env_flag(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() not in ("0", "false", "no", "")


# ══════════════════════════════════════════════════════════════════════════════
# 0. Experiment knobs (the handful toggled run-to-run)
# ══════════════════════════════════════════════════════════════════════════════
# Quick-access mirror of knobs documented in full elsewhere (HINT_MODEL/
# HINT_VIA_VLLM/HINT_MODE → §4, STUDENT_ROLLOUT → §7, LORA_TARGET_EXPERTS → §5,
# LOG_ENTROPY → §8, SEED → §2). Defined once, here; later sections just reference
# these values.

# The fixed OpenRouter teacher that writes the hint (see §4). Overridden per item
# by HINT_VIA_VLLM, which serves the hint from the live student instead.
HINT_MODEL: str = os.getenv("OPENROUTER_HINT_MODEL", "openai/gpt-oss-120b")

# Serve the hint from the live vLLM policy (current student weights) instead of
# the fixed OpenRouter HINT_MODEL, so the hint co-evolves with the student.
HINT_VIA_VLLM: bool = _env_flag("HINT_VIA_VLLM", True)

# How much guidance the generated hint reveals — vague/partial/concept/full/
# custom/curriculum (see §4 and pipeline/hint_script.py for the per-mode spec).
# "full" is the paper's Fin-SD hint ("Your logic diverges when X... Assume Y").
# "vague" emits exactly "Your mistake is conceptual." — that is the PLACEBO arm
# of §5.3, not Fin-SD. Use configs/placebo.env to select it deliberately.
HINT_MODE: str = os.getenv("HINT_MODE", "full").strip().lower()

# On-policy reverse-KL objective: the student samples its own rollouts and the
# loss is reverse KL(student ‖ teacher) over them (vs. the default teacher-rollout
# forward-KL path). See §7.
STUDENT_ROLLOUT: bool = _env_flag("STUDENT_ROLLOUT", True)

# Adapt the MoE experts (with profiled per-expert targeting) instead of attention
# only. See §5.
LORA_TARGET_EXPERTS: bool = _env_flag("LORA_TARGET_EXPERTS", True)

# Log the mean per-token predictive entropy (nats) of the student — and, on the
# full-vocab path, the teacher — alongside the per-step loss. Off by default: it
# adds a full-vocab softmax + reduction over the (N, V) logits each step. See §8.
LOG_ENTROPY: bool = _env_flag("LOG_ENTROPY", False)

# Master seed for reproducibility (>=0 seeds Python/NumPy/torch + vLLM; -1 = off).
SEED: int = int(os.getenv("SEED", "7"))


# ══════════════════════════════════════════════════════════════════════════════
# 2. Training hyperparameters
# ══════════════════════════════════════════════════════════════════════════════
# SEED (flag defined in §0): seeds Python/NumPy/torch (utils.set_global_seed,
# called once at run start) plus the vLLM server and per-request sampling; -1
# disables all seeding. Does NOT control the train/val split (hardcoded in
# run_pipeline.main so every seed scores the same val set). Even fixed, vLLM is
# seed- not bit-exact reproducible — batching can still perturb floating-point
# order. Seed set used across sweeps: 2, 7, 27, 42, 72.
LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "2e-5"))
# Cosine warmup over optimizer steps, then cosine DECAY over epochs (see
# optimization_script._lr_lambda). A run here is only tens of optimizer steps
# — roughly NUM_EPOCHS × (wrong items per epoch) / (GRAD_ACCUM_STEPS ×
# TRAIN_BATCH_SIZE) — so warmup has to be counted against that, not against the
# thousands-of-steps budget the usual defaults assume. 15 is the paper value
# (Table 1); §7 notes it occupied 15 of ~20 optimizer steps and likely limited
# gains. A shorter ramp measured better post-paper — see docs/CONFIG.md,
# "Post-paper findings", and configs/tuned.env.
NUM_WARMUP_STEPS: int = int(os.getenv("NUM_WARMUP_STEPS", "15"))
# Floor of the post-warmup cosine decay, as a fraction of LEARNING_RATE. The
# decay runs on EPOCH progress, not step progress: the total step count is not
# known up front (it depends on the per-epoch error rate) but NUM_EPOCHS is, so
# the schedule anneals against the one horizon that is known — and the run ends
# on small, converging steps instead of full-size ones. 1.0 disables the decay
# (warmup-then-constant), which is what the paper's runs used — Table 1 lists a
# cosine schedule with no decay floor. See docs/CONFIG.md for the post-paper
# measurement behind the 0.1 floor.
LR_DECAY_FLOOR: float = float(os.getenv("LR_DECAY_FLOOR", "1.0"))
# beta2=0.95 (~20-step horizon) rather than 0.999 (~1000-step) to adapt within the run's actual budget.
ADAM_BETA1: float = float(os.getenv("ADAM_BETA1", "0.9"))
ADAM_BETA2: float = float(os.getenv("ADAM_BETA2", "0.95"))
# AdamW decoupled weight decay; 0.0 disables regularization.
WEIGHT_DECAY: float = float(os.getenv("WEIGHT_DECAY", "0.0"))
# Micro-batches per optimizer step. Sized against the fact that only WRONG items
# train: at ~30 wrong items/epoch, 16 gave ~2 steps/epoch (~20 for the whole run),
# which is too few for any LR schedule to act on and makes the epoch-boundary
# flush_gradients() remainder a large fraction of all steps. 16 is the paper
# value: × TRAIN_BATCH_SIZE=1 it gives Table 1's effective batch size of 16.
# 8 roughly doubles the step count at the cost of a noisier per-step gradient,
# which measured better post-paper — see docs/CONFIG.md and configs/tuned.env.
GRAD_ACCUM_STEPS: int = int(os.getenv("GRAD_ACCUM_STEPS", "16"))
# Items per forward/backward pass; effective batch = TRAIN_BATCH_SIZE ×
# GRAD_ACCUM_STEPS. Only the top-k probing path batches — the full-vocab HF path
# (PROBE_TOPK_VIA_VLLM=0) always runs per item. Memory scales ~linearly with this
# (the (B, N, V) logit + backward buffer dominates), so raise cautiously.
TRAIN_BATCH_SIZE: int = int(os.getenv("TRAIN_BATCH_SIZE", "1"))
NUM_EPOCHS: int = int(os.getenv("NUM_EPOCHS", "10"))
# Skip training on items the hint model classifies as arithmetic/rounding slips
# (correct setup, bad math): the reasoning isn't the lesson to teach. These items
# still count as incorrect for accuracy; only the weight update is suppressed.
SKIP_ARITHMETIC_ERRORS: bool = _env_flag("SKIP_ARITHMETIC_ERRORS", True)
TRAIN_VAL_SPLIT: float = float(os.getenv("TRAIN_VAL_SPLIT", "0.8"))
MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "8192"))
# Cap for the probe's "with hint" completion only. This bounds N (the number of
# completion tokens), and hence the (N, V) student-logits tensor whose backward
# buffer dominates GPU memory — separate from MAX_NEW_TOKENS so the initial
# (no-grad) reasoning rollout can stay long enough to reach the mistake.
MAX_PROBE_TOKENS: int = int(os.getenv("MAX_PROBE_TOKENS", "100"))
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))

# ══════════════════════════════════════════════════════════════════════════════
# 3. Compute backend (local GPUs vs Modal)
# ══════════════════════════════════════════════════════════════════════════════
# USE_MODAL=1 runs the whole pipeline inside a Modal GPU container (see
# modal_app.py + launch.py) instead of on local CUDA devices. Consumed only by
# launch.py at dispatch time; the pipeline itself is host-agnostic.
USE_MODAL: bool = _env_flag("USE_MODAL", False)
MODAL_GPU: str = os.getenv("MODAL_GPU", "A100-80GB:8")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Models & serving
# ══════════════════════════════════════════════════════════════════════════════

# ── OpenRouter (eval / hint / state models) ─────────────────────────────────
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL: str = "https://openrouter.ai/api/v1/chat/completions"
EVAL_MODEL: str = os.getenv("OPENROUTER_EVAL_MODEL", "deepseek/deepseek-v4-flash")
# HINT_MODEL (flag defined in §0): OPENROUTER_HINT_MODEL, the fixed OpenRouter
# teacher that writes the hint unless HINT_VIA_VLLM serves it from the student.
# How much guidance the generated hint reveals (see pipeline/hint_script.py):
#   "vague"      — states only whether the mistake was arithmetic or conceptual
#   "partial"    — names only the concept that was misused, not the correct interpretation
#   "concept"    — returns the relevant verbatim problem text, or the applicable concept from standard practice
#   "full"       — a 2-3 sentence hint that states the correct premise/assumption
#   "custom"     — self-contained "full"-style prompt with stricter leakage-calibration rules
#   "curriculum" — partial for the first half of epochs, full for the second half
# HINT_MODE (flag defined in §0).
# HINT_VIA_VLLM (flag defined in §0): serves the hint from the active served
# model (inference_script.get_active_model(), else VLLM_MODEL) instead of an
# external teacher, so it co-evolves with the student. Tracks last-epoch merged
# weights on the merge+restart path (same staleness as the vLLM top-k teacher).
# Bounded by MAX_NEW_TOKENS rather than OpenRouter's larger budget, since the
# local server is capped at VLLM_MAX_MODEL_LEN.
STATE_MODEL: str = os.getenv("OPENROUTER_STATE_MODEL", "openai/gpt-5.4-mini")

# ── vLLM: OpenAI-compatible serving endpoint ────────────────────────────────
VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
VLLM_MODEL: str = os.getenv("VLLM_MODEL", "unsloth/gpt-oss-120b")
VLLM_API_KEY: str = os.getenv("VLLM_API_KEY", "EMPTY")

# ── vLLM: process management (pipeline-owned server) ────────────────────────
# gpt-oss packs experts as fused 3-D params with no per-expert modules, so a
# trained expert LoRA cannot be hot-swapped into vLLM. The pipeline therefore
# always owns the vLLM process: it serves the base model, then at each epoch
# boundary it restarts on a freshly merged full checkpoint (see
# pipeline/vllm_server.py). Do not start `vllm serve` yourself — it would collide
# on the port.
# GPUs the vLLM subprocess may use (CUDA_VISIBLE_DEVICES). Keep DISJOINT from the
# trainer's block (its CUDA_VISIBLE_DEVICES — see utils.configure_train_gpus): the
# HF student spreads over every card visible to it, and its per-GPU ceilings are
# read off those cards' free VRAM, which a co-located vLLM has already claimed.
# NOTE: for the default 20B BF16 model, launch.py overrides this at startup —
# it auto-picks the next two idle cards (first -> vLLM here, second -> trainer)
# and sets VLLM_TENSOR_PARALLEL=1, so this "0,1,2,3" default only applies to a
# non-20B run or when CUDA_VISIBLE_DEVICES is pinned. See launch._autoselect_20b_gpus.
#
# FOUR cards, not two, because of what this server has to serve. On the expert-LoRA
# path vLLM is restarted at each epoch boundary on the MERGED full checkpoint
# (optimization_script.save_merged_model — fused-MoE experts have no module for a
# hot-swapped adapter), and for the 120B that checkpoint is ~218 GiB of BF16. Two
# 80 GB cards cannot hold it at any gpu-memory-utilization, so the previous 2-card
# default did not fail at configuration time — it failed at the FIRST epoch
# boundary, after a full epoch of training, when restart() tried to load the merge.
# At tensor-parallel 4 the weights land at ~55 GiB/card with ~17 GiB left for KV.
VLLM_GPUS: str = os.getenv("VLLM_GPUS", "0,1,2,3")
VLLM_TENSOR_PARALLEL: int = int(os.getenv("VLLM_TENSOR_PARALLEL", "4"))
VLLM_GPU_MEM_UTIL: float = float(os.getenv("VLLM_GPU_MEM_UTIL", "0.9"))
VLLM_MAX_MODEL_LEN: int = int(os.getenv("VLLM_MAX_MODEL_LEN", "16384"))
VLLM_MAX_LOGPROBS: int = int(os.getenv("VLLM_MAX_LOGPROBS", "20"))
# Extra raw args appended to `vllm serve` (space-separated), for anything not
# covered above. Example: "--enforce-eager --swap-space 8".
VLLM_SERVE_EXTRA_ARGS: str = os.getenv("VLLM_SERVE_EXTRA_ARGS", "")
# Seconds to wait for a (re)started server to become ready before giving up.
VLLM_STARTUP_TIMEOUT: int = int(os.getenv("VLLM_STARTUP_TIMEOUT", "1800"))
# The "Current date: …" line gpt-oss carries in its harmony system message. vLLM
# re-derives it from datetime.now() on EVERY REQUEST unless VLLM_SYSTEM_START_DATE
# is set (its own source flags this as non-determinism), and the HF chat template
# stamps it at render time with a non-overridable strftime_now — so a run that
# crosses midnight silently changes its own prompt, and a checkpoint served the
# day after it was trained is conditioned on a prompt one token off the one it
# learned. Resolved ONCE here (default: today), exported to the vLLM subprocess by
# vllm_server.start, and used by pipeline.prompts for the training-side render, so
# every prompt in a run agrees. Pin it in .env to compare runs across days.
SYSTEM_START_DATE: str = os.getenv(
    "VLLM_SYSTEM_START_DATE", _dt.datetime.now().strftime("%Y-%m-%d")
)
# Written back so the resolved value — not a fresh datetime.now() — is what any
# child that re-imports this module sees (the expert profiler, a serve tool, the
# vLLM subprocess). Without it each process picks its own "today" and a run that
# spans midnight ends up with two different prompts in flight. NOTE: this only
# pins the date WITHIN a process tree. A checkpoint served by a later run (e.g.
# tools/serve_checkpoint the next day) is still conditioned on that run's date
# unless VLLM_SYSTEM_START_DATE is pinned in .env — do that when a checkpoint has
# to be served identically to how it was trained.
os.environ["VLLM_SYSTEM_START_DATE"] = SYSTEM_START_DATE

# ── HuggingFace model (for probing + gradient computation) ──────────────────
# bf16 upcast of gpt-oss-120b: MXFP4 (the release format) has no backward kernels.
HF_MODEL_PATH: str = os.getenv("HF_MODEL_PATH", "unsloth/gpt-oss-120b-BF16")
HF_DEVICE: str = os.getenv("HF_DEVICE", "auto")
HF_DTYPE: str = os.getenv("HF_DTYPE", "bfloat16")
# Attention kernel the HF student is loaded under. Empty → autoselect: prefer
# flash_attention_2, else flex_attention, and never fall through to `eager`.
# gpt-oss sets _supports_sdpa = False (its attention sinks are an extra per-head
# logit SDPA's fused kernels cannot express), so transformers' OWN autoselect
# lands on eager, whose [batch, heads, seq, seq] score matrix is quadratic in
# sequence length — measured at ~30 GiB of transient VRAM per layer for the 120B
# at the 8192-token sequences the length-penalty probe forwards. Set explicitly to
# pin one ("eager" is accepted, and warned about). See pipeline/attention.py.
HF_ATTN_IMPLEMENTATION: str = os.getenv("HF_ATTN_IMPLEMENTATION", "").strip()

# ── VRAM balancing ──────────────────────────────────────────────────────────
# Per-card VRAM the trainer holds BACK from the weights, so a training step has
# somewhere to run. This is the only VRAM number the pipeline states, and every
# per-GPU ceiling is derived from it: at load time utils.resolve_max_memory reads
# each visible card's live free VRAM off the driver and caps device_map="auto" at
# `free - HF_MIN_HEADROOM_GIB`, and after the weights land utils.log_vram_budget
# re-checks the same figure against where they actually went.
#
# It is a measured, model-side quantity rather than a tuning knob: a step's
# transient cost is one MoE layer's activations plus the expert-LoRA factors PEFT
# materialises for it — ~3 GiB per card for the 120B at the 8192-token sequences
# the length-penalty probe forwards — plus AdamW's moments (~1 GiB/card, 4-card
# dense) and the fp32 logits on the lm_head card. 6 GiB is roughly 2x that.
# Raise it if a run OOMs mid-epoch anyway; every layout follows.
#
# This replaced a hand-tuned cap string ("62,62,62,62" for the 120B on 4 cards,
# collapsed to "72" for a 20B on one), which encoded an 80 GB A100 and the width
# of one trainer block in numbers nothing could re-derive: on any other box the
# caps either overcommitted a card or stranded VRAM, and a partially-occupied card
# was invisible to them. A reserve against live free VRAM is the same thing said
# in the direction that survives a change of hardware.
HF_MIN_HEADROOM_GIB: float = float(os.getenv("HF_MIN_HEADROOM_GIB", "6"))
# Escape hatch: comma-separated absolute per-GPU GiB ceilings, one per visible card
# by position ("62,62,62,62"), honored exactly as written and mapped to CUDA
# ordinals in order (extra entries beyond the visible GPU count are dropped with a
# warning). EMPTY by default, which means "derive the ceilings from the cards" as
# above — set it only to pin a placement, e.g. to reproduce an older run's exact
# sharding, or to hold a card back for something else on the box.
HF_MAX_MEMORY: str = os.getenv("HF_MAX_MEMORY", "").strip()

# ══════════════════════════════════════════════════════════════════════════════
# 5. LoRA
# ══════════════════════════════════════════════════════════════════════════════
LORA_RANK: int = int(os.getenv("LORA_RANK", "64"))
LORA_ALPHA: int = int(os.getenv("LORA_ALPHA", "128"))

# gpt-oss packs experts as 3-D nn.Parameters, not nn.Linear; requires PEFT>=0.17 and vLLM>=0.11.2.
# LORA_TARGET_EXPERTS (flag defined in §0): on by default, so the adapter covers
# the MoE experts (with the profiled per-expert targeting below), not attention only.

# gpt-oss has 128 experts, so rank//num_experts collapses to 1; pin to 2 to keep meaningful capacity.
_expert_rank_env = os.getenv("LORA_EXPERT_RANK", "8").strip()
if _expert_rank_env.lower() in ("", "0", "auto"):
    LORA_EXPERT_RANK: Optional[int] = None  # resolve to rank // num_experts at load
else:
    LORA_EXPERT_RANK = int(_expert_rank_env)

# Merge adapter before probing's greedy gen to avoid per-token LoRA delta over all experts under KV cache.
LORA_MERGE_FOR_GEN: bool = _env_flag("LORA_MERGE_FOR_GEN", True)

# Apply the expert adapter inside gpt-oss's per-expert loop (activation space)
# instead of through PEFT's fused-delta parametrization. Same arithmetic — checked
# against PEFT's own get_delta_weight to 3.6e-7 — but PEFT materialises the whole
# [128, 2880, 5760] delta AND the summed weight on every forward, which is 18.1 GiB
# of transient VRAM per MoE layer on the 120B (measured), paid on every card since
# device_map runs one layer at a time. It also makes the profiled expert selection
# an allocation rather than a mask, so frozen experts cost no parameters, no
# gradients and no optimizer moments. Set to 0 to fall back to stock PEFT (which
# the loader also does on its own if PEFT's layout stops matching).
# See pipeline/expert_lora.py.
EXPERT_LORA_ACTIVATION_SPACE: bool = _env_flag("EXPERT_LORA_ACTIVATION_SPACE", True)

# ── Profiled expert selection (Design B: per-expert, top-utilization) ────────
# On by default, so the LoRA path uses per-expert targeting.
# When on AND a profile cache exists, expert-LoRA adapts only the busiest experts
# in each layer, chosen on absolute utilization (raw router selection counts, NOT
# finance lift): tools/profile_experts.py forwards the training set through the
# frozen router, tallies per-layer/per-expert selection counts, and writes
# EXPERT_PROFILE_PATH. _load_model_and_tokenizer then keeps, per layer, the busiest
# experts whose CUMULATIVE selection mass first reaches PROFILE_EXPERT_MASS_THRESHOLD
# and freezes the LoRA adapter (grad-masked to zero) for the rest. This is a
# variable count per layer, not a fixed top-k: near-uniform early layers keep ~20+
# experts, concentrated late layers only 6–8. PEFT attaches one rank-r adapter to
# the whole fused expert tensor, so the per-expert restriction is enforced by
# masking each non-selected expert's lora_A/lora_B blocks (see
# probing_script._install_expert_masks). No cache / flag off → every expert adapts.
PROFILE_EXPERT_ACTIVATIONS: bool = _env_flag("PROFILE_EXPERT_ACTIVATIONS", True)
EXPERT_PROFILE_PATH: str = os.getenv(
    "EXPERT_PROFILE_PATH", str(Path(__file__).parent.parent / "data" / "expert_profile.json")
)
# Cumulative per-layer selection-mass fraction to cover when picking experts to
# adapt (0.85 → keep the busiest experts that together carry 85% of the layer's
# routing). Higher → more experts per layer; 1.0 → all experts. Tuned to the
# training-set profile: 0.85 keeps ~20+ experts in the near-uniform early layers
# and ~6–8 in the concentrated late ones (0.90 pushed the late tail to ~12).
PROFILE_EXPERT_MASS_THRESHOLD: float = float(os.getenv("PROFILE_EXPERT_MASS_THRESHOLD", "0.85"))
# Max training items to forward during profiling (router stats saturate quickly).
PROFILE_MAX_ITEMS: int = int(os.getenv("PROFILE_MAX_ITEMS", "256"))

# ══════════════════════════════════════════════════════════════════════════════
# 6. Runtime (concurrency + retry)
# ══════════════════════════════════════════════════════════════════════════════

# ── Concurrency ─────────────────────────────────────────────────────────────
# Items within an epoch are independent (served weights only refresh at epoch
# end), so network-bound stages — inference + teacher decode (vLLM), eval + hint
# (OpenRouter) — run concurrently across a window of items while the GPU
# forward/backward stays serialized on the main thread. MAX_CONCURRENT_ITEMS sizes
# the thread pool; the backend caps bound requests-in-flight per server (vLLM
# KV-cache budget / OpenRouter rate limits). Set all to 1 for fully-sequential
# behaviour (e.g. reproducible debugging).
MAX_CONCURRENT_ITEMS: int = int(os.getenv("MAX_CONCURRENT_ITEMS", "16"))
VLLM_MAX_CONCURRENCY: int = int(os.getenv("VLLM_MAX_CONCURRENCY", "16"))
OPENROUTER_MAX_CONCURRENCY: int = int(os.getenv("OPENROUTER_MAX_CONCURRENCY", "16"))

# ── Retry / rate-limit ──────────────────────────────────────────────────────
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "5"))
RETRY_BASE_DELAY: float = float(os.getenv("RETRY_BASE_DELAY", "1.0"))

# ══════════════════════════════════════════════════════════════════════════════
# 7. Distillation objective
#    (probing path → teacher source → KL direction → EXOPD)
# ══════════════════════════════════════════════════════════════════════════════

# ── Top-k vLLM probing (the default probing path) ───────────────────────────
# Teacher top-k logprobs come from vLLM (no HF teacher forward); requires --max-logprobs >= PROBE_TOPK_K.
PROBE_TOPK_VIA_VLLM: bool = _env_flag("PROBE_TOPK_VIA_VLLM", True)
PROBE_TOPK_K: int = int(os.getenv("PROBE_TOPK_K", "20"))

# ── Frozen teacher ──────────────────────────────────────────────────────────
# By default the teacher (with-hint) distribution comes from the live model being
# trained, conditioned on the hint — a self-distillation signal that co-evolves
# with the student. FROZEN_TEACHER instead targets the original base weights
# (adapter disabled), a fixed target. Changes the objective, not just perf.
#   • HF paths: disable the adapter for both the teacher forward and the greedy
#     decode defining the target trajectory — exact, no extra memory.
#   • vLLM top-k path: targets the un-adapted base (config.VLLM_MODEL) instead of
#     the hot-loaded adapter. Only truly frozen on the hot-swap LoRA path — on the
#     merge+restart expert-LoRA path vLLM serves merged (trained) weights, so the
#     vLLM teacher still tracks training there. Use PROBE_TOPK_VIA_VLLM=0 for a
#     guaranteed-frozen teacher with expert LoRA.
FROZEN_TEACHER: bool = _env_flag("FROZEN_TEACHER", False)

# ── On-policy reverse-KL student-rollout mode ───────────────────────────────
# Alternative to the default teacher-rollout forward-KL objective (flag defined in
# §0). The student samples NUM_STUDENT_ROLLOUTS completions from its own
# without-hint prefix on vLLM in one n-way request
# (probing_script.sample_student_rollouts), then minimises reverse KL(student ‖
# teacher) teacher-forced on the live HF model over those tokens, full vocab
# (loss_script.run_reverse_loss). Each rollout is one grad-accum micro-batch,
# running inline on HF (bypasses the top-k and forward-KL paths). Only truly
# on-policy at each epoch boundary — vLLM serves stale last-epoch weights
# mid-epoch (same staleness as PROBE_TOPK_VIA_VLLM).
NUM_STUDENT_ROLLOUTS: int = int(os.getenv("NUM_STUDENT_ROLLOUTS", "1"))
# DEPRECATED / unused: rollout generation now runs on vLLM, which batches the
# n-way decode internally, so there is no HF generation KV cache to sub-batch.
# Retained only so existing .env files that set it don't break.
STUDENT_ROLLOUT_GEN_BATCH: int = int(os.getenv("STUDENT_ROLLOUT_GEN_BATCH", "1"))
# Sampling must be stochastic (temperature != 0) so the rollouts differ.
STUDENT_ROLLOUT_TEMPERATURE: float = float(os.getenv("STUDENT_ROLLOUT_TEMPERATURE", "1.0"))
STUDENT_ROLLOUT_TOP_P: float = float(os.getenv("STUDENT_ROLLOUT_TOP_P", "0.9"))
# Restrict the reverse-KL sum to the teacher's top-k token ids per position (a
# mode-seeking partial sum — see loss_script.run_reverse_topk_loss) instead of
# the full vocab (loss_script.run_reverse_loss). The teacher's top-k is selected
# from the same with-hint logits already computed per rollout, so this adds no
# extra forward. STUDENT_ROLLOUT_BIASED_K sets k.
STUDENT_ROLLOUT_BIASED: bool = _env_flag("STUDENT_ROLLOUT_BIASED", False)
STUDENT_ROLLOUT_BIASED_K: int = int(os.getenv("STUDENT_ROLLOUT_BIASED_K", "20"))
# Default: reverse KL(student ‖ teacher) (mode-seeking) over student-sampled
# tokens. STUDENT_ROLLOUT_FORWARD instead minimises forward KL(teacher ‖ student)
# (mass-covering) on those same tokens — full-vocab loss_script.run_loss, same
# objective as the teacher-rollout path but evaluated on the student's rollout
# positions. A change of objective, not a perf knob; ignored unless STUDENT_ROLLOUT
# is set, always full-vocab (STUDENT_ROLLOUT_BIASED is ignored), and overridden by
# EXOPD/ADAPTIVE_KL.
STUDENT_ROLLOUT_FORWARD: bool = _env_flag("STUDENT_ROLLOUT_FORWARD", False)

# ── Reverse-KL on teacher rollouts ───────────────────────────────────────────
# The teacher-rollout paths (default top-k vLLM and full-vocab HF) greedily
# decode one completion from the with-hint prefix and by default minimise forward
# KL(teacher ‖ student) over those tokens (run_loss / run_topk_loss).
# TEACHER_ROLLOUT_REVERSE instead applies the same reverse/biased-reverse
# KL(student ‖ teacher) objective the student-rollout path uses (run_reverse_loss
# / run_reverse_topk_loss), evaluated on the teacher-decoded tokens. A change of
# objective, not a perf knob — reverse KL is mode-seeking where forward is
# mass-covering, even on the same positions.
#
# Independent of STUDENT_ROLLOUT: that flag swaps the rollout SOURCE (student vs
# teacher tokens) and forces reverse KL; this flag swaps the LOSS DIRECTION on the
# teacher-rollout source. When STUDENT_ROLLOUT is on, its path runs instead and
# this flag is ignored.
#
#   • TEACHER_ROLLOUT_BIASED selects the biased (teacher-top-k-renormalised) reverse
#     KL over full-vocab; TEACHER_ROLLOUT_BIASED_K sets k.
#   • The vLLM top-k path only has the teacher's top-k, not its full distribution,
#     so full-vocab reverse KL needs TEACHER_ROLLOUT_BIASED there. The full-vocab HF
#     path (PROBE_TOPK_VIA_VLLM=0) supports both.
# Ignored when STUDENT_ROLLOUT is on (its path runs instead) and overridden by
# ADAPTIVE_KL (see §7 for the full teacher-vs-student rollout interaction).
TEACHER_ROLLOUT_REVERSE: bool = _env_flag("TEACHER_ROLLOUT_REVERSE", False)
TEACHER_ROLLOUT_BIASED: bool = _env_flag("TEACHER_ROLLOUT_BIASED", False)
TEACHER_ROLLOUT_BIASED_K: int = int(os.getenv("TEACHER_ROLLOUT_BIASED_K", "20"))

# ── Generalized Off-Policy Distillation (EXOPD) ──────────────────────────────
# Student-rollout-only objective (requires STUDENT_ROLLOUT=1):
#   max E_{y~π_θ}[ λ·log(π*/π_ref) − KL(π_θ ‖ π_ref) ]
# π* = with-hint teacher logits; π_ref = base (adapter-disabled) without-hint
# logits (the BF16 base is a lossless upcast of the released MXFP4 weights);
# π_θ = live student. Full-vocab (loss_script.run_exopd_loss); adds one extra
# no-grad forward per rollout, no separate model or extra VRAM needed.
EXOPD: bool = _env_flag("EXOPD", False)
EXOPD_LAMBDA: float = float(os.getenv("EXOPD_LAMBDA", "1.25"))

# ── Adaptive forward↔reverse KL annealing ────────────────────────────────────
# Instead of committing to one fixed direction (forward KL, mass-covering; or
# reverse KL, mode-seeking), blends both over the same tokens with the mix
# annealed linearly across epochs — 1·forward+0·reverse at epoch 1 down to
# 0·forward+1·reverse at the final epoch (adaptive_kl_forward_weight). Applies to
# both teacher rollouts (full-vocab HF + vLLM top-k) and student rollouts
# (STUDENT_ROLLOUT); top-k paths blend over the teacher's renormalised top-k
# support, selected by the same STUDENT_ROLLOUT_BIASED / TEACHER_ROLLOUT_BIASED flags.
# Overrides TEACHER_ROLLOUT_REVERSE; incompatible with EXOPD. See
# loss_script.run_adaptive_loss / run_adaptive_topk_loss /
# run_adaptive_topk_loss_from_teacher.
ADAPTIVE_KL: bool = _env_flag("ADAPTIVE_KL", False)


def adaptive_kl_forward_weight(epoch: int) -> float:
    """Forward-KL weight in adaptive-KL mode at the given (1-based) epoch.

    Anneals linearly from 1.0 (pure mass-covering forward KL) at epoch 1 to 0.0
    (pure mode-seeking reverse KL) at the final epoch; the reverse-KL weight is
    1 − this. With NUM_EPOCHS ≤ 1 there is no schedule to run, so it stays at the
    1.0 start (pure forward).
    """
    if NUM_EPOCHS <= 1:
        return 1.0
    progress = (epoch - 1) / (NUM_EPOCHS - 1)
    return max(0.0, min(1.0, 1.0 - progress))


# ── Token-budget (length) penalty ────────────────────────────────────────────
# A rollout that runs into the MAX_NEW_TOKENS budget used to contribute NOTHING:
# it is cut off mid-thought, so it has no <answer>, is graded wrong, and is then
# sent down the hint → state-finder path looking for "the sentence where the
# mistake happened" — but the failure was not finishing, not a mistake. The hint
# model's quote is therefore unanchored, state_finder rejects it, and the item
# drops out as an 'error'. Items that merely came CLOSE to the budget while
# answering correctly were skipped outright (correct ⇒ no update). Either way the
# model got no signal about length.
#
# LENGTH_PENALTY adds a term that supplies exactly that signal. Over the tail of
# an over-budget rollout it MAXIMISES the probability of the tokens that end the
# assistant turn (harmony <|end|> / <|return|>):
#
#     L_len = (1/M) Σ_i  w_i · ( −log Σ_{t ∈ end-tokens} π_θ(t | y_<i) )
#     w_i   = clamp((i/MAX_NEW_TOKENS − SOFT_FRAC) / (1 − SOFT_FRAC), 0, 1) ** POWER
#
# and the item's loss becomes L_KL + LENGTH_PENALTY_WEIGHT · L_len.
#
# Design notes (this term is easy to get wrong):
#   • It is a POSITIVE objective on terminating, not a penalty on the tokens the
#     model emitted. Pushing down the likelihood of what was generated
#     (unlikelihood / negative REINFORCE) has an unbounded minimiser and says
#     nothing about what to do instead; this says "you could have stopped here".
#   • w_i is exactly 0 below SOFT_FRAC, so a normal-length rollout receives no
#     length gradient at all and hard problems keep the room to reason. It also
#     means the term can never be satisfied by answering immediately — the
#     degenerate "emit <answer> at token 5" optimum is not reachable through it.
#   • The POWER=2 ramp is what makes the penalty sharp: at 80% of budget w=0.11,
#     at 95% w=0.69, at 100% w=1.
#   • Mean over the window, not sum: severity must come from w_i, not from how
#     many tokens overran, or the effective LR would scale with rollout length.
#   • Uncapped −log p is fine here, and deliberately so: it is a cross-entropy, so
#     its gradient w.r.t. the logits is bounded (‖·‖₁ ≤ 2 per position) however
#     small p gets. A hard clamp on the value would instead zero the gradient
#     exactly at the positions where the model is most determined to keep going.
#     Magnitude is controlled by LENGTH_PENALTY_WEIGHT, which is what to tune.
#   • Watch for the one real reward hack: terminating early with a guessed answer.
#     Track epoch_mean_completion_tokens against val accuracy; if both fall, the
#     weight is too high.
# Added AFTER the paper; no reported result uses it. Off by default so the
# defaults match the published method — enable with configs/length-penalty.env.
LENGTH_PENALTY: bool = _env_flag("LENGTH_PENALTY", False)
# Weight on L_len relative to the KL. L_len sits around −log p(end) ≈ 8–14 nats at
# an arbitrary mid-reasoning position while the KL is ~0.5–2, so 0.05 puts the two
# terms within a small factor of each other. Both are logged separately (loss /
# length_loss) — tune so the weighted term is ~0.1–0.3× the KL early on.
LENGTH_PENALTY_WEIGHT: float = float(os.getenv("LENGTH_PENALTY_WEIGHT", "0.05"))
# Fraction of MAX_NEW_TOKENS at which the penalty starts to bite (w_i > 0).
LENGTH_SOFT_FRAC: float = float(os.getenv("LENGTH_SOFT_FRAC", "0.7"))
# Exponent of the ramp between SOFT_FRAC and the budget. >1 concentrates the
# penalty near the budget; 1.0 is a linear ramp.
LENGTH_PENALTY_POWER: float = float(os.getenv("LENGTH_PENALTY_POWER", "2.0"))
# How many trailing rollout positions carry the term. The scored positions need a
# grad-tracked forward over the whole rollout (prompt + up to MAX_NEW_TOKENS
# tokens), so this caps the (M, V) logits and their backward buffer — not the
# forward itself, which is the real cost and is why gradient checkpointing (on by
# default at load) matters here. These are the last M positions, i.e. the ones
# with the largest w_i.
LENGTH_PENALTY_WINDOW: int = int(os.getenv("LENGTH_PENALTY_WINDOW", "128"))
# Length probes per batched forward. 1 by default: unlike a KL probe (prefix +
# ≤MAX_PROBE_TOKENS) these sequences are up to MAX_NEW_TOKENS long, and
# _batched_completion_logits left-pads to the longest in the batch, so batching
# them multiplies the largest activation footprint in the run.
LENGTH_PENALTY_BATCH: int = int(os.getenv("LENGTH_PENALTY_BATCH", "1"))


def length_penalty_weight(token_index: int) -> float:
    """Per-position severity w_i for the token at 0-based rollout index `token_index`.

    Zero at or below LENGTH_SOFT_FRAC of MAX_NEW_TOKENS, then a POWER-shaped ramp
    to 1.0 at the budget. Clamped to 1.0 above it (a rollout cannot normally
    exceed the budget, but a resumed/replayed trace could).
    """
    if MAX_NEW_TOKENS <= 0:
        return 0.0
    frac = (token_index + 1) / MAX_NEW_TOKENS
    if frac <= LENGTH_SOFT_FRAC:
        return 0.0
    if LENGTH_SOFT_FRAC >= 1.0:
        return 1.0
    ramp = min(1.0, (frac - LENGTH_SOFT_FRAC) / (1.0 - LENGTH_SOFT_FRAC))
    return float(ramp ** LENGTH_PENALTY_POWER)

# ══════════════════════════════════════════════════════════════════════════════
# 8. Logging (Weights & Biases)
# ══════════════════════════════════════════════════════════════════════════════
# W&B is OFF by default so a fresh clone trains without a Weights & Biases
# account. "disabled" is wandb's own no-op mode: wandb.init() returns a dummy run
# and every wandb.log() is a no-op, with no login and no network — so the ~18
# call sites across run_pipeline/opsd/sft need no guards. Export it here rather
# than only reading it, so the wandb library sees it however config was imported.
# Set WANDB_MODE=online in .env to actually log a run.
WANDB_MODE: str = os.getenv("WANDB_MODE", "disabled").strip().lower()
os.environ["WANDB_MODE"] = WANDB_MODE
WANDB_ENTITY: str = os.getenv("WANDB_ENTITY", "")
WANDB_PROJECT: str = os.getenv("WANDB_PROJECT", "pipeline")
WANDB_RUN_NAME: str = os.getenv("WANDB_RUN_NAME", "")

# LOG_ENTROPY (flag defined in §0): log the mean per-token predictive entropy
# (nats) of the student — and, on the full-vocab path, the teacher — alongside the
# per-step loss (see loss_script.token_entropy). Off by default: it adds a
# full-vocab softmax + reduction over the (N, V) logits each training step. On the
# vLLM top-k path only the student forward is full-vocab, so only student entropy
# is logged there.

# ══════════════════════════════════════════════════════════════════════════════
# 9. Prompts (shared system prompt, used by both inference and probing)
# ══════════════════════════════════════════════════════════════════════════════
INFERENCE_SYSTEM_PROMPT = """\
Act as a senior financial analyst and expert. Before providing your final output, you must explicitly think through the problem step-by-step. Select and apply the most relevant advanced reasoning techniques from the framework below to guarantee absolute accuracy:

1. Systematic Analysis (SA): Deconstruct the problem's structure, identifying all inputs, variables, variables over time, and core financial objectives.
2. Method Reuse (MR): Map the problem to established financial models or formulas (e.g., NPV, DCF, Black-Scholes, CAPM, Portfolio Optimization) where applicable.
3. Divide and Conquer (DC): Break complex multi-stage financial calculations into isolated, sequential sub-steps.
4. Self-Refinement (SR): Audit your intermediate calculations. Cross-check for mathematical consistency and logical fallacies before proceeding.
5. Context Identification (CI): Align the solution with industry-specific realities (e.g., tax implications, compounding frequencies, macroeconomic assumptions).
6. Emphasizing Constraints (EC): Strictly adhere to specified constraints, including rounding rules, decimal precision, percentages, and currency units.

Format your response EXACTLY as:
<reasoning>
[your step-by-step reasoning]
</reasoning>
<answer>
[your concise final answer]
</answer>
CRITICAL OUTPUT REQUIREMENT: In your final answer, do not include any introductory text, reasoning, step-by-step explanation, or concluding remarks. It must consist of exactly and only the final answer itself (whether it is a number, word, or phrase), with no other text whatsoever.
"""
