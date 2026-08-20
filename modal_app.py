"""Run the whole training pipeline on Modal GPUs instead of local ones.

This wraps the existing `run_pipeline.main()` unchanged: the vLLM subprocess and
the in-process HF trainer both run *inside one Modal container* that mirrors the
local 8-GPU box (vLLM on GPUs 0-1, HF training on 2-7). Only the host changes.

Usage (normally via `python launch.py` with USE_MODAL=1, or directly):
    modal run modal_app.py

Prerequisites:
  • `pip install modal && modal token new`
  • Two Modal secrets:
        modal secret create openrouter OPENROUTER_API_KEY=sk-or-...
        modal secret create wandb      WANDB_API_KEY=...
  • GPU type comes from MODAL_GPU (default A100-80GB:8); see pipeline/config.py.

Storage (see the plan): two Modal Volumes back the heavy paths so a cold
container never re-downloads the ~240GB base model and artifacts persist:
  • hf-cache   → HF_HOME/HF_HUB_CACHE (base weights, downloaded once, reused)
  • checkpoints → CHECKPOINT_DIR (adapters + any merged expert-LoRA checkpoints)
"""
import os

import modal

from pipeline import config

# ── Storage ────────────────────────────────────────────────────────────────
HF_CACHE_DIR = "/cache/huggingface"
CHECKPOINT_DIR = "/checkpoints"
hf_cache_vol = modal.Volume.from_name("pipeline-hf-cache", create_if_missing=True)
checkpoint_vol = modal.Volume.from_name("pipeline-checkpoints", create_if_missing=True)

# ── Non-secret config forwarded from the launching host ────────────────────
# `import config` above already loaded .env into os.environ (via setdefault), so
# the whole .env plus anything the user exported on the CLI is visible here.
# Forward every .env key into the container so we don't hand-maintain a list;
# the container has no .env (image build ignores it), so this is its only source.
#
# Excluded:
#   • secrets — delivered via named Modal secrets, not this plaintext dict.
#   • Modal control-plane vars — only meaningful on the launching host.
_EXCLUDE = {
    "OPENROUTER_API_KEY", "WANDB_API_KEY",  # → Secret.from_name(...)
    "USE_MODAL", "MODAL_GPU",               # host-side dispatch knobs
}


def _env_file_keys() -> set[str]:
    """Keys defined in the repo-root .env (same parse as config.py)."""
    from pathlib import Path

    env_path = Path(__file__).parent / ".env"
    keys: set[str] = set()
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                keys.add(line.partition("=")[0].strip())
    return keys


# Read values from os.environ (so CLI overrides of a .env key win), keyed off the
# .env's key set. setdefault in config.py already merged .env under real env vars.
_forwarded = {
    k: os.environ[k]
    for k in _env_file_keys() - _EXCLUDE
    if k in os.environ
}

# ── Image ──────────────────────────────────────────────────────────────────
# The pinned requirements bundle their own CUDA *runtime* wheels (nvidia-*), but
# vLLM/FlashInfer/Torch JIT-compile kernels at runtime and need `nvcc` + a real
# CUDA toolkit at CUDA_HOME. A slim base has neither, hence the classic
# "Could not find nvcc / cuda_home='/usr/local/cuda' doesn't exist". The NVIDIA
# `devel` image ships the full toolkit at /usr/local/cuda with nvcc on PATH; the
# tag matches our cu13 wheels (torch 2.11 + cuda-toolkit 13.0.2 in requirements).
# `add_python` provides the interpreter the base image lacks; `.entrypoint([])`
# clears the base ENTRYPOINT so Modal controls the container.
# Code is added last (local files can't precede build steps).
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(requirements=["requirements.txt"])
    .env(
        {
            # Toolkit location so Torch/vLLM find nvcc for runtime compilation.
            "CUDA_HOME": "/usr/local/cuda",
            # Cache base weights on the hf-cache Volume, shared by vLLM + HF trainer.
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_CACHE": f"{HF_CACHE_DIR}/hub",
            # Persist adapters / merged checkpoints on the checkpoints Volume.
            "CHECKPOINT_DIR": CHECKPOINT_DIR,
        }
    )
    .add_local_dir(
        ".",
        "/root",
        ignore=[
            "venv/**", ".git/**", "checkpoints/**", "wandb/**",
            "**/__pycache__/**", ".env", "*.pyc",
        ],
    )
)

app = modal.App("pipeline-train", image=image)


@app.function(
    gpu=config.MODAL_GPU,
    memory=256 * 1024,  # request 256 GiB RAM (Modal takes memory in MiB)
    timeout=24 * 60 * 60,  # long training loop; Modal's max function timeout
    # Auto-restart on preemption / unexpected container death. The restarted
    # attempt reuses the same run_id arg (below) → same CHECKPOINT_DIR, so it
    # resumes from the last committed epoch (see run_pipeline._find_resume_epoch)
    # rather than starting over. A manual `modal app stop` / Ctrl-C cancels the
    # app and does NOT retry. backoff_coefficient=1.0 keeps the delay flat.
    retries=modal.Retries(max_retries=10, backoff_coefficient=1.0, initial_delay=30.0),
    volumes={HF_CACHE_DIR: hf_cache_vol, CHECKPOINT_DIR: checkpoint_vol},
    secrets=[
        modal.Secret.from_name("openrouter"),  # OPENROUTER_API_KEY
        modal.Secret.from_name("wandb"),        # WANDB_API_KEY
        modal.Secret.from_dict(_forwarded),     # W&B settings + pipeline knobs
    ],
)
def train(run_id: str) -> None:
    import sys

    # Stable run identity across automatic retries: run_pipeline keys its
    # CHECKPOINT_DIR (and W&B run) off RUN_ID, so forwarding the same value on
    # every retry is what lets a preempted run resume from its committed epochs.
    # Set before importing run_pipeline (which reads it at import time).
    os.environ["RUN_ID"] = run_id

    # Mirror the local box: vLLM subprocess uses VLLM_GPUS (0,1); the HF trainer
    # binds the disjoint remainder. Set before importing run_pipeline (which pulls
    # in torch). Matches setup.sh and utils._FIXED_TRAIN_GPUS, leaving GPUs 0-3 for
    # the vLLM server (config.VLLM_GPUS). Per-card memory ceilings are not pinned
    # here — they're derived from these cards' free VRAM at load time.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4,5,6,7")

    # A retried container gets a fresh mount; reload so it sees epochs committed
    # by the preempted attempt.
    checkpoint_vol.reload()

    os.chdir("/root")
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    from run_pipeline import main

    try:
        # Commit the checkpoints Volume after each epoch's adapter is written so
        # progress is durable mid-run — a hard container kill (OOM, 24h timeout,
        # preemption) skips the `finally` below, which would otherwise be the
        # only commit and would lose every uncommitted epoch.
        main(on_checkpoint=checkpoint_vol.commit)
    finally:
        # Final flush for the normal/clean-exception exit path.
        checkpoint_vol.commit()


@app.local_entrypoint()
def entry() -> None:
    from datetime import datetime, timezone

    # Mint the run id locally so it is fixed for this launch and reused verbatim
    # by every automatic retry of this call (Modal re-invokes with the same args),
    # giving all attempts one shared, resumable CHECKPOINT_DIR.
    run_id = os.getenv("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    # `.spawn()`, not `.remote()`: submit the input and return immediately instead
    # of blocking here for hours. `.remote()` keeps this `modal run` client tethered
    # for the whole run, maintaining a client-side heartbeat for the input; when the
    # laptop sleeps that heartbeat drops, Modal treats the in-flight input as lost,
    # and `retries=` above spins up a *fresh* container (new vLLM server, W&B run
    # closed as "finished", resume from checkpoint) — a restart, not a detach.
    # Spawning lets the local entrypoint return in seconds so `modal run --detach`
    # exits cleanly with no long-lived client to lose, truly decoupling the job.
    call = train.spawn(run_id)
    print(f"Spawned Modal training run {run_id} (call id {call.object_id}).", flush=True)
    print(f"  Logs:   modal app logs {app.app_id}", flush=True)
