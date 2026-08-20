"""Run the SFT distillation pipeline on Modal GPUs instead of local ones.

Mirrors the root `modal_app.py`, but the container runs the SFT flow
(`sft.generate_data.generate` + `sft.train.train`) instead of
`run_pipeline.main()`. Only the host changes: vLLM (per-epoch validation) still
lands on GPUs 0-1 and the HF trainer on 2-7, exactly as the local 8-GPU box.

It reuses the SAME two Modal Volumes as the main pipeline
(`pipeline-hf-cache`, `pipeline-checkpoints`), so the ~240GB base model is
downloaded once and shared, and the distilled data + SFT adapters persist on the
checkpoints Volume across cold containers.

Usage (normally via `python -m sft.run` / `python -m sft.train` with USE_MODAL=1,
or directly):
    modal run sft/modal_app.py                        # generate + train
    modal run sft/modal_app.py --skip-generate        # train on existing data
    modal run sft/modal_app.py --skip-train --limit 20

Prerequisites (identical to the root modal_app.py):
  • `pip install modal && modal token new`
  • Two Modal secrets:
        modal secret create openrouter OPENROUTER_API_KEY=sk-or-...
        modal secret create wandb      WANDB_API_KEY=...
  • GPU type comes from MODAL_GPU (default A100-80GB:8); see pipeline/config.py.

On Modal the distilled data and outputs default to the checkpoints Volume
(SFT_DISTILL_PATH / SFT_OUTPUT_DIR below); override them, or any other knob, by
declaring the key in .env (only .env-declared keys are forwarded, matching the
root app).
"""
import os

import modal

from pipeline import config

# ── Storage (shared with the main pipeline) ────────────────────────────────
HF_CACHE_DIR = "/cache/huggingface"
CHECKPOINT_DIR = "/checkpoints"
hf_cache_vol = modal.Volume.from_name("pipeline-hf-cache", create_if_missing=True)
checkpoint_vol = modal.Volume.from_name("pipeline-checkpoints", create_if_missing=True)

# ── Non-secret config forwarded from the launching host ────────────────────
# `import config` above already loaded .env into os.environ (via setdefault), so
# the whole .env plus anything the user exported on the CLI is visible here.
# Forward every .env key into the container (the image has no .env), except:
#   • secrets — delivered via named Modal secrets, not this plaintext dict.
#   • Modal control-plane vars — only meaningful on the launching host.
_EXCLUDE = {
    "OPENROUTER_API_KEY", "WANDB_API_KEY",  # → Secret.from_name(...)
    "USE_MODAL", "MODAL_GPU",               # host-side dispatch knobs
}


def _env_file_keys() -> set[str]:
    """Keys defined in the repo-root .env (same parse as config.py)."""
    from pathlib import Path

    env_path = Path(__file__).parent.parent / ".env"
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
# Same devel CUDA base + pinned requirements as the root app (vLLM/Torch JIT need
# nvcc at CUDA_HOME). Code is added last (local files can't precede build steps).
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(requirements=["requirements.txt"])
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            # Base weights on the hf-cache Volume, shared by vLLM + HF trainer.
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_CACHE": f"{HF_CACHE_DIR}/hub",
            # Persist distilled data + adapters on the checkpoints Volume by
            # default (a forwarded .env value for either key overrides these).
            "SFT_OUTPUT_DIR": f"{CHECKPOINT_DIR}/sft/checkpoints",
            "SFT_DISTILL_PATH": f"{CHECKPOINT_DIR}/sft/data/distill.jsonl",
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

app = modal.App("sft-train", image=image)


@app.function(
    gpu=config.MODAL_GPU,
    memory=256 * 1024,  # request 256 GiB RAM (Modal takes memory in MiB)
    timeout=24 * 60 * 60,  # Modal's max function timeout
    # Generation is resumable (the JSONL on the checkpoints Volume is committed
    # below, and `generate` skips problems already present), so a preempted
    # generate stage picks up where it left off. Training has no mid-run resume,
    # so a retry restarts the SFT loop from scratch — keep the retry budget small.
    retries=modal.Retries(max_retries=2, backoff_coefficient=1.0, initial_delay=30.0),
    volumes={HF_CACHE_DIR: hf_cache_vol, CHECKPOINT_DIR: checkpoint_vol},
    secrets=[
        modal.Secret.from_name("openrouter"),  # OPENROUTER_API_KEY
        modal.Secret.from_name("wandb"),        # WANDB_API_KEY
        modal.Secret.from_dict(_forwarded),     # W&B settings + pipeline/SFT knobs
    ],
)
def run_sft(skip_generate: bool, skip_train: bool, limit: int, run_id: str) -> None:
    import sys

    # Stable run identity across automatic retries: sft.config keys this run's
    # checkpoint dir (SFT_OUTPUT_DIR/<RUN_ID>) off it, so forwarding the same
    # value on every retry keeps all attempts in one dir instead of scattering
    # partial runs. Set before importing sft.config (which reads it at import).
    os.environ["SFT_RUN_ID"] = run_id

    # Mirror the local box: vLLM (per-epoch validation) uses VLLM_GPUS (0,1); the
    # HF trainer binds the disjoint remainder. Set before anything imports torch.
    # Matches setup.sh and pipeline.utils._FIXED_TRAIN_GPUS, leaving GPUs 0-3 for the
    # vLLM server (pipeline.config.VLLM_GPUS). Per-card memory ceilings are not
    # pinned here — they're derived from these cards' free VRAM at load time.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4,5,6,7")

    # A retried container gets a fresh mount; reload so it sees data committed by
    # the preempted attempt.
    checkpoint_vol.reload()

    os.chdir("/root")
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    from pipeline.utils import setup_logging

    from sft.generate_data import generate
    from sft.train import train

    setup_logging()
    try:
        if not skip_generate:
            generate(limit=None if limit < 0 else limit)
            # Commit the distilled JSONL so it's durable before the (long,
            # non-resumable) training stage starts.
            checkpoint_vol.commit()
        if not skip_train:
            # Commit the Volume at every epoch boundary, so each epoch's adapter +
            # merged checkpoint is durable (and servable) as soon as it is written
            # rather than only when the container exits.
            train(on_checkpoint=checkpoint_vol.commit)
    finally:
        # Final flush (adapter / merged checkpoint) for the normal exit path.
        checkpoint_vol.commit()


@app.local_entrypoint()
def entry(skip_generate: bool = False, skip_train: bool = False, limit: int = -1) -> None:
    from datetime import datetime, timezone

    # Mint the run id locally so it is fixed for this launch and reused verbatim by
    # every automatic retry of this call (Modal re-invokes with the same args),
    # giving all attempts one shared checkpoint dir on the Volume.
    run_id = os.getenv("SFT_RUN_ID") or os.getenv("RUN_ID") \
        or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    # `.spawn()`, not `.remote()`: submit and return immediately so the wrapping
    # `modal run --detach` (see sft._modal_dispatch) exits cleanly with no
    # long-lived client to lose when the laptop sleeps. `limit=-1` means "no cap".
    call = run_sft.spawn(skip_generate, skip_train, limit, run_id)
    print(f"Spawned Modal SFT run {run_id} (call id {call.object_id}).", flush=True)
    print(f"  Logs:   modal app logs {app.app_id}", flush=True)
