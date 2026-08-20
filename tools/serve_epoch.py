#!/usr/bin/env python3
"""Serve one main-pipeline epoch checkpoint on vLLM (for benchmarking).

The RL pipeline (``run_pipeline.py``) writes one LoRA adapter per epoch under
``CHECKPOINT_DIR/<RUN_ID>/epoch_<N>/``. To benchmark a specific epoch we do
exactly what the pipeline does at each epoch boundary on the non-expert LoRA
path: bring vLLM up on the base model (``config.VLLM_MODEL``, the quantized
release) and hot-load the epoch's adapter on top (``vllm_server.load_adapter``).
Requests then target the adapter's served name — the epoch dir name, e.g.
``epoch_4`` (see ``run_pipeline.main``).

This keeps the served footprint identical to training — the quantized base plus
a small rank-``LORA_RANK`` adapter — instead of materializing a full unquantized
merged checkpoint, which needs several times the GPU memory ``config.VLLM_GPUS``
provides and OOMs at weight load.

vLLM runs on ``config.VLLM_GPUS`` (disjoint from any HF process) and is launched
with ``--enable-lora`` (``vllm_server._serve_command`` adds it on the non-expert
LoRA path). Expert-LoRA epochs (``LORA_TARGET_EXPERTS=1``) can't be hot-loaded —
their fused-MoE experts have nothing to attach an adapter to — so this tool
refuses them; serve their pre-merged ``epoch_<N>_merged`` checkpoint instead.

Note: this is a benchmarking tool for the *main pipeline's* epoch checkpoints,
not the SFT run's ``checkpoint-<step>`` adapters (those live under
``SFT_OUTPUT_DIR`` and are named differently — use the SFT tooling for those).

Examples::

    # By epoch number, latest run under CHECKPOINT_DIR:
    python tools/serve_epoch.py --epoch 4

    # Pin the run:
    python tools/serve_epoch.py --run-id 20260717-1330 --epoch 4

    # By explicit adapter dir:
    python tools/serve_epoch.py --adapter checkpoints/20260717-1330/epoch_4

    # Resolve only, don't serve:
    python tools/serve_epoch.py --epoch 4 --no-serve
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# This tool lives in tools/; put the repo root on sys.path so `pipeline`
# imports resolve when run as `python tools/serve_epoch.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import config
from pipeline import utils
from pipeline.utils import logger, setup_logging

_REPO_ROOT = Path(__file__).resolve().parent.parent
# Base holding the per-run checkpoint dirs — mirrors run_pipeline._CHECKPOINT_BASE.
_CHECKPOINT_BASE = Path(os.getenv("CHECKPOINT_DIR", str(_REPO_ROOT / "checkpoints")))


def _reserve_gpus() -> None:
    """Reserve this instance's serving GPUs, auto-selecting idle cards if unpinned.

    Delegates to utils.configure_serve_gpus: when GPUs aren't pinned it auto-selects
    config.VLLM_TENSOR_PARALLEL idle cards (the same reserved idle-GPU finder
    launch.py uses) instead of the fixed VLLM_GPUS default, so two serve_epoch runs
    on one box land on different cards. serve_epoch only hot-loads an adapter onto a
    vLLM subprocess (no in-process HF load), so it needs no CUDA_VISIBLE_DEVICES
    pinning (set_visible=False) — the subprocess inherits config.VLLM_GPUS from
    vllm_server.start.
    """
    utils.configure_serve_gpus(set_visible=False)


def _resolve_run_dir(base: Path, run_id: str | None) -> Path:
    """Return the run dir under `base` for `--run-id`, or the most recent one."""
    if run_id:
        run_dir = base / run_id
        if not run_dir.is_dir():
            sys.exit(f"Run dir does not exist: {run_dir}")
        return run_dir

    runs = sorted(p for p in base.glob("*") if p.is_dir())
    if not runs:
        sys.exit(f"No run dirs under {base}; nothing to serve.")
    # RUN_IDs default to UTC timestamps, so the lexically-largest is the newest.
    run_dir = runs[-1]
    if len(runs) > 1:
        logger.info(
            "No --run-id; using most recent of %d runs: %s (others: %s)",
            len(runs), run_dir.name, ", ".join(r.name for r in runs[:-1]),
        )
    return run_dir


def _resolve_checkpoint(run_dir: Path, epoch: int | None, adapter: str | None) -> Path:
    """Return the checkpoint dir for `adapter` (a direct path) or `--epoch`."""
    if adapter:
        path = Path(adapter)
        if not path.is_dir():
            sys.exit(f"Checkpoint dir does not exist: {path}")
        return path

    ckpt = run_dir / f"epoch_{epoch}"
    if ckpt.is_dir():
        logger.info("Epoch %d → %s", epoch, ckpt)
        return ckpt

    avail = ", ".join(
        sorted(p.name for p in run_dir.glob("epoch_*") if not p.name.endswith("_merged"))
    ) or "none"
    sys.exit(f"No epoch_{epoch} under {run_dir}. Available epoch dirs: {avail}")


def serve(adapter_dir: Path) -> None:
    """Serve the base model + hot-loaded `adapter_dir`, blocking until Ctrl-C.

    Mirrors the pipeline's non-expert epoch boundary (run_pipeline.main): start
    vLLM on the quantized base (config.VLLM_MODEL, launched with --enable-lora by
    vllm_server._serve_command), then hot-load the epoch adapter under a served
    name equal to its dir name (e.g. ``epoch_4``). Requests target that name.
    """
    from pipeline import vllm_server

    # Expert-LoRA adapters have no per-expert modules for vLLM to attach to, so
    # they can't be hot-loaded — bail with the same guidance the pipeline follows.
    if config.LORA_TARGET_EXPERTS:
        sys.exit(
            "serve_epoch hot-loads a LoRA adapter, which needs the non-expert "
            "LoRA path (LORA_TARGET_EXPERTS=0). Expert-LoRA epochs must be served "
            f"from their pre-merged epoch_<N>_merged checkpoint instead — e.g. "
            f"`vllm serve {adapter_dir.parent / (adapter_dir.name + '_merged')}`."
        )

    # The adapter and the base it is served on are configured independently (an
    # --epoch argument vs VLLM_MODEL); flag a 20B/120B mismatch here, where it's
    # legible, instead of letting vLLM report it as a tensor-shape error.
    utils.check_adapter_matches_base(adapter_dir, config.VLLM_MODEL, "VLLM_MODEL")

    adapter_name = adapter_dir.name
    # Tear the server down on every exit path — including the SIGTERM/SIGHUP that
    # would otherwise kill this process without running the finally below, orphaning
    # the vLLM session with all its VRAM still allocated and the port still bound.
    vllm_server.install_shutdown_handlers()
    # One try/finally around the whole lifecycle: if start() or load_adapter()
    # raises (or the engine later dies), stop() still runs and killpg's the vLLM
    # session — otherwise an abnormal exit orphans the engine / leaves a zombie API
    # front-end bound to the port, and the GPU reservation stays held until this
    # process is manually killed.
    try:
        logger.info("Starting vLLM base %s …", config.VLLM_MODEL)
        vllm_server.start(config.VLLM_MODEL)
        logger.info("Hot-loading adapter %s as '%s' …", adapter_dir, adapter_name)
        vllm_server.load_adapter(adapter_name, str(adapter_dir))
        logger.info(
            "vLLM ready at %s (served model name: %s). Press Ctrl-C to stop.",
            config.VLLM_BASE_URL, adapter_name,
        )
        # Block until Ctrl-C, but poll so a dead engine (crash during inference,
        # OOM, killed worker) ends the wait instead of pausing forever with the
        # GPUs reserved. signal.pause() wouldn't wake on the child's death.
        while vllm_server.is_alive():
            time.sleep(5)
        logger.error("vLLM server exited on its own; shutting down.")
    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down vLLM server.")
    finally:
        vllm_server.stop()


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--epoch", type=int, help="Epoch number to serve (epoch_<N> under the run dir).")
    src.add_argument("--adapter", help="Explicit epoch adapter dir to serve.")
    ap.add_argument("--checkpoint-dir", default=str(_CHECKPOINT_BASE),
                    help="Base dir holding the per-run checkpoint dirs (default: $CHECKPOINT_DIR or ./checkpoints).")
    ap.add_argument("--run-id", default=None,
                    help="RUN_ID subdir under --checkpoint-dir (default: most recent run).")
    ap.add_argument("--no-serve", action="store_true", help="Resolve the adapter dir only; don't start vLLM.")
    args = ap.parse_args()
    _reserve_gpus()

    if args.adapter:
        ckpt_dir = _resolve_checkpoint(Path("."), None, args.adapter)
    else:
        run_dir = _resolve_run_dir(Path(args.checkpoint_dir), args.run_id)
        ckpt_dir = _resolve_checkpoint(run_dir, args.epoch, None)

    if args.no_serve:
        logger.info("Resolve complete (--no-serve). Adapter dir: %s", ckpt_dir)
        return
    serve(ckpt_dir)


if __name__ == "__main__":
    main()
