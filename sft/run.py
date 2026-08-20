"""End-to-end SFT distillation: generate teacher data, then train.

    python -m sft.run                  # generate (resumable) then train
    python -m sft.run --skip-generate  # train on existing sft/data/distill.jsonl
    python -m sft.run --skip-train     # only (re)build the distillation dataset
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import config as pc
from pipeline.utils import setup_logging

from . import config
from .generate_data import generate
from .gpus import configure_training_gpus
from .train import train


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full SFT distillation pipeline.")
    ap.add_argument("--skip-generate", action="store_true", help="Reuse existing distillation data.")
    ap.add_argument("--skip-train", action="store_true", help="Only build the distillation data.")
    ap.add_argument("--limit", type=int, default=None, help="Distil only the first N problems.")
    # Consumed by pipeline.config at import time (see its bootstrap); declared
    # here only so argparse does not reject it.
    ap.add_argument("--config", default=None,
                    help="Preset from configs/ to layer over .env.")
    args = ap.parse_args()

    # USE_MODAL=1 dispatches the whole flow to a Modal GPU container instead of
    # running on local GPUs (mirrors the root launch.py). The stage flags ride
    # along; all other config travels via the forwarded .env (see sft/modal_app.py).
    if pc.USE_MODAL:
        from ._modal_dispatch import dispatch
        raise SystemExit(dispatch(
            skip_generate=args.skip_generate, skip_train=args.skip_train, limit=args.limit
        ))

    setup_logging()
    if not args.skip_generate:
        generate(limit=args.limit)
    if not args.skip_train:
        # SFT_SERVE_AFTER_TRAIN is honoured here exactly as in `python -m sft.train`
        # (which reads it alongside its --serve flag); this entrypoint has no flag of
        # its own, so the env var is the only way in.
        serve = config.SERVE_AFTER_TRAIN
        # Reserve the training (and serving) GPUs here rather than up front: the
        # generate stage above is OpenRouter-only, so reserving before it would hold
        # idle cards for the whole distillation. train() below is the first thing to
        # touch CUDA, so this still lands before initialisation. Only --serve pulls
        # in a vLLM card — held-out validation decodes from the live weights.
        configure_training_gpus(needs_vllm=serve)
        train(serve=serve)
    run_dir = Path(config.OUTPUT_DIR) / config.RUN_ID
    print(f"Done. Data: {config.DISTILL_PATH} | Checkpoints: {run_dir}")


if __name__ == "__main__":
    main()
