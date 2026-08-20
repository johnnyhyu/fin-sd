"""End-to-end opsd: build reference solutions, then train.

    python -m opsd.run                  # build references (resumable) then train
    python -m opsd.run --skip-generate  # train on existing opsd/data/reference.jsonl
    python -m opsd.run --skip-train     # only (re)build the reference dataset
    python -m opsd.run --limit 20       # smoke test on the first 20 problems
"""
from __future__ import annotations

import argparse

from pipeline.utils import setup_logging

from . import config
from .generate_data import generate
from .train import train


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full opsd pipeline (build references + train).")
    ap.add_argument("--skip-generate", action="store_true", help="Reuse existing reference data.")
    ap.add_argument("--skip-train", action="store_true", help="Only build the reference dataset.")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N problems.")
    # Consumed by pipeline.config at import time (see its bootstrap); declared
    # here only so argparse does not reject it.
    ap.add_argument("--config", default=None,
                    help="Preset from configs/ to layer over .env.")
    args = ap.parse_args()

    setup_logging()
    if not args.skip_generate:
        generate(limit=args.limit)
    if not args.skip_train:
        train()
    print(f"Done. References: {config.REFERENCE_PATH} | Checkpoints: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
