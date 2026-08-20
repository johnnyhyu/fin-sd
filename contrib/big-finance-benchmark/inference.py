"""Config-driven inference (eval) phase for the Big Finance harness.

Mirrors FinanceReasoning/MMLU-Pro:

    python inference.py --config config/config.yaml

Reads the `inference:` block and `llms:` catalog from the config, runs the model
under test over the dataset (agentic tool-use loop), and writes resumable traces
to `<output_dir>/<run_id>/<model_name>.traces.jsonl` plus a manifest. Grade the
traces afterward with `python evaluation.py --config config/config.yaml`.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from big_finance_harness import __version__
from big_finance_harness.config import Config
from big_finance_harness.models.base import LiteLLMClient
from big_finance_harness.orchestrate import run_eval_phase
from big_finance_harness.preflight import assert_ready
from big_finance_harness.prompts import SYSTEM_PROMPT
from big_finance_harness.tools import default_tools
from big_finance_harness.trace import jsonl_lines
from big_finance_harness.types import DatasetItem


def _load_dataset(path: Path) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    for line in jsonl_lines(path):
        if line.strip():
            items.append(DatasetItem.model_validate_json(line))
    return items


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature to send with every request; overrides the config. "
             "Omit to use the config value (default: send no temperature at all, which "
             "means a local vLLM route samples at ITS default of 1.0).",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    # CLI wins over the YAML, but only when actually passed: `is not None` so
    # `--temperature 0` overrides a non-zero config value instead of reading as unset.
    if args.temperature is not None:
        config.inference.temperature = args.temperature
    cfg = config.inference
    model = config.resolve_model(cfg.model_name)

    dataset_path = config.dataset_path(cfg)
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    # Fail before spending a single API call on a run that cannot produce valid results.
    assert_ready([model])

    full_items = _load_dataset(dataset_path)
    if cfg.sample_n is not None:
        if cfg.sample_n > len(full_items):
            raise SystemExit(
                f"inference.sample_n={cfg.sample_n} exceeds the dataset size "
                f"({len(full_items)} items in {dataset_path}). Lower sample_n, or set "
                "it to null to run the full dataset."
            )
        items = random.Random(cfg.sample_seed).sample(full_items, cfg.sample_n)
    else:
        items = full_items

    out_dir = Path(cfg.output_dir) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    started_mono = time.monotonic()

    manifest: dict[str, Any] = {
        "run_id": cfg.run_id,
        "kind": cfg.kind,
        "phase": "inference",
        "started_at": started_at,
        "completed_at": None,
        "harness_version": __version__,
        "dataset": {
            "path": str(dataset_path),
            "sha256": _sha256_file(dataset_path),
            "size": len(full_items),
            "sampled_n": len(items) if cfg.sample_n is not None else None,
            "sample_seed": cfg.sample_seed if cfg.sample_n is not None else None,
        },
        "config": {
            "thinking": cfg.thinking,
            "temperature": cfg.temperature,
            "max_steps": cfg.max_steps,
            "max_output_tokens": cfg.max_output_tokens,
            "token_budget": cfg.token_budget,
            "n_trials": cfg.n_trials,
            "resume": cfg.resume,
            "concurrency_per_model": cfg.concurrency,
            "num_retries": LiteLLMClient.NUM_RETRIES,
            "tools": [t.name for t in default_tools()],
            "system_prompt": SYSTEM_PROMPT,
        },
        "models": [{"label": model.label, "model_id": model.model_id}],
        "results": {"eval": None},
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote manifest to {manifest_path}")

    print(
        f"\n=== eval phase: {model.label} × {len(items)} items × {cfg.n_trials} trials ==="
    )
    eval_summaries = asyncio.run(
        run_eval_phase(
            [model],
            items,
            out_dir,
            cfg.concurrency,
            cfg.max_steps,
            cfg.max_output_tokens,
            cfg.thinking,
            cfg.token_budget,
            cfg.n_trials,
            cfg.resume,
            temperature=cfg.temperature,
        )
    )
    manifest["results"]["eval"] = eval_summaries
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["total_elapsed_s"] = round(time.monotonic() - started_mono, 1)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\ndone: {manifest_path} (total {manifest['total_elapsed_s']:.0f}s)")

    # Repeat any health warnings last so they're the final thing on screen — they are
    # the difference between "this model scored badly" and "this run is not valid".
    flagged = [w for s in eval_summaries for w in s.get("health_warnings", [])]
    if flagged:
        print("\nrun health warnings — inspect before grading:")
        for w in flagged:
            print(f"  ! {w}")


if __name__ == "__main__":
    main()
