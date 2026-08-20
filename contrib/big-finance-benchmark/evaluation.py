"""Config-driven evaluation (grade) phase for the Big Finance harness.

Mirrors FinanceReasoning/MMLU-Pro:

    python evaluation.py --config config/config.yaml

Reads the `evaluation:` block from the config, grades the traces written by
`inference.py` (`<output_dir>/<run_id>/<model_name>.traces.jsonl`) with each judge,
and writes resumable grades to `<model_name>.grades.jsonl`. Aggregate the grades
into headline tables with `scripts/headline_table.py`.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from big_finance_harness.config import Config
from big_finance_harness.orchestrate import run_grade_phase
from big_finance_harness.preflight import assert_ready
from big_finance_harness.trace import jsonl_lines
from big_finance_harness.types import DatasetItem


def _load_dataset(path: Path) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    for line in jsonl_lines(path):
        if line.strip():
            items.append(DatasetItem.model_validate_json(line))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML.")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    cfg = config.evaluation
    # Resolve so an unknown model_name fails loudly here rather than as a missing file.
    model = config.resolve_model(cfg.model_name)
    label = model.label

    # Judge credentials only — grading replays traces and never calls a tool, so the
    # tool keys the eval phase needs are irrelevant here.
    assert_ready([], list(cfg.judges), require_tool_keys=False)

    out_dir = Path(cfg.output_dir) / cfg.run_id
    traces_path = out_dir / f"{label}.traces.jsonl"
    if not traces_path.exists():
        raise FileNotFoundError(
            f"no traces at {traces_path} — run inference.py first for run_id "
            f"'{cfg.run_id}' and model '{cfg.model_name}'."
        )

    dataset_path = config.dataset_path(cfg)
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    items_by_id = {it.id: it for it in _load_dataset(dataset_path)}

    print(f"\n=== grade phase: {label} × judges = {cfg.judges} ===")
    grade_summaries = asyncio.run(
        run_grade_phase(
            [label],
            items_by_id,
            list(cfg.judges),
            out_dir,
            cfg.grade_concurrency,
            cfg.resume,
        )
    )
    for summary in grade_summaries:
        print(
            f"[{summary['label']}] graded {summary['n_graded']} "
            f"({summary['n_errors']} errors) -> {out_dir / summary['grades_path']}"
        )

    print(f"\ndone: grades under {out_dir}")


if __name__ == "__main__":
    main()
