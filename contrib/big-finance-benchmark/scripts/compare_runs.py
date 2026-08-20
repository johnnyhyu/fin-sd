#!/usr/bin/env python3
"""Pool several single-model run dirs into one comparable table.

`inference.py` writes one run dir per model (its manifest describes that model only), so
a three-model comparison lives in three sibling dirs. This gathers their traces and
grades into a single pooled dir — symlinks, so nothing is copied or duplicated — and
runs the standard analysis on it:

    python scripts/compare_runs.py --out runs/audit-pooled \\
        --dataset data/big_finance_subset.jsonl \\
        runs/audit-base runs/audit-0727e1 runs/audit-0728e1

Produces `<out>/analysis/{per_grade,per_question,headline_table,inter_judge_kappa,
cost_throughput}.csv` via `build_analysis_csv.py` + `headline_table.py`, and prints a
run-health + accuracy summary per model: the numbers to read together, since a low score
on a run whose answer rate collapsed is a scaffold failure, not a model result.
"""

from __future__ import annotations

import collections
import json
import subprocess
import sys
from pathlib import Path

import click

from big_finance_harness.trace import health_warnings, summarize_traces

_SCRIPTS = Path(__file__).resolve().parent


@click.command()
@click.argument("run_dirs", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out", required=True, type=click.Path(path_type=Path), help="Pooled run dir to create.")
@click.option("--dataset", required=True, type=click.Path(exists=True, path_type=Path))
def main(run_dirs: tuple[Path, ...], out: Path, dataset: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pooled: list[Path] = []
    for d in run_dirs:
        for src in sorted(d.glob("*.traces.jsonl")) + sorted(d.glob("*.grades*.jsonl")):
            link = out / src.name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(src.resolve())
            pooled.append(link)
        manifest = d / "manifest.json"
        if manifest.exists():
            link = out / f"manifest.{d.name}.json"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(manifest.resolve())
    click.echo(f"pooled {len(pooled)} files into {out}")

    analysis = out / "analysis"
    subprocess.run(
        [sys.executable, str(_SCRIPTS / "build_analysis_csv.py"), "--run-dir", str(out),
         "--dataset", str(dataset), "--out-dir", str(analysis)],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(_SCRIPTS / "headline_table.py"),
         "--per-grade-csv", str(analysis / "per_grade.csv"), "--out-dir", str(analysis)],
        check=True,
    )

    click.echo("\n=== run health (from traces) ===")
    for path in sorted(out.glob("*.traces.jsonl")):
        label = path.name.removesuffix(".traces.jsonl")
        s = summarize_traces(path)
        click.echo(
            f"{label:<20} n={s['n_traces']:<4} answers={s['n_with_final_answer']}/{s['n_traces']} "
            f"({s['answer_rate']:.0%})  mean_steps={s['mean_steps']:<6} "
            f"tool_errors={s['tool_error_rate']:.0%}  stops={s['stop_reasons']}"
        )
        for w in health_warnings(s):
            click.echo(f"{'':<20} ! {w}")

    click.echo("\n=== accuracy (per model × judge) ===")
    rows: dict[tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)
    for path in sorted(out.glob("*.grades*.jsonl")):
        for line in path.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            g = json.loads(line)
            c = rows[(path.name.split(".grades")[0], g["judge"].split("/")[-1])]
            c["n"] += 1
            c["fa"] += int(bool(g.get("final_answer_correct")))
            c["pe"] += g["rubric_points_earned"]
            c["pp"] += g["rubric_points_possible"]
            c["le"] += g["rubric_lines_earned"]
            c["lp"] += g["rubric_lines_possible"]
    for (label, judge), c in sorted(rows.items()):
        click.echo(
            f"{label:<20} {judge:<26} n={c['n']:<4} "
            f"final_answer={c['fa']}/{c['n']} ({100 * c['fa'] / max(c['n'], 1):.1f}%)  "
            f"rubric_points={100 * c['pe'] / max(c['pp'], 1):.1f}%  "
            f"rubric_lines={100 * c['le'] / max(c['lp'], 1):.1f}%"
        )
    click.echo(f"\nCSVs: {analysis}")


if __name__ == "__main__":
    main()
