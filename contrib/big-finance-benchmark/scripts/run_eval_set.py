"""Orchestrator for a Big Finance run set.

A run set is one combination of a dataset (or sample of one) plus a list of models. It
produces a directory with a manifest plus per-model trace and grade JSONLs:

    runs/<run_id>/
        manifest.json
        opus47.traces.jsonl
        opus47.grades.jsonl
        sonnet46.traces.jsonl
        sonnet46.grades.jsonl
        ...

The manifest records dataset hash, harness version, model list, and configuration so the
run is fully reproducible from `runs/<run_id>/manifest.json` alone.

Usage:

  # Dry run on a 30-question sample
  python scripts/run_eval_set.py \\
    --dataset data/big_finance_full.jsonl \\
    --sample-n 30 --sample-seed 0 \\
    --run-id dryrun-20260430 \\
    --kind dry_run

  # Headline run on the full dataset
  python scripts/run_eval_set.py \\
    --dataset data/big_finance_full.jsonl \\
    --run-id headline-20260430 \\
    --kind headline
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

from big_finance_harness import __version__
from big_finance_harness.agent import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_STEPS,
)
from big_finance_harness.config import ResolvedModel
from big_finance_harness.models.base import LiteLLMClient
from big_finance_harness.orchestrate import run_eval_phase, run_grade_phase
from big_finance_harness.preflight import assert_ready
from big_finance_harness.prompts import SYSTEM_PROMPT
from big_finance_harness.tools import default_tools
from big_finance_harness.trace import jsonl_lines
from big_finance_harness.types import DatasetItem


# Default model lineup — one entry per snapshot we'll evaluate. Label is what shows up in
# filenames and the manifest; model_id is the harness's `provider:snapshot` form.
#
# Every model routes through OpenRouter with a single `OPENROUTER_API_KEY` — no
# per-provider keys. The snapshot is the OpenRouter slug `<vendor>/<model>`. Edit this
# list to swap in / out specific snapshots.
DEFAULT_MODELS: list[tuple[str, str]] = [
    # Closed frontier
    ("opus47", "openrouter:anthropic/claude-opus-4.7"),
    ("sonnet46", "openrouter:anthropic/claude-sonnet-4.6"),
    ("gpt55", "openrouter:openai/gpt-5.5"),
    ("gpt54mini", "openrouter:openai/gpt-5.4-mini"),
    ("gem31pro", "openrouter:google/gemini-3.1-pro-preview"),
    ("gem3flash", "openrouter:google/gemini-3-flash-preview"),
    ("gem35flash", "openrouter:google/gemini-3.5-flash"),
    # Open frontier
    ("kimi-k26", "openrouter:moonshotai/kimi-k2.6"),
    ("deepseek-v4-pro", "openrouter:deepseek/deepseek-v4-pro"),
    ("glm-51", "openrouter:z-ai/glm-5.1"),
    ("gemma4-31b", "openrouter:google/gemma-4-31b-it"),
    ("qwen36-27b", "openrouter:qwen/qwen3.6-27b"),
    ("gpt-oss-120b", "openrouter:openai/gpt-oss-120b"),
]

# Judges. Two non-evaluated judges from different families so callers can report
# inter-judge Cohen's kappa (see grader.py). Gemini 3.1 Pro and Opus 4.7 both sit
# outside most of the model lineup, giving a clean "judge is not the system under test"
# story. Both route through OpenRouter.
DEFAULT_JUDGES = (
    "openrouter:google/gemini-3.1-pro-preview",
    "openrouter:anthropic/claude-opus-4.7",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_dataset(path: Path) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    for line in jsonl_lines(path):
        if line.strip():
            items.append(DatasetItem.model_validate_json(line))
    return items


def _sample_items(items: list[DatasetItem], n: int, seed: int) -> list[DatasetItem]:
    rng = random.Random(seed)
    return rng.sample(items, n)


@click.command()
@click.option("--dataset", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--run-id", required=True, type=str, help="Directory name under runs/.")
@click.option(
    "--kind",
    type=click.Choice(["dry_run", "pilot", "headline", "ablation"]),
    default="headline",
    show_default=True,
)
@click.option("--sample-n", type=int, default=None, help="If set, sample this many items.")
@click.option("--sample-seed", type=int, default=0, show_default=True)
@click.option(
    "--concurrency",
    type=int,
    default=3,
    show_default=True,
    help="Per-model parallelism. Stress test confirmed 3 is safe under Vertex PT load.",
)
@click.option(
    "--temperature",
    type=float,
    default=None,
    help="Sampling temperature sent with every request. Omitted by default, which "
    "means each provider applies its own — and a local vLLM route uses 1.0, so an "
    "A/B without this measures the sampler as much as the weights. Pass 0 for a "
    "reproducible comparison.",
)
@click.option("--max-steps", type=int, default=DEFAULT_MAX_STEPS, show_default=True)
@click.option("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, show_default=True)
@click.option(
    "--token-budget",
    type=int,
    default=None,
    show_default=True,
    help="Optional cumulative prompt+completion token cap per question. Off by "
    "default — max_steps + request_timeout already bound runaway behavior, and the "
    "1M cap empirically cut off legitimate methodical retrieval work rather than "
    "catching loops. Set explicitly for cost-bounded ablations.",
)
@click.option(
    "--n-trials",
    type=int,
    default=1,
    show_default=True,
    help="Run each (question, model) pair this many times. n=3 gives mean ± std for "
    "the paper; n=1 records single-shot accuracy.",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    show_default=True,
    help="Resume a previous run by skipping (question, trial) pairs whose traces "
    "already exist, and (question, trial, judge) triples whose grades exist. With "
    "--no-resume, existing traces and grades for this run_id are deleted.",
)
@click.option(
    "--thinking",
    type=click.Choice(["off", "low", "medium", "high"]),
    default="off",
    show_default=True,
    help="'off' means no explicit thinking config — vendors use their defaults.",
)
@click.option(
    "--judge",
    "judges",
    multiple=True,
    default=DEFAULT_JUDGES,
    show_default=True,
    help="Judge model id. Pass multiple times for inter-judge agreement: "
    "--judge openrouter:google/gemini-3.1-pro-preview "
    "--judge openrouter:anthropic/claude-opus-4.7",
)
@click.option(
    "--grade-concurrency",
    type=int,
    default=2,
    show_default=True,
    help="Per-model parallelism for grading. 2 keeps the judge under quota during a "
    "many-model parallel grade phase.",
)
@click.option("--skip-grade", is_flag=True, default=False, help="Run eval only, skip grading.")
@click.option(
    "--skip-model",
    "skip_models",
    multiple=True,
    default=(),
    help="Exclude these model labels from both eval and grade phases this session. "
    "Existing traces on disk are preserved; the model just doesn't get worked on this "
    "run. Re-run later without the flag to pick it back up.",
)
@click.option(
    "--grades-suffix",
    type=str,
    default="",
    help="Suffix inserted into the grades filename: `{label}.grades{suffix}.jsonl`. "
    "Use to write disjoint grade files per parallel orchestrator process (e.g. one "
    "process per judge to avoid cross-judge asyncio fairness issues).",
)
@click.option(
    "--judge-alias",
    type=str,
    default=None,
    help="Override the stored `GradedRun.judge` label. Lets a substituted same-family "
    "model record under a unified judge label so downstream analysis treats the "
    "grades as one bucket.",
)
def main(
    dataset: Path,
    run_id: str,
    kind: str,
    sample_n: int | None,
    sample_seed: int,
    concurrency: int,
    temperature: float | None,
    max_steps: int,
    max_output_tokens: int,
    token_budget: int | None,
    n_trials: int,
    resume: bool,
    thinking: str,
    judges: tuple[str, ...],
    grade_concurrency: int,
    skip_grade: bool,
    skip_models: tuple[str, ...],
    grades_suffix: str,
    judge_alias: str | None,
) -> None:
    """Run all default models on a dataset (or sample) and write a manifest+traces+grades."""
    out_dir = Path("runs") / run_id
    if out_dir.exists() and any(out_dir.iterdir()):
        click.echo(
            f"warning: {out_dir} already exists and is non-empty. Continuing will "
            "overwrite trace and grade files for this run_id.",
            err=True,
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    full_items = _load_dataset(dataset)
    dataset_sha = _sha256_file(dataset)
    if sample_n is not None:
        if sample_n > len(full_items):
            raise click.ClickException(
                f"--sample-n {sample_n} exceeds the dataset size ({len(full_items)} "
                f"items in {dataset}); drop the flag to run everything."
            )
        items = _sample_items(full_items, sample_n, sample_seed)
    else:
        items = full_items

    started_at = datetime.now(timezone.utc).isoformat()
    started_mono = time.monotonic()

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "kind": kind,
        "started_at": started_at,
        "completed_at": None,
        "harness_version": __version__,
        "dataset": {
            "path": str(dataset),
            "sha256": dataset_sha,
            "size": len(full_items),
            "sampled_n": len(items) if sample_n is not None else None,
            "sample_seed": sample_seed if sample_n is not None else None,
        },
        "config": {
            "thinking": thinking,
            "temperature": temperature,
            "max_steps": max_steps,
            "max_output_tokens": max_output_tokens,
            "token_budget": token_budget,
            "n_trials": n_trials,
            "resume": resume,
            "concurrency_per_model": concurrency,
            "num_retries": LiteLLMClient.NUM_RETRIES,
            "tools": [t.name for t in default_tools()],
            "system_prompt": SYSTEM_PROMPT,
        },
        # Filled in below after we apply --skip-model.
        "models": None,
        "judges": list(judges) if not skip_grade else None,
        "results": {"eval": None, "grade": None},
    }

    skip_set = set(skip_models)
    active_models = [(label, mid) for label, mid in DEFAULT_MODELS if label not in skip_set]
    if skip_set:
        click.echo(f"skipping models this session: {sorted(skip_set)}")
    manifest["models"] = [{"label": label, "model_id": mid} for label, mid in active_models]
    if skip_set:
        manifest["skipped_models"] = sorted(skip_set)

    # All DEFAULT_MODELS are hosted `provider:snapshot` routes (no local api_base).
    resolved_models = [ResolvedModel(label=label, model_id=mid) for label, mid in active_models]

    # A headline run is 13 models × 928 questions × 3 trials. Check credentials before
    # any of that starts — a missing SERP_API_KEY does not fail the run, it silently
    # voids it.
    assert_ready(resolved_models, list(judges) if not skip_grade else None)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    click.echo(f"wrote manifest to {manifest_path}")

    # Eval phase: run all models in parallel, n_trials trials each.
    click.echo(
        f"\n=== eval phase: {len(active_models)} models × {len(items)} items × "
        f"{n_trials} trials ==="
    )
    eval_summaries = asyncio.run(
        run_eval_phase(
            resolved_models,
            items,
            out_dir,
            concurrency,
            max_steps,
            max_output_tokens,
            thinking,
            token_budget,
            n_trials,
            resume,
            temperature=temperature,
        )
    )
    manifest["results"]["eval"] = eval_summaries
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Grade phase: every trace × every judge.
    if not skip_grade:
        click.echo(f"\n=== grade phase: judges = {list(judges)} ===")
        items_by_id = {it.id: it for it in items}
        grade_summaries = asyncio.run(
            run_grade_phase(
                [label for label, _ in active_models],
                items_by_id,
                list(judges),
                out_dir,
                grade_concurrency,
                resume,
                grades_suffix,
                judge_alias,
            )
        )
        manifest["results"]["grade"] = grade_summaries

    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["total_elapsed_s"] = round(time.monotonic() - started_mono, 1)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    click.echo(f"\ndone: {out_dir}/manifest.json (total {manifest['total_elapsed_s']:.0f}s)")


if __name__ == "__main__":
    main()
