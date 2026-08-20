"""Reusable eval (inference) and grade (evaluation) phases.

These functions are the single implementation shared by the config-driven
`inference.py` / `evaluation.py` entry points and the multi-model
`scripts/run_eval_set.py` orchestrator. Each phase is resumable and writes
per-model JSONL under a run directory.

A model to run is described by a `ResolvedModel` (label + harness model id +
optional `api_base`/`api_key` for local vLLM routes).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import click

from big_finance_harness.agent import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_STEPS,
    run_question,
)
from big_finance_harness.config import ResolvedModel
from big_finance_harness.grader import grade
from big_finance_harness.models import make_client
from big_finance_harness.prompts import SYSTEM_PROMPT
from big_finance_harness.resumption import (
    eval_completed_pairs,
    eval_work_list,
    grade_completed_triples,
    grade_work_list,
)
from big_finance_harness.tools import default_tools
from big_finance_harness.trace import (
    TraceWriter,
    health_warnings,
    read_traces,
    summarize_traces,
)
from big_finance_harness.types import DatasetItem


def concurrency_for(model_id: str, base_concurrency: int) -> int:
    """Per-provider concurrency override.

    OpenRouter fans out to many upstream providers and has generous headroom, so we
    bump its floor to 8. The conservative `base_concurrency` applies to everything
    else, including local vLLM (a single self-hosted server saturates quickly).
    """
    if model_id.startswith("openrouter:"):
        return max(base_concurrency, 8)
    return base_concurrency


async def run_one_model(
    *,
    model: ResolvedModel,
    items: list[DatasetItem],
    out_dir: Path,
    concurrency: int,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    thinking: str = "off",
    temperature: float | None = None,
    token_budget: int | None = None,
    n_trials: int = 1,
    resume: bool = True,
) -> dict[str, Any]:
    """Run all (item, trial) pairs against one model. Resumable: skips pairs whose
    trace already exists in `<label>.traces.jsonl` when `resume=True`."""
    label = model.label
    traces_path = out_dir / f"{label}.traces.jsonl"

    if resume:
        completed, error_count = eval_completed_pairs(traces_path)
        if traces_path.exists():
            msg = f"[{label}] resuming with {len(completed)} traces already on disk"
            if error_count:
                msg += f" ({error_count} errored traces will be retried)"
            click.echo(msg)
    else:
        completed = set()
        if traces_path.exists():
            traces_path.unlink()

    client = make_client(
        model.model_id,
        api_base=model.api_base,
        api_key=model.api_key,
        extra_body=model.extra_body,
    )
    tools = default_tools()
    writer = TraceWriter(traces_path)

    work: list[tuple[DatasetItem, int]] = eval_work_list(
        items, n_trials, completed, id_of=lambda it: it.id
    )
    target_total = len(items) * n_trials
    effective_concurrency = concurrency_for(model.model_id, concurrency)
    if effective_concurrency != concurrency:
        click.echo(
            f"[{label}] effective concurrency: {effective_concurrency} "
            f"(base={concurrency}, bumped per-provider override)"
        )
    sem = asyncio.Semaphore(effective_concurrency)
    write_lock = asyncio.Lock()
    counters = {"done": len(completed), "errors": 0, "new": 0}
    started = time.monotonic()

    async def one(item: DatasetItem, trial_idx: int) -> None:
        async with sem:
            try:
                run = await run_question(
                    question_id=item.id,
                    question=item.query,
                    reference_answer=item.reference_answer,
                    client=client,
                    tools=tools,
                    system_prompt=SYSTEM_PROMPT,
                    thinking=thinking,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_steps=max_steps,
                    max_output_tokens=max_output_tokens,
                    token_budget=token_budget,
                    trial_idx=trial_idx,
                )
            except Exception as e:  # noqa: BLE001
                counters["errors"] += 1
                click.echo(f"[{label}] [error] {item.id}/t{trial_idx}: {e}", err=True)
                return
            async with write_lock:
                writer.write(run)
                counters["done"] += 1
                counters["new"] += 1
                if counters["new"] % 25 == 0:
                    click.echo(
                        f"[{label}] {counters['done']}/{target_total} done "
                        f"({counters['new']} new this session)",
                        err=True,
                    )

    await asyncio.gather(*[one(it, t) for it, t in work])
    elapsed = time.monotonic() - started
    click.echo(
        f"[{label}] complete: {counters['done']}/{target_total} traces "
        f"({counters['new']} new), {counters['errors']} errors, {elapsed:.0f}s"
    )

    # `counters` only sees this session and counts an errored trace as done, so it
    # cannot tell a good run from a broken one. Summarize what actually landed on disk
    # and say so out loud — a run where nothing answered used to be indistinguishable
    # from a clean one until someone opened the traces by hand.
    summary = summarize_traces(traces_path)
    click.echo(
        f"[{label}] stop reasons: {summary['stop_reasons']} | "
        f"answers: {summary['n_with_final_answer']}/{summary['n_traces']} "
        f"({summary['answer_rate']:.0%}) | tool errors: {summary['n_tool_errors']}/"
        f"{summary['n_tool_calls']} ({summary['tool_error_rate']:.0%})"
    )
    warnings = health_warnings(summary)
    for w in warnings:
        click.echo(f"[{label}] WARNING: {w}", err=True)

    return {
        "label": label,
        "model_id": model.model_id,
        "traces_path": str(traces_path.relative_to(out_dir)),
        "n_trials": n_trials,
        "concurrency": effective_concurrency,
        "max_output_tokens": max_output_tokens,
        "n_traces": counters["done"],
        "n_traces_new_this_session": counters["new"],
        "n_errors": counters["errors"],
        "elapsed_s": round(elapsed, 1),
        "trace_summary": summary,
        "health_warnings": warnings,
    }


async def grade_one_model(
    *,
    label: str,
    items_by_id: dict[str, DatasetItem],
    judges: list[str],
    out_dir: Path,
    concurrency: int,
    resume: bool = True,
    grades_suffix: str = "",
    judge_alias: str | None = None,
) -> dict[str, Any]:
    """Grade every trace for `label` with every judge in `judges`. Resumable: skips
    `(question_id, trial_idx, judge)` triples already in `<label>.grades.jsonl`.

    `grades_suffix` lets parallel orchestrator processes write to disjoint grade files
    (e.g. `{label}.grades.gemini.jsonl` and `{label}.grades.opus.jsonl`).
    """
    traces_path = out_dir / f"{label}.traces.jsonl"
    grades_filename = f"{label}.grades{grades_suffix}.jsonl"
    grades_path = out_dir / grades_filename

    if resume:
        completed = grade_completed_triples(grades_path)
        if grades_path.exists():
            click.echo(
                f"[{label}/grade] resuming with {len(completed)} grades already on disk"
            )
    else:
        completed = set()
        if grades_path.exists():
            grades_path.unlink()

    runs = list(read_traces(traces_path))
    # A trace whose question_id isn't in the dataset can never be graded. That happens
    # when the grade phase is pointed at a different dataset than the eval phase used
    # (e.g. `big_finance_subset` after a `big_finance_full` run) — previously those runs
    # were dropped one by one inside the worker with no output at all, and the phase
    # reported a clean finish having graded nothing.
    orphans = sorted({r.question_id for r in runs if r.question_id not in items_by_id})
    if orphans:
        click.echo(
            f"[{label}/grade] WARNING: {len(orphans)} question ids in {traces_path.name} "
            f"are not in the dataset and cannot be graded (e.g. {orphans[:3]}). "
            "Is the evaluation `dataset` the same one inference ran on?",
            err=True,
        )
        runs = [r for r in runs if r.question_id in items_by_id]

    work: list[tuple[Any, str]] = grade_work_list(
        runs, judges, completed, judge_alias=judge_alias
    )
    target_total = len(runs) * len(judges)

    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    counters = {"done": len(completed), "errors": 0, "new": 0}
    started = time.monotonic()

    async def one(run, judge_model_id: str) -> None:
        item = items_by_id[run.question_id]
        async with sem:
            try:
                graded = await grade(
                    run=run,
                    item=item,
                    judge_model_id=judge_model_id,
                    judge_alias=judge_alias,
                )
            except Exception as e:  # noqa: BLE001
                counters["errors"] += 1
                click.echo(
                    f"[{label}/grade/{judge_model_id}] [error] "
                    f"{run.question_id}/t{run.trial_idx}: {e}",
                    err=True,
                )
                return
            async with write_lock:
                with grades_path.open("a", encoding="utf-8") as f:
                    f.write(graded.model_dump_json() + "\n")
                counters["done"] += 1
                counters["new"] += 1

    await asyncio.gather(*[one(r, j) for r, j in work])
    elapsed = time.monotonic() - started
    click.echo(
        f"[{label}/grade] complete: {counters['done']}/{target_total} graded "
        f"({counters['new']} new), {counters['errors']} errors, {elapsed:.0f}s"
    )
    return {
        "label": label,
        "grades_path": str(grades_path.relative_to(out_dir)),
        "judges": judges,
        "n_graded": counters["done"],
        "n_graded_new_this_session": counters["new"],
        "n_errors": counters["errors"],
        "elapsed_s": round(elapsed, 1),
    }


async def run_eval_phase(
    models: list[ResolvedModel],
    items: list[DatasetItem],
    out_dir: Path,
    concurrency: int,
    max_steps: int,
    max_output_tokens: int,
    thinking: str,
    token_budget: int | None,
    n_trials: int,
    resume: bool,
    temperature: float | None = None,
) -> list[dict[str, Any]]:
    return await asyncio.gather(
        *[
            run_one_model(
                model=model,
                items=items,
                out_dir=out_dir,
                concurrency=concurrency,
                max_steps=max_steps,
                max_output_tokens=max_output_tokens,
                thinking=thinking,
                temperature=temperature,
                token_budget=token_budget,
                n_trials=n_trials,
                resume=resume,
            )
            for model in models
        ]
    )


async def run_grade_phase(
    labels: list[str],
    items_by_id: dict[str, DatasetItem],
    judges: list[str],
    out_dir: Path,
    concurrency: int,
    resume: bool,
    grades_suffix: str = "",
    judge_alias: str | None = None,
) -> list[dict[str, Any]]:
    return await asyncio.gather(
        *[
            grade_one_model(
                label=label,
                items_by_id=items_by_id,
                judges=judges,
                out_dir=out_dir,
                concurrency=concurrency,
                resume=resume,
                grades_suffix=grades_suffix,
                judge_alias=judge_alias,
            )
            for label in labels
        ]
    )
