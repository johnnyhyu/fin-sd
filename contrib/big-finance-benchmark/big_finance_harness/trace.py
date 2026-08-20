"""JSONL trace I/O plus the post-run health summary.

`summarize_traces` exists because a Big Finance run can fail completely while every
counter the orchestrator prints says it succeeded. See `preflight.py` for the incident:
150/150 traces written, 0 errors reported, 0.0% accuracy. The distinguishing signal was
never in the counts of traces — it was in the stop-reason mix and the tool-error rate,
neither of which anything looked at.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from big_finance_harness.types import RunRecord

# Thresholds for `health_warnings`. Deliberately loose — these are meant to catch a
# broken run, not to editorialize about a model that merely did badly.
_MIN_ANSWER_RATE = 0.5
_MAX_TOOL_ERROR_RATE = 0.25
_MAX_UNKNOWN_TOOL_RATE = 0.05


def jsonl_lines(path: str | Path) -> list[str]:
    """Read a JSONL file and split it into records on NEWLINES ONLY.

    Use this instead of `read_text().splitlines()` for anything JSONL. `splitlines`
    breaks on Unicode's full set of line boundaries, not just `\\n`: vertical tab,
    form feed, NEL (U+0085), and LINE/PARAGRAPH SEPARATOR (U+2028/U+2029) all split
    a string. Scraped page text routinely contains them, they reach the trace
    through a tool result, and pydantic's `model_dump_json` — unlike
    `json.dumps`, which escapes non-ASCII by default — writes U+0085/2028/2029 out
    RAW. (VT and FF are control characters and do get escaped; the other three are
    the ones seen in practice.)

    One such character therefore cuts a record into two fragments, and since every
    reader here skips lines that don't parse as JSON, BOTH halves are silently
    dropped. The trace is gone from the count with no error anywhere: that is why
    a 50-trace run reported 49, and why resumption re-bought grades that had
    already been paid for — the record that recorded them was unreadable.

    Returns raw strings including any blank trailing entry; callers skip empties as
    they already do.
    """
    return Path(path).read_text(encoding="utf-8").split("\n")


class TraceWriter:
    """Append-only JSONL writer for run records.

    One JSON object per line. Safe to append to from multiple async runs as long as each
    record is written in a single call (Python writes are atomic up to PIPE_BUF on most
    systems for small payloads; for large traces the caller should serialize writes via
    a lock).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: RunRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")


def read_traces(path: str | Path) -> Iterator[RunRecord]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield RunRecord.model_validate_json(line)


def summarize_traces(path: str | Path) -> dict[str, Any]:
    """Summarize a model's trace file into run-health fields.

    Reads raw JSON rather than validating into `RunRecord` — the `raw_response` blobs
    make full validation of a headline run needlessly slow, and every field used here is
    a top-level scalar.
    """
    p = Path(path)
    summary: dict[str, Any] = {
        "n_traces": 0,
        "stop_reasons": {},
        "n_with_final_answer": 0,
        "answer_rate": 0.0,
        "mean_steps": 0.0,
        "n_tool_calls": 0,
        "n_tool_errors": 0,
        "tool_error_rate": 0.0,
        "n_unknown_tool_calls": 0,
        "top_tool_errors": [],
    }
    if not p.exists():
        return summary

    stop_reasons: Counter[str] = Counter()
    tool_errors: Counter[str] = Counter()
    n_traces = n_answers = n_steps = n_calls = n_errors = n_unknown = 0

    for line in jsonl_lines(p):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        n_traces += 1
        stop_reasons[r.get("stop_reason") or "unknown"] += 1
        if (r.get("final_answer") or "").strip():
            n_answers += 1
        steps = r.get("steps") or []
        n_steps += len(steps)
        for s in steps:
            n_calls += len(s.get("tool_calls") or [])
            for tr in s.get("tool_results") or []:
                if not tr.get("is_error"):
                    continue
                n_errors += 1
                message = (tr.get("content") or "").strip()
                if message.startswith("unknown tool:"):
                    n_unknown += 1
                # Bucket by prefix: the tail carries per-call detail (a URL, a ticker)
                # that would otherwise scatter one bucket per occurrence.
                tool_errors[message[:80]] += 1

    summary["n_traces"] = n_traces
    summary["stop_reasons"] = dict(stop_reasons.most_common())
    summary["n_with_final_answer"] = n_answers
    summary["answer_rate"] = round(n_answers / n_traces, 4) if n_traces else 0.0
    summary["mean_steps"] = round(n_steps / n_traces, 2) if n_traces else 0.0
    summary["n_tool_calls"] = n_calls
    summary["n_tool_errors"] = n_errors
    summary["tool_error_rate"] = round(n_errors / n_calls, 4) if n_calls else 0.0
    summary["n_unknown_tool_calls"] = n_unknown
    summary["top_tool_errors"] = [
        {"count": c, "message": m} for m, c in tool_errors.most_common(5)
    ]
    return summary


def health_warnings(summary: dict[str, Any]) -> list[str]:
    """Flag trace summaries that indicate a broken run rather than a weak model."""
    warnings: list[str] = []
    if not summary.get("n_traces"):
        return warnings

    if summary["answer_rate"] < _MIN_ANSWER_RATE:
        warnings.append(
            f"only {summary['n_with_final_answer']}/{summary['n_traces']} traces "
            f"({summary['answer_rate']:.0%}) produced any final answer — grading these "
            "will report near-zero accuracy regardless of model quality"
        )
    if summary["tool_error_rate"] > _MAX_TOOL_ERROR_RATE:
        top = summary["top_tool_errors"][0] if summary["top_tool_errors"] else None
        detail = f"; most common: {top['message']!r} ×{top['count']}" if top else ""
        warnings.append(
            f"{summary['n_tool_errors']}/{summary['n_tool_calls']} tool calls errored "
            f"({summary['tool_error_rate']:.0%}){detail}"
        )
    n_api_errors = summary["stop_reasons"].get("error", 0)
    if n_api_errors:
        warnings.append(
            f"{n_api_errors} traces ended in an API error (stop_reason='error'); "
            "re-running with resume enabled will retry them"
        )
    if summary["n_tool_calls"]:
        unknown_rate = summary["n_unknown_tool_calls"] / summary["n_tool_calls"]
        if unknown_rate > _MAX_UNKNOWN_TOOL_RATE:
            warnings.append(
                f"{summary['n_unknown_tool_calls']}/{summary['n_tool_calls']} tool "
                f"calls named a tool that does not exist ({unknown_rate:.0%}) — check "
                "the server's tool-call parser if this is a local route"
            )
    return warnings


def read_dataset(path: str | Path) -> list[dict]:
    """Read a JSONL dataset file. Returns raw dicts; the caller is responsible for
    validating against `DatasetItem` if it wants pydantic types."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
