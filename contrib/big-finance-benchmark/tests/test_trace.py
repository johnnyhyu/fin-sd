"""Tests for JSONL trace I/O.

The bug these lock down is a silent one: a trace that vanishes from the counts with no
error raised anywhere. See `trace.jsonl_lines`.
"""

from __future__ import annotations

import json

import pytest

from big_finance_harness.trace import TraceWriter, jsonl_lines, summarize_traces
from big_finance_harness.types import RunRecord, StepRecord, ToolResultBlock, ToolUseBlock

# Characters `str.splitlines()` treats as line boundaries but `split("\n")` does not.
# NEL and LINE/PARAGRAPH SEPARATOR are the ones that matter in practice: pydantic's
# `model_dump_json` writes them RAW (unlike `json.dumps`, which escapes non-ASCII by
# default), so they reach the file verbatim from scraped page text.
_SURVIVES_JSON_RAW = ["", " ", " "]
_SPLITS_SPLITLINES = _SURVIVES_JSON_RAW + ["\v", "\f"]


def _run_with(text: str) -> RunRecord:
    return RunRecord(
        question_id="bf-1",
        question="q",
        reference_answer="a",
        model="local:epoch_3",
        harness_version="0.1.0",
        thinking="off",
        temperature=None,
        max_steps=30,
        steps=[
            StepRecord(
                step=0,
                assistant_text="checking",
                tool_calls=[ToolUseBlock(id="t1", name="fetch_url", input={"url": "u"})],
                # The scraped page text lands here.
                tool_results=[ToolResultBlock(tool_use_id="t1", content=text)],
                prompt_tokens=1,
                completion_tokens=1,
                wallclock_seconds=0.1,
            )
        ],
        final_answer="42",
        stop_reason="final_answer",
        total_prompt_tokens=1,
        total_completion_tokens=1,
        total_wallclock_seconds=0.1,
        started_at="2026-08-10T00:00:00+00:00",
        completed_at="2026-08-10T00:00:01+00:00",
    )


@pytest.mark.parametrize("char", _SPLITS_SPLITLINES)
def test_splitlines_would_split_these_but_jsonl_lines_does_not(char, tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(f'{{"a": "x{char}y"}}\n', encoding="utf-8")
    # The premise: this is exactly the set of characters that made the old code wrong.
    assert len(p.read_text(encoding="utf-8").splitlines()) > 1
    assert [line for line in jsonl_lines(p) if line.strip()] == [f'{{"a": "x{char}y"}}']


@pytest.mark.parametrize("char", _SURVIVES_JSON_RAW)
def test_a_trace_containing_a_unicode_line_break_is_still_counted(char, tmp_path):
    """The end-to-end failure: one such character in a tool result cut the record in
    two, and every reader skips unparseable lines — so BOTH halves were dropped and a
    50-trace run reported 49."""
    traces = tmp_path / "m.traces.jsonl"
    writer = TraceWriter(traces)
    writer.write(_run_with(f"Revenue was{char}$410.5 million"))
    writer.write(_run_with("an ordinary result"))

    # Precondition: pydantic really did write the character out raw.
    assert char in traces.read_text(encoding="utf-8")

    summary = summarize_traces(traces)
    assert summary["n_traces"] == 2
    assert summary["n_with_final_answer"] == 2
    assert summary["answer_rate"] == 1.0


def test_written_traces_round_trip_as_json(tmp_path):
    traces = tmp_path / "m.traces.jsonl"
    TraceWriter(traces).write(_run_with("plain"))
    records = [json.loads(line) for line in jsonl_lines(traces) if line.strip()]
    assert len(records) == 1
    assert records[0]["question_id"] == "bf-1"


def test_jsonl_lines_on_an_empty_file(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    assert [line for line in jsonl_lines(p) if line.strip()] == []
