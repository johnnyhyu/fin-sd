"""A run that produced nothing usable must not look like a run that went fine.

`runs/headline/` in this repo is the counterexample these tests lock down: 150 traces
written, `"n_errors": 0` in the manifest, and 0.0% accuracy because `web_search` was
unconfigured and 148 traces ended with no answer.
"""

from __future__ import annotations

import json

import pytest

from big_finance_harness.trace import health_warnings, summarize_traces


def _trace(qid: str, *, stop_reason: str, final_answer: str | None, results: list[tuple[bool, str]]):
    return {
        "question_id": qid,
        "trial_idx": 0,
        "question": "q",
        "reference_answer": "a",
        "model": "openai/gpt-oss-20b",
        "harness_version": "1.0.0",
        "thinking": "off",
        "max_steps": 50,
        "steps": [
            {
                "step": 0,
                "assistant_text": "",
                "tool_calls": [{"type": "tool_use", "id": "t1", "name": "web_search", "input": {}}],
                "tool_results": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": c, "is_error": e}
                    for e, c in results
                ],
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "wallclock_seconds": 0.1,
            }
        ],
        "final_answer": final_answer,
        "stop_reason": stop_reason,
        "total_prompt_tokens": 10,
        "total_completion_tokens": 5,
        "total_wallclock_seconds": 0.1,
        "started_at": "2026-07-31T00:00:00+00:00",
        "completed_at": "2026-07-31T00:00:01+00:00",
    }


def _write(tmp_path, traces):
    path = tmp_path / "m.traces.jsonl"
    path.write_text("\n".join(json.dumps(t) for t in traces) + "\n", encoding="utf-8")
    return path


def test_summary_of_a_healthy_run(tmp_path):
    path = _write(
        tmp_path,
        [
            _trace(f"q{i}", stop_reason="final_answer", final_answer="$1.0m", results=[(False, "ok")])
            for i in range(10)
        ],
    )
    summary = summarize_traces(path)
    assert summary["n_traces"] == 10
    assert summary["answer_rate"] == 1.0
    assert summary["tool_error_rate"] == 0.0
    assert summary["stop_reasons"] == {"final_answer": 10}
    assert health_warnings(summary) == []


def test_summary_flags_the_headline_failure_shape(tmp_path):
    """Traces written, no answers, every tool call erroring — the exact shape that
    previously reported `0 errors` and shipped straight to the grade phase."""
    err = "unexpected tool error: ValueError: web_search requires either SERP_API_KEY"
    path = _write(
        tmp_path,
        [
            _trace(f"q{i}", stop_reason="no_tool_call", final_answer=None, results=[(True, err)])
            for i in range(10)
        ],
    )
    summary = summarize_traces(path)
    assert summary["answer_rate"] == 0.0
    assert summary["tool_error_rate"] == 1.0
    assert summary["top_tool_errors"][0]["count"] == 10

    warnings = health_warnings(summary)
    assert any("final answer" in w for w in warnings)
    assert any("tool calls errored" in w for w in warnings)


def test_summary_flags_api_errors_and_unknown_tools(tmp_path):
    path = _write(
        tmp_path,
        [_trace("q0", stop_reason="error", final_answer=None, results=[(True, "unknown tool: 'search'")])]
        + [
            _trace(f"q{i}", stop_reason="final_answer", final_answer="x", results=[(False, "ok")])
            for i in range(1, 10)
        ],
    )
    summary = summarize_traces(path)
    assert summary["n_unknown_tool_calls"] == 1
    warnings = health_warnings(summary)
    assert any("stop_reason='error'" in w for w in warnings)
    assert any("does not exist" in w for w in warnings)


def test_summary_of_a_missing_file_is_empty(tmp_path):
    summary = summarize_traces(tmp_path / "nope.jsonl")
    assert summary["n_traces"] == 0
    assert health_warnings(summary) == []


@pytest.mark.parametrize("junk", ["", "   ", "{not json"])
def test_summary_skips_unparseable_lines(tmp_path, junk):
    path = tmp_path / "m.traces.jsonl"
    good = _trace("q0", stop_reason="final_answer", final_answer="x", results=[(False, "ok")])
    path.write_text(f"{junk}\n{json.dumps(good)}\n", encoding="utf-8")
    assert summarize_traces(path)["n_traces"] == 1
