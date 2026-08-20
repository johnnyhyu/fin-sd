"""Tests for the rubric-based grader.

We mock `litellm.acompletion` rather than calling a real judge — the grader's job is to
turn a structured judge response into a `GradedRun`, and that translation needs to be
verified independent of any provider.
"""

from __future__ import annotations

import json

import pytest

from big_finance_harness import grader as grader_module
from big_finance_harness.grader import grade
from big_finance_harness.types import (
    DatasetItem,
    RubricLine,
    RunRecord,
    StepRecord,
    ToolResultBlock,
    ToolUseBlock,
)


def _make_run(question_id: str, final_answer: str | None = "$114.3 billion") -> RunRecord:
    return RunRecord(
        question_id=question_id,
        question="What was Apple's FY2023 operating income?",
        reference_answer="$114.3 billion",
        model="anthropic:claude-opus-4-7",
        harness_version="0.1.0",
        thinking="off",
        temperature=None,
        max_steps=30,
        steps=[
            StepRecord(
                step=0,
                assistant_text="Looking up Apple's FY2023 10-K.",
                tool_calls=[ToolUseBlock(id="t1", name="edgar_search", input={"ticker": "AAPL"})],
                tool_results=[ToolResultBlock(tool_use_id="t1", content="...filings...")],
                prompt_tokens=100,
                completion_tokens=20,
                wallclock_seconds=1.0,
            )
        ],
        final_answer=final_answer,
        stop_reason="final_answer",
        total_prompt_tokens=100,
        total_completion_tokens=20,
        total_wallclock_seconds=1.0,
        started_at="2026-04-30T00:00:00+00:00",
        completed_at="2026-04-30T00:00:01+00:00",
    )


def _make_item() -> DatasetItem:
    return DatasetItem(
        id="bf-test-001",
        query="What was Apple's FY2023 operating income?",
        reference_answer="$114.3 billion",
        rubric=[
            RubricLine(text="Identifies AAPL as ticker", points=1),
            RubricLine(text="Locates FY2023 10-K", points=2),
            RubricLine(text="Reports operating income of $114.3 billion", points=5),
        ],
    )


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str, finish_reason: str = "stop"):
        self.message = _FakeMessage(content)
        self.finish_reason = finish_reason


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeResponse:
    def __init__(
        self,
        content: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        finish_reason: str = "stop",
    ):
        self.choices = [_FakeChoice(content, finish_reason)]
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)
        self._hidden_params = {"response_cost": cost}


@pytest.mark.asyncio
async def test_grader_translates_judge_response_to_graded_run(monkeypatch):
    """Judge returns 2 of 3 rubric lines satisfied → grader awards 1+5=6 of 8 points."""

    judge_payload = {
        "final_answer_correct": True,
        "rubric": [
            {"index": 1, "satisfied": True, "explanation": "Trace mentions AAPL"},
            {"index": 2, "satisfied": False, "explanation": "Did not locate the 10-K"},
            {"index": 3, "satisfied": True, "explanation": "Reported $114.3B"},
        ],
    }
    fake_response = _FakeResponse(
        content=json.dumps(judge_payload),
        prompt_tokens=2500,
        completion_tokens=180,
        cost=0.012,
    )

    async def fake_acompletion(**_kwargs):
        return fake_response

    monkeypatch.setattr(grader_module.litellm, "acompletion", fake_acompletion)

    run = _make_run("bf-test-001")
    item = _make_item()
    graded = await grade(run=run, item=item, judge_model_id="vertex:gemini-3.1-pro-preview")

    assert graded.final_answer_correct is True
    assert graded.rubric_lines_earned == 2
    assert graded.rubric_lines_possible == 3
    # Points: rubric 1 (+1) and rubric 3 (+5) earned; rubric 2 (+2) not. Total possible 8.
    assert graded.rubric_points_earned == 6
    assert graded.rubric_points_possible == 8
    # Judge accounting carried through.
    assert graded.judge_prompt_tokens == 2500
    assert graded.judge_completion_tokens == 180
    assert graded.judge_cost_usd == pytest.approx(0.012)
    # Per-line explanations preserved.
    assert graded.rubric_lines[0].earned is True
    assert graded.rubric_lines[1].earned is False
    assert graded.rubric_lines[2].judge_explanation == "Reported $114.3B"


@pytest.mark.asyncio
async def test_grader_handles_missing_rubric_index(monkeypatch):
    """If the judge omits an index, the grader marks that line as unsatisfied."""

    judge_payload = {
        "final_answer_correct": False,
        "rubric": [
            {"index": 1, "satisfied": True, "explanation": "ok"},
            # index 2 missing intentionally
            {"index": 3, "satisfied": True, "explanation": "ok"},
        ],
    }
    fake_response = _FakeResponse(
        content=json.dumps(judge_payload),
        prompt_tokens=1000,
        completion_tokens=50,
        cost=0.005,
    )

    async def fake_acompletion(**_kwargs):
        return fake_response

    monkeypatch.setattr(grader_module.litellm, "acompletion", fake_acompletion)

    graded = await grade(
        run=_make_run("bf-test-001"),
        item=_make_item(),
        judge_model_id="vertex:gemini-3.1-pro-preview",
    )
    assert graded.final_answer_correct is False
    # Lines 1 and 3 earned (1 + 5 = 6 points), line 2 missing → unsatisfied.
    assert graded.rubric_lines_earned == 2
    assert graded.rubric_points_earned == 6
    assert graded.rubric_lines[1].earned is False
    assert graded.rubric_lines[1].judge_explanation == "missing"


def _respond(monkeypatch, content: str, finish_reason: str = "stop"):
    async def fake_acompletion(**_kwargs):
        return _FakeResponse(
            content=content,
            prompt_tokens=100,
            completion_tokens=10,
            cost=0.001,
            finish_reason=finish_reason,
        )

    monkeypatch.setattr(grader_module.litellm, "acompletion", fake_acompletion)


@pytest.mark.asyncio
async def test_grader_tolerates_a_markdown_fenced_response(monkeypatch):
    """Open-weight judges and some OpenRouter upstreams ignore `response_format` and
    fence the object. `json.loads` on the raw string used to raise, and the orchestrator
    logged it as a grade error."""
    payload = {"final_answer_correct": True, "rubric": [{"index": 1, "satisfied": True}]}
    _respond(monkeypatch, f"Here you go:\n```json\n{json.dumps(payload)}\n```")

    graded = await grade(
        run=_make_run("bf-test-001"),
        item=_make_item(),
        judge_model_id="openrouter:google/gemini-3.1-pro-preview",
    )
    assert graded.final_answer_correct is True
    assert graded.rubric_lines[0].earned is True


@pytest.mark.asyncio
async def test_grader_raises_rather_than_silently_scoring_zero_on_empty_content(monkeypatch):
    """A reasoning judge can burn max_tokens before emitting the object. Defaulting to
    `{}` marked every rubric line unsatisfied and looked like a model failure."""
    _respond(monkeypatch, "")

    with pytest.raises(grader_module.JudgeResponseError, match="empty content"):
        await grade(
            run=_make_run("bf-test-001"),
            item=_make_item(),
            judge_model_id="openrouter:google/gemini-3.1-pro-preview",
        )


@pytest.mark.asyncio
async def test_grader_raises_on_non_json_content(monkeypatch):
    _respond(monkeypatch, "I cannot grade this trace.")

    with pytest.raises(grader_module.JudgeResponseError, match="not a JSON object"):
        await grade(
            run=_make_run("bf-test-001"),
            item=_make_item(),
            judge_model_id="openrouter:google/gemini-3.1-pro-preview",
        )


@pytest.mark.asyncio
async def test_grader_assigns_index_free_entries_positionally(monkeypatch):
    """Judges sometimes drop the `index` field entirely. Keying on `entry["index"]`
    raised KeyError and failed the whole grade."""
    payload = {
        "final_answer_correct": False,
        "rubric": [
            {"satisfied": True, "explanation": "a"},
            {"satisfied": False, "explanation": "b"},
            {"satisfied": True, "explanation": "c"},
        ],
    }
    _respond(monkeypatch, json.dumps(payload))

    graded = await grade(
        run=_make_run("bf-test-001"),
        item=_make_item(),
        judge_model_id="openrouter:google/gemini-3.1-pro-preview",
    )
    assert [line.earned for line in graded.rubric_lines] == [True, False, True]
    assert graded.rubric_points_earned == 6


@pytest.mark.asyncio
async def test_grader_survives_a_duplicated_index(monkeypatch):
    payload = {
        "final_answer_correct": False,
        "rubric": [
            {"index": 1, "satisfied": True, "explanation": "a"},
            {"index": 1, "satisfied": False, "explanation": "dup"},
        ],
    }
    _respond(monkeypatch, json.dumps(payload))

    graded = await grade(
        run=_make_run("bf-test-001"),
        item=_make_item(),
        judge_model_id="openrouter:google/gemini-3.1-pro-preview",
    )
    # First entry keeps index 1; the duplicate falls into the next free slot.
    assert graded.rubric_lines[0].earned is True
    assert graded.rubric_lines[1].earned is False


def test_judge_trace_includes_reasoning():
    """Reasoning models leave `assistant_text` empty; without reasoning the judge sees
    tool calls with no analysis behind them and marks 'Identifies X' unsatisfied."""
    run = _make_run("bf-test-001")
    run.steps[0].assistant_text = ""
    run.steps[0].assistant_reasoning = "AAPL is the ticker; pulling the FY2023 10-K."

    trace = grader_module._format_trace(run.steps)
    assert "AAPL is the ticker" in trace


def test_judge_trace_reasoning_can_be_disabled(monkeypatch):
    monkeypatch.setenv("BFH_JUDGE_INCLUDE_REASONING", "0")
    run = _make_run("bf-test-001")
    run.steps[0].assistant_reasoning = "secret analysis"

    assert "secret analysis" not in grader_module._format_trace(run.steps)


def test_judge_trace_caps_runaway_reasoning():
    run = _make_run("bf-test-001")
    run.steps[0].assistant_reasoning = "A" * 50_000 + "FINAL FIGURE 114301"

    trace = grader_module._format_trace(run.steps)
    assert len(trace) < 10_000
    # Head and tail both survive, so the conclusion is not the part that gets dropped.
    assert "FINAL FIGURE 114301" in trace
    assert "[reasoning truncated]" in trace


@pytest.mark.asyncio
async def test_grader_names_the_token_cap_when_the_judge_is_truncated(monkeypatch):
    """A judge cut off mid-JSON must blame the token cap, not the judge.

    `finish_reason='length'` is the only thing separating "this judge cannot follow the
    schema" from "this judge ran out of budget" — the partial object is well-formed up to
    where it stops. Reported as a parse failure it sends you off inspecting the prompt or
    swapping models, when the fix is to raise max_output_tokens.
    """
    truncated = '{"final_answer_correct": true, "rubric": [{"index": 1, "satisf'
    _respond(monkeypatch, truncated, finish_reason="length")

    with pytest.raises(grader_module.JudgeResponseError, match="output cap") as exc:
        await grade(
            run=_make_run("bf-test-001"),
            item=_make_item(),
            judge_model_id="openrouter:google/gemini-3.1-pro-preview",
            max_output_tokens=4096,
        )
    message = str(exc.value)
    # The cap has to be IN the message — it is the number the operator has to change.
    assert "4096" in message
    assert "max_output_tokens" in message
    # And it must not read as a schema failure.
    assert "not a JSON object" not in message


@pytest.mark.asyncio
async def test_grader_still_parses_a_complete_verdict_that_finished_on_length(monkeypatch):
    """Guard the check's placement: it fires on finish_reason, before parsing, so it
    cannot be mistaken for a generic 'judge returned something odd' fallback."""
    payload = {"final_answer_correct": True, "rubric": [{"index": 1, "satisfied": True}]}
    # Same complete payload, but finished normally — must grade, not raise.
    _respond(monkeypatch, json.dumps(payload), finish_reason="stop")

    graded = await grade(
        run=_make_run("bf-test-001"),
        item=_make_item(),
        judge_model_id="openrouter:google/gemini-3.1-pro-preview",
    )
    assert graded.final_answer_correct is True
