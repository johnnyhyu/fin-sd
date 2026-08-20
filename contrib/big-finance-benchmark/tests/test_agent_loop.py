"""End-to-end test of the agent loop with a stubbed model client and stubbed tools.

Verifies that the loop:
  - calls the model
  - dispatches tool calls
  - terminates on `final_answer`
  - records steps and token usage
"""

from __future__ import annotations

import pytest

from big_finance_harness.agent import MAX_EMPTY_TURNS, run_question
from big_finance_harness.models.base import ModelClient, ThinkingLevel
from big_finance_harness.tools.base import Tool
from big_finance_harness.tools.final_answer import FinalAnswerTool
from big_finance_harness.types import (
    Message,
    ModelResponse,
    ToolSpec,
    ToolUseBlock,
)


class _ScriptedClient(ModelClient):
    snapshot = "anthropic:claude-test-2026-01-01"

    def __init__(self, responses: list[ModelResponse]):
        self._responses = list(responses)
        self.calls = 0
        self.tool_choices: list[str] = []
        self.seen_messages: list[list[Message]] = []

    async def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        temperature: float = 0.0,
        thinking: ThinkingLevel = "off",
        max_output_tokens: int = 4096,
        tool_choice: str = "auto",
    ) -> ModelResponse:
        resp = self._responses[self.calls]
        self.calls += 1
        self.tool_choices.append(tool_choice)
        self.seen_messages.append(list(messages))
        return resp


class _CalcTool(Tool):
    name = "calc"
    description = "Adds two numbers."
    input_schema = {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    }

    async def run(self, args):
        return str(args["a"] + args["b"])


@pytest.mark.asyncio
async def test_agent_terminates_on_final_answer():
    client = _ScriptedClient(
        [
            ModelResponse(
                text="Let me calculate.",
                tool_calls=[ToolUseBlock(id="c1", name="calc", input={"a": 2, "b": 3})],
                stop_reason="tool_use",
                prompt_tokens=10,
                completion_tokens=5,
            ),
            ModelResponse(
                text="The answer is 5.",
                tool_calls=[ToolUseBlock(id="c2", name="final_answer", input={"answer": "5"})],
                stop_reason="tool_use",
                prompt_tokens=20,
                completion_tokens=8,
            ),
        ]
    )
    tools = [_CalcTool(), FinalAnswerTool()]
    record = await run_question(
        question_id="q1",
        question="What is 2 + 3?",
        reference_answer="5",
        client=client,
        tools=tools,
        system_prompt="test",
        max_steps=5,
    )
    assert record.stop_reason == "final_answer"
    assert record.final_answer == "5"
    assert len(record.steps) == 2
    assert record.total_prompt_tokens == 30
    assert record.total_completion_tokens == 13


@pytest.mark.asyncio
async def test_agent_handles_unknown_tool():
    client = _ScriptedClient(
        [
            ModelResponse(
                text="Trying a hallucinated tool.",
                tool_calls=[ToolUseBlock(id="c1", name="bogus_tool", input={})],
                stop_reason="tool_use",
                prompt_tokens=10,
                completion_tokens=5,
            ),
            ModelResponse(
                text="Falling back.",
                tool_calls=[
                    ToolUseBlock(id="c2", name="final_answer", input={"answer": "fallback"})
                ],
                stop_reason="tool_use",
                prompt_tokens=12,
                completion_tokens=4,
            ),
        ]
    )
    record = await run_question(
        question_id="q2",
        question="?",
        reference_answer=None,
        client=client,
        tools=[FinalAnswerTool()],
        system_prompt="test",
        max_steps=5,
    )
    assert record.stop_reason == "final_answer"
    # First step's tool_result for the bogus tool was an error.
    assert any(tr.is_error for tr in record.steps[0].tool_results)


@pytest.mark.asyncio
async def test_agent_terminates_on_no_tool_call():
    client = _ScriptedClient(
        [
            ModelResponse(
                text="The answer is 42.",
                tool_calls=[],
                stop_reason="end_turn",
                prompt_tokens=5,
                completion_tokens=4,
            )
        ]
    )
    record = await run_question(
        question_id="q3",
        question="?",
        reference_answer="42",
        client=client,
        tools=[FinalAnswerTool()],
        system_prompt="test",
        max_steps=5,
    )
    assert record.stop_reason == "no_tool_call"
    assert record.final_answer == "The answer is 42."


def _empty_turn() -> ModelResponse:
    """What gpt-oss hands back when it reasons and then closes the turn with nothing:
    no text, no tool call."""
    return ModelResponse(
        text="",
        tool_calls=[],
        stop_reason="end_turn",
        prompt_tokens=5,
        completion_tokens=4,
    )


@pytest.mark.asyncio
async def test_agent_nudges_through_an_empty_turn():
    client = _ScriptedClient(
        [
            _empty_turn(),
            ModelResponse(
                text="",
                tool_calls=[
                    ToolUseBlock(id="c1", name="final_answer", input={"answer": "42"})
                ],
                stop_reason="tool_use",
                prompt_tokens=6,
                completion_tokens=3,
            ),
        ]
    )
    record = await run_question(
        question_id="q5",
        question="?",
        reference_answer="42",
        client=client,
        tools=[FinalAnswerTool()],
        system_prompt="test",
        max_steps=5,
    )
    assert record.stop_reason == "final_answer"
    assert record.final_answer == "42"
    # The empty turn was nudged (a user message was appended) and the retry forced a
    # tool call rather than asking for one.
    assert client.tool_choices == ["auto", "required"]
    assert client.seen_messages[-1][-1].role == "user"


class _FlakyClient(_ScriptedClient):
    """Raises on the first call (as vLLM does when its harmony parser rejects the
    model's own output), then follows the script."""

    def __init__(self, exc: Exception, responses: list[ModelResponse]):
        super().__init__(responses)
        self._exc: Exception | None = exc

    async def chat(self, *args, **kwargs) -> ModelResponse:
        if self._exc is not None:
            exc, self._exc = self._exc, None
            self.tool_choices.append(kwargs.get("tool_choice", "auto"))
            raise exc
        return await super().chat(*args, **kwargs)


@pytest.mark.asyncio
async def test_agent_recovers_from_malformed_generation_error():
    client = _FlakyClient(
        RuntimeError(
            "InternalServerError: Hosted_vllmException - "
            "unexpected tokens remaining in message header: Some(\"...\")"
        ),
        [
            ModelResponse(
                text="",
                tool_calls=[
                    ToolUseBlock(id="c1", name="final_answer", input={"answer": "42"})
                ],
                stop_reason="tool_use",
                prompt_tokens=6,
                completion_tokens=3,
            )
        ],
    )
    record = await run_question(
        question_id="q7",
        question="?",
        reference_answer="42",
        client=client,
        tools=[FinalAnswerTool()],
        system_prompt="test",
        max_steps=5,
    )
    assert record.stop_reason == "final_answer"
    assert record.final_answer == "42"
    assert record.error is None
    assert client.tool_choices == ["auto", "required"]


@pytest.mark.asyncio
async def test_agent_surfaces_other_api_errors():
    client = _FlakyClient(RuntimeError("AuthenticationError: invalid api key"), [])
    record = await run_question(
        question_id="q8",
        question="?",
        reference_answer=None,
        client=client,
        tools=[FinalAnswerTool()],
        system_prompt="test",
        max_steps=5,
    )
    assert record.stop_reason == "error"
    assert "AuthenticationError" in (record.error or "")


@pytest.mark.asyncio
async def test_agent_gives_up_after_repeated_empty_turns():
    client = _ScriptedClient([_empty_turn()] * (MAX_EMPTY_TURNS + 1))
    record = await run_question(
        question_id="q6",
        question="?",
        reference_answer=None,
        client=client,
        tools=[FinalAnswerTool()],
        system_prompt="test",
        max_steps=20,
    )
    assert record.stop_reason == "no_tool_call"
    assert record.final_answer is None
    assert client.calls == MAX_EMPTY_TURNS + 1


@pytest.mark.asyncio
async def test_agent_hits_max_steps():
    # Always loop a no-op tool call that won't terminate.
    looping = ModelResponse(
        text="thinking",
        tool_calls=[ToolUseBlock(id="c", name="calc", input={"a": 1, "b": 1})],
        stop_reason="tool_use",
        prompt_tokens=1,
        completion_tokens=1,
    )
    client = _ScriptedClient([looping] * 3)
    record = await run_question(
        question_id="q4",
        question="?",
        reference_answer=None,
        client=client,
        tools=[_CalcTool(), FinalAnswerTool()],
        system_prompt="test",
        max_steps=3,
    )
    assert record.stop_reason == "max_steps"
    assert record.final_answer is None
    assert len(record.steps) == 3


@pytest.mark.asyncio
async def test_unknown_tool_error_names_the_available_tools():
    """gpt-oss-20b traces show the model guessing `search?`, then `search`, then
    `search_query` — one wasted step each — because the error told it only that the name
    was wrong, never what the right names were."""
    client = _ScriptedClient(
        [
            ModelResponse(
                text="",
                tool_calls=[ToolUseBlock(id="c1", name="search", input={"query": "x"})],
                stop_reason="tool_use",
                prompt_tokens=1,
                completion_tokens=1,
            ),
            ModelResponse(
                text="",
                tool_calls=[ToolUseBlock(id="c2", name="final_answer", input={"answer": "5"})],
                stop_reason="tool_use",
                prompt_tokens=1,
                completion_tokens=1,
            ),
        ]
    )
    record = await run_question(
        question_id="q9",
        question="?",
        reference_answer=None,
        client=client,
        tools=[_CalcTool(), FinalAnswerTool()],
        system_prompt="test",
        max_steps=5,
    )
    error = record.steps[0].tool_results[0]
    assert error.is_error is True
    assert "unknown tool: 'search'" in error.content
    assert "calc" in error.content and "final_answer" in error.content


@pytest.mark.asyncio
async def test_agent_records_reasoning_on_each_step():
    """Reasoning models return empty `content` on intermediate turns. If the harness
    drops the reasoning, the judge grades a trace with no stated analysis in it."""
    client = _ScriptedClient(
        [
            ModelResponse(
                text="",
                reasoning_content="AAPL is the ticker; I need the FY2023 10-K.",
                tool_calls=[ToolUseBlock(id="c1", name="calc", input={"a": 1, "b": 1})],
                stop_reason="tool_use",
                prompt_tokens=1,
                completion_tokens=1,
            ),
            ModelResponse(
                text="",
                reasoning_content="That gives 2; answering.",
                tool_calls=[ToolUseBlock(id="c2", name="final_answer", input={"answer": "2"})],
                stop_reason="tool_use",
                prompt_tokens=1,
                completion_tokens=1,
            ),
        ]
    )
    record = await run_question(
        question_id="q10",
        question="?",
        reference_answer=None,
        client=client,
        tools=[_CalcTool(), FinalAnswerTool()],
        system_prompt="test",
        max_steps=5,
    )
    assert record.steps[0].assistant_reasoning == "AAPL is the ticker; I need the FY2023 10-K."
    assert record.steps[1].assistant_reasoning == "That gives 2; answering."


@pytest.mark.asyncio
async def test_agent_records_reasoning_on_an_empty_turn():
    """The nudge path writes its own StepRecord; it must carry reasoning too, or the
    turn where a harmony model reasoned and then emitted nothing is lost entirely."""
    client = _ScriptedClient(
        [
            ModelResponse(
                text="",
                reasoning_content="Thinking, but emitting nothing.",
                tool_calls=[],
                stop_reason="end_turn",
                prompt_tokens=1,
                completion_tokens=1,
            ),
            ModelResponse(
                text="",
                tool_calls=[ToolUseBlock(id="c1", name="final_answer", input={"answer": "42"})],
                stop_reason="tool_use",
                prompt_tokens=1,
                completion_tokens=1,
            ),
        ]
    )
    record = await run_question(
        question_id="q11",
        question="?",
        reference_answer=None,
        client=client,
        tools=[FinalAnswerTool()],
        system_prompt="test",
        max_steps=5,
    )
    assert record.steps[0].assistant_reasoning == "Thinking, but emitting nothing."
    assert record.stop_reason == "final_answer"
