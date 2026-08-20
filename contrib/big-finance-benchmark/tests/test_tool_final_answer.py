import pytest

from big_finance_harness.tools.base import ToolError
from big_finance_harness.tools.final_answer import FinalAnswerTool


@pytest.mark.asyncio
async def test_returns_answer_verbatim():
    tool = FinalAnswerTool()
    out = await tool.run({"answer": "$114.3 billion"})
    assert out == "$114.3 billion"


@pytest.mark.asyncio
async def test_is_terminal():
    assert FinalAnswerTool.is_terminal is True


@pytest.mark.asyncio
async def test_rejects_empty_answer():
    tool = FinalAnswerTool()
    with pytest.raises(ToolError):
        await tool.run({"answer": ""})
    with pytest.raises(ToolError):
        await tool.run({})


def test_terminal_tool_name_is_renameable_and_reaches_prompt_and_nudges(monkeypatch):
    """gpt-oss on vLLM emits `<|channel|>final_answer<|message|>…` — it writes the tool
    name where harmony expects a CHANNEL. vLLM's parser knows only analysis/commentary/
    final, so it drops the message whole and the turn arrives empty; that cost the 20B
    about a third of its runs on the 50-question subset.

    Renaming has to reach the tool, the system prompt AND both empty-turn nudges
    together — a prompt naming a tool the model wasn't given just produces another empty
    turn. The name is resolved at import, so this reloads the modules that bake it in.
    """
    import importlib

    from big_finance_harness import agent, prompts
    from big_finance_harness.tools import final_answer

    monkeypatch.setenv("BFH_TERMINAL_TOOL_NAME", "submit_answer")
    try:
        fa = importlib.reload(final_answer)
        pr = importlib.reload(prompts)
        ag = importlib.reload(agent)

        assert fa.TERMINAL_TOOL_NAME == "submit_answer"
        assert fa.FinalAnswerTool.name == "submit_answer"
        assert fa.FinalAnswerTool().spec.name == "submit_answer"
        # The tool is still the terminal one — the agent loop keys on this, not the name.
        assert fa.FinalAnswerTool.is_terminal is True

        blob = pr.SYSTEM_PROMPT + ag.EMPTY_TURN_NUDGE + ag.MALFORMED_OUTPUT_NUDGE
        assert "submit_answer" in blob
        # No stale reference left anywhere the model can read.
        assert "final_answer" not in blob
    finally:
        # Restore the default-named modules for the rest of the session.
        monkeypatch.delenv("BFH_TERMINAL_TOOL_NAME", raising=False)
        importlib.reload(final_answer)
        importlib.reload(prompts)
        importlib.reload(agent)


def test_terminal_tool_defaults_to_final_answer():
    """Default is unchanged so previously recorded runs stay comparable; the rename is
    opt-in per run (and should be set for both arms of an A/B, or neither)."""
    import os

    from big_finance_harness.tools.final_answer import TERMINAL_TOOL_NAME

    assert "BFH_TERMINAL_TOOL_NAME" not in os.environ
    assert TERMINAL_TOOL_NAME == "final_answer"
