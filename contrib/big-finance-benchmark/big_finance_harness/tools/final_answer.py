from __future__ import annotations

import os
from typing import Any

from big_finance_harness.tools.base import Tool, require_str

# Name of the terminal tool, overridable because the default COLLIDES with harmony.
#
# gpt-oss served on vLLM emits a tool call as `<|channel|>final_answer<|message|>…`
# — it writes the tool's name where harmony expects a CHANNEL. vLLM's harmony
# parser only knows the channels `analysis`, `commentary` and `final`, so it does
# not recognise `final_answer`, and rather than surfacing the text it drops the
# message whole: the turn arrives with no content and no tool call. The agent loop
# reads that as an empty turn, nudges (MAX_EMPTY_TURNS), and eventually abandons
# the question. Measured on the 50-question public subset, it cost gpt-oss-20b
# roughly a THIRD of its runs — a harness artifact scored as model failure.
#
# Renaming the tool moves it out of the collision entirely; anything that isn't a
# harmony channel name works, and `submit_answer` is what the 20B/120B audit runs
# used. The default is left at `final_answer` so previously recorded runs stay
# comparable — set BFH_TERMINAL_TOOL_NAME=submit_answer for any gpt-oss arm (and,
# for a fair A/B, for the arm it is compared against).
#
# The name is interpolated into the system prompt and the agent's empty-turn
# nudges rather than written out in each, so those can never drift from the tool
# that is actually registered. The agent loop itself keys on `Tool.is_terminal`,
# never on the name.
TERMINAL_TOOL_NAME = (os.environ.get("BFH_TERMINAL_TOOL_NAME") or "final_answer").strip()


class FinalAnswerTool(Tool):
    """Terminal tool. The agent loop checks `is_terminal` and breaks when this fires.

    The tool returns the answer string verbatim so it appears in the trace as a tool
    result, but the agent loop also captures the answer separately on the run record.

    Its registered name is `TERMINAL_TOOL_NAME` (see above), not necessarily
    "final_answer".
    """

    name = TERMINAL_TOOL_NAME
    description = (
        "Submit your final answer to the question and end the session. The reference "
        "answers in this benchmark are typically a single number with units (e.g. "
        "'$410.5 million', '47%', '2.1'). Match that format when possible. If the "
        "question cannot be answered from the available sources, state that explicitly."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The final answer to the question.",
            },
        },
        "required": ["answer"],
        "additionalProperties": False,
    }
    is_terminal = True

    async def run(self, args: dict[str, Any]) -> str:
        # `strip=False` keeps the answer verbatim — it is what lands on
        # `RunRecord.final_answer` and what the judge compares to the reference.
        return require_str(
            args, "answer", 'expected {"answer": "$410.5 million"}', strip=False
        )
