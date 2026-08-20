from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from big_finance_harness.types import ToolSpec

# Key `LiteLLMClient` stores tool-call arguments under when the provider hands back a
# string that isn't valid JSON. Tools check for it so the model is told the truth
# ("your arguments didn't parse") instead of the misleading "code is required" — which
# reads as if it forgot a parameter it actually sent.
UNPARSED_ARGS_KEY = "_unparsed_arguments"


class ToolError(Exception):
    """Raised by a tool when it cannot complete its operation. The agent receives the
    string form as a tool_result with is_error=True and continues."""


def require_str(
    args: dict[str, Any],
    name: str,
    hint: str,
    *,
    strip: bool = True,
) -> str:
    """Read a required string argument, or raise a `ToolError` the model can act on.

    Observed failure modes this exists to make legible, all from real traces:
      - the provider returned unparseable JSON, so every argument is missing at once;
      - the model sent the value under no key at all (empty `{}` arguments);
      - the model sent a bare number where a string was expected.

    `hint` should show the expected shape, e.g. `'expected {"query": "<search terms>"}'`.
    """
    if UNPARSED_ARGS_KEY in args:
        raw = str(args[UNPARSED_ARGS_KEY])[:300]
        raise ToolError(
            "tool-call arguments were not valid JSON, so no parameters could be read. "
            f"Re-issue the call with valid JSON — {hint}. Received: {raw!r}"
        )
    value = args.get(name)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(
            f"{name!r} is required and must be a non-empty string — {hint}. "
            f"Received argument keys: {sorted(args)!r}"
        )
    return value.strip() if strip else value


class Tool(ABC):
    """Base class for harness tools.

    A tool exposes a JSON schema (`spec`) that is sent to the model, plus an async
    `run(input)` that takes the model's parsed arguments and returns a string. Returning
    a string keeps the trace serialization-trivial; tools that produce structured data
    should JSON-encode it themselves.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    is_terminal: bool = False

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )

    @abstractmethod
    async def run(self, args: dict[str, Any]) -> str: ...
