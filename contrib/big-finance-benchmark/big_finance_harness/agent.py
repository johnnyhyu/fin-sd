"""ReAct agent loop.

The loop is intentionally short. On each step:

  1. Call the model with the current message history and tool definitions.
  2. If the model produced no tool calls, we treat the assistant text as the answer and
     terminate with `stop_reason='no_tool_call'`. (This is a fallback; the canonical way
     to terminate is by calling the terminal tool.) A turn with no text *and* no
     tool call has no answer to fall back on, so it is nudged rather than terminated —
     up to `MAX_EMPTY_TURNS` consecutive times.
  3. Dispatch every tool call concurrently. Capture stdout/error per call.
  4. If any tool call was the terminal one (`Tool.is_terminal`, whose registered name is
     BFH_TERMINAL_TOOL_NAME — "final_answer" by default), terminate with the captured
     answer. Note `stop_reason='final_answer'` is a fixed record value and does NOT
     track that rename.
  5. Otherwise append the assistant message and a tool-result message and loop.

The harness records every step's prompt/completion tokens and wallclock time; cost is
estimated against a pinned price table.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Literal

import litellm

from big_finance_harness import __version__
from big_finance_harness.models.base import ModelClient, ThinkingLevel
from big_finance_harness.tools.base import Tool, ToolError
from big_finance_harness.tools.final_answer import TERMINAL_TOOL_NAME
from big_finance_harness.types import (
    ContentBlock,
    Message,
    RunRecord,
    StepRecord,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

# 50 matches Bigeard 2025's Finance Agent Benchmark (the closest precedent for our 4-5
# tool ReAct setup). Modal value across the agent-benchmark literature. Empirically our
# slowest open models were hitting 25-29 steps with diverse retrieval queries — i.e.
# converging on answers rather than looping. 50 gives them headroom to land.
DEFAULT_MAX_STEPS = 50
# 64k gives high-effort reasoning models comfortable headroom (Opus 4.7 with adaptive
# high effort can use 20-30k tokens for thinking; 64k covers thinking + a long answer).
# max_tokens is a cap, so setting higher only costs more if the model actually fills it.
DEFAULT_MAX_OUTPUT_TOKENS = 65536
# No cumulative-token cap by default. Empirically the cap was firing at step 25-29 with
# diverse tool queries — i.e. cutting off methodical retrieval work, not catching loops.
# `max_steps` and `request_timeout` already bound runaway behavior; the field standard
# (Bigeard 2025, SWE-bench, τ-bench, AgentBench) uses step budgets alone. The parameter
# is preserved for future ablations.
DEFAULT_TOKEN_BUDGET: int | None = None
# How many *consecutive* empty assistant turns (no text, no tool call) we nudge through
# before giving up on the question. Harmony models (gpt-oss) routinely end a turn with
# reasoning only — the analysis channel says "so I'll call final_answer" and then the
# message closes with an empty final channel and no tool call. Treating that as terminal
# threw away otherwise-complete runs (the model had already done the retrieval work), so
# we hand the turn back with an explicit instruction instead. 3 is enough that a model
# which is genuinely stuck still terminates promptly.
MAX_EMPTY_TURNS = 3
# Both nudges interpolate the terminal tool's registered name (BFH_TERMINAL_TOOL_NAME)
# instead of hardcoding "final_answer": naming a tool the model was never given is
# exactly the sort of instruction that produces another empty turn.
EMPTY_TURN_NUDGE = (
    "Your last message was empty — it contained no text and no tool call. "
    f"Continue the task: either call a tool, or call `{TERMINAL_TOOL_NAME}` with your answer."
)
MALFORMED_OUTPUT_NUDGE = (
    "Your last message could not be read by the server. Reply with a single tool call — "
    f"either the next tool you need, or `{TERMINAL_TOOL_NAME}` with your answer."
)

# Signatures of a server rejecting its *own* model's output rather than our request.
# vLLM's gpt-oss path decodes generations with the harmony parser and raises a 500 when
# the model emits a malformed message (a header it can't read, or a stream that runs past
# the end of a message). It is a generation-shaped failure, not a request-shaped one, so
# re-asking with a different prompt usually clears it — unlike a 4xx, which will fail
# identically forever.
_MALFORMED_GENERATION_SIGNATURES = (
    "harmonyerror",
    "expecting start token",
    "unexpected tokens remaining in message header",
)


def _is_malformed_generation(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(sig in msg for sig in _MALFORMED_GENERATION_SIGNATURES)


async def _dispatch_tool(tool: Tool, call: ToolUseBlock) -> ToolResultBlock:
    try:
        content = await tool.run(call.input)
        return ToolResultBlock(tool_use_id=call.id, content=content, is_error=False)
    except ToolError as e:
        return ToolResultBlock(tool_use_id=call.id, content=str(e), is_error=True)
    except Exception as e:  # noqa: BLE001 — we want to surface anything to the model
        return ToolResultBlock(
            tool_use_id=call.id,
            content=f"unexpected tool error: {type(e).__name__}: {e}",
            is_error=True,
        )


async def run_question(
    *,
    question_id: str,
    question: str,
    reference_answer: str | None,
    client: ModelClient,
    tools: list[Tool],
    system_prompt: str,
    temperature: float | None = None,
    thinking: ThinkingLevel = "off",
    max_steps: int = DEFAULT_MAX_STEPS,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    token_budget: int | None = DEFAULT_TOKEN_BUDGET,
    trial_idx: int = 0,
) -> RunRecord:
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()

    tools_by_name: dict[str, Tool] = {t.name: t for t in tools}
    known_tool_names = [t.name for t in tools]
    tool_specs = [t.spec for t in tools]

    messages: list[Message] = [Message(role="user", content=[TextBlock(text=question)])]

    steps: list[StepRecord] = []
    final_answer: str | None = None
    stop_reason: Literal[
        "final_answer",
        "max_steps",
        "no_tool_call",
        "error",
        "context_exceeded",
        "token_budget",
    ] = "max_steps"
    error: str | None = None

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_reasoning_tokens = 0
    total_cached_tokens = 0
    total_cost_usd: float | None = None
    resolved_model: str | None = None
    consecutive_empty_turns = 0

    for step_idx in range(max_steps):
        step_started = time.monotonic()
        try:
            response = await client.chat(
                system=system_prompt,
                messages=messages,
                tools=tool_specs,
                temperature=temperature,
                thinking=thinking,
                max_output_tokens=max_output_tokens,
                # After an empty turn, stop asking nicely: force the next message to be
                # a tool call so the model either keeps working or answers.
                tool_choice="required" if consecutive_empty_turns else "auto",
            )
        except litellm.ContextWindowExceededError as e:
            # Distinct stop_reason so the analysis can flag context-overflow as a
            # different failure mode from API errors. We don't try to compact and retry —
            # for academic clarity, surfaced as its own outcome.
            stop_reason = "context_exceeded"
            error = f"{type(e).__name__}: {e}"
            break
        except Exception as e:  # noqa: BLE001
            # A server that failed to parse its own model's output (see
            # `_is_malformed_generation`) has produced nothing for us to record, but the
            # transcript so far is still good. Treat it like an empty turn: nudge, force
            # a tool call, and let the changed prompt shake the model out of whatever
            # malformed shape it settled into. Anything else ends the run.
            if _is_malformed_generation(e) and consecutive_empty_turns < MAX_EMPTY_TURNS:
                consecutive_empty_turns += 1
                messages.append(
                    Message(role="user", content=[TextBlock(text=MALFORMED_OUTPUT_NUDGE)])
                )
                continue
            stop_reason = "error"
            error = f"{type(e).__name__}: {e}"
            break

        total_prompt_tokens += response.prompt_tokens
        total_completion_tokens += response.completion_tokens
        if response.reasoning_tokens is not None:
            total_reasoning_tokens += response.reasoning_tokens
        if response.cached_tokens is not None:
            total_cached_tokens += response.cached_tokens
        if response.cost_usd is not None:
            total_cost_usd = (total_cost_usd or 0.0) + response.cost_usd
        if resolved_model is None and response.resolved_model:
            resolved_model = response.resolved_model

        assistant_blocks: list[ContentBlock] = []
        if response.text:
            assistant_blocks.append(TextBlock(text=response.text))
        for tc in response.tool_calls:
            # Some providers (Gemini) don't surface a stable call id; assign one.
            if not tc.id:
                tc.id = f"call_{uuid.uuid4().hex[:12]}"
            assistant_blocks.append(tc)

        if not response.tool_calls:
            messages.append(
                Message(
                    role="assistant",
                    content=assistant_blocks,
                    # Always carried; only put on the wire when
                    # BFH_REPLAY_REASONING is set (see models.base._replay_reasoning).
                    reasoning=response.reasoning_content,
                )
            )
            steps.append(
                StepRecord(
                    step=step_idx,
                    assistant_text=response.text,
                    assistant_reasoning=response.reasoning_content,
                    tool_calls=[],
                    tool_results=[],
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    wallclock_seconds=time.monotonic() - step_started,
                    request_id=response.request_id,
                    reasoning_tokens=response.reasoning_tokens,
                    cached_tokens=response.cached_tokens,
                    cost_usd=response.cost_usd,
                    raw_response=response.raw_response,
                )
            )
            # An empty turn (no text, no tool call) carries no answer to fall back on, so
            # ending here would discard the run. Nudge and let the model continue.
            if not response.text.strip() and consecutive_empty_turns < MAX_EMPTY_TURNS:
                consecutive_empty_turns += 1
                messages.append(
                    Message(role="user", content=[TextBlock(text=EMPTY_TURN_NUDGE)])
                )
                continue
            stop_reason = "no_tool_call"
            final_answer = response.text or None
            break

        consecutive_empty_turns = 0

        results: list[ToolResultBlock] = await asyncio.gather(
            *[
                _dispatch_tool(
                    tools_by_name.get(tc.name) or _UnknownTool(tc.name, known_tool_names),
                    tc,
                )
                for tc in response.tool_calls
            ]
        )

        # Detect terminal-tool firing. We use the `is_terminal` flag on the Tool base
        # class rather than hardcoding the tool's name so the loop stays correct if a
        # caller supplies a custom terminal tool with a different name.
        for tc, result in zip(response.tool_calls, results):
            tool = tools_by_name.get(tc.name)
            if tool is not None and tool.is_terminal and not result.is_error:
                final_answer = result.content
                stop_reason = "final_answer"

        messages.append(
            Message(
                role="assistant",
                content=assistant_blocks,
                # Always carried; only put on the wire when
                # BFH_REPLAY_REASONING is set (see models.base._replay_reasoning).
                reasoning=response.reasoning_content,
            )
        )
        messages.append(
            Message(role="tool", content=list(results))  # type: ignore[arg-type]
        )

        steps.append(
            StepRecord(
                step=step_idx,
                assistant_text=response.text,
                assistant_reasoning=response.reasoning_content,
                tool_calls=response.tool_calls,
                tool_results=results,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                wallclock_seconds=time.monotonic() - step_started,
                request_id=response.request_id,
                reasoning_tokens=response.reasoning_tokens,
                cached_tokens=response.cached_tokens,
                cost_usd=response.cost_usd,
                raw_response=response.raw_response,
            )
        )

        if stop_reason == "final_answer":
            break

        # Optional per-question cumulative token cap. Off by default; set explicitly
        # for cost-bounded ablation studies.
        if (
            token_budget is not None
            and (total_prompt_tokens + total_completion_tokens) >= token_budget
        ):
            stop_reason = "token_budget"
            break

    completed_at = datetime.now(timezone.utc).isoformat()
    return RunRecord(
        question_id=question_id,
        trial_idx=trial_idx,
        question=question,
        reference_answer=reference_answer,
        model=client.snapshot,
        resolved_model=resolved_model,
        harness_version=__version__,
        thinking=thinking,
        temperature=temperature,
        max_steps=max_steps,
        max_output_tokens=max_output_tokens,
        num_retries=getattr(client, "num_retries", 0),
        system_prompt=system_prompt,
        tool_specs=tool_specs,
        steps=steps,
        final_answer=final_answer,
        stop_reason=stop_reason,
        error=error,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_reasoning_tokens=total_reasoning_tokens,
        total_cached_tokens=total_cached_tokens,
        total_wallclock_seconds=time.monotonic() - started,
        # Cost is whatever LiteLLM accumulated across steps. If LiteLLM has no pricing
        # data for the snapshot (notably some Vercel AI Gateway routes), this stays None;
        # downstream analysis can recompute from token counts using vendor-published
        # rates. We deliberately do not maintain a separate price table.
        cost_usd=total_cost_usd,
        started_at=started_at,
        completed_at=completed_at,
    )


class _UnknownTool(Tool):
    """Surfaced to the model when it invents a tool name not in the catalog. The agent
    sees a tool_result with is_error=True so it can self-correct.

    The error names the available tools. Without that, gpt-oss-20b traces show the model
    guessing repeatedly — `search?`, then `search`, then `search_query` — burning a step
    on each, because `unknown tool: 'search'` alone gives it nothing to correct toward.
    """

    is_terminal = False

    def __init__(self, name: str, known: list[str] | None = None) -> None:
        self.name = name
        self.description = ""
        self.input_schema = {}
        self.known = known or []

    async def run(self, args):  # type: ignore[override]
        if self.known:
            raise ToolError(
                f"unknown tool: {self.name!r}. Available tools: "
                f"{', '.join(self.known)}. Re-issue the call using one of those names."
            )
        raise ToolError(f"unknown tool: {self.name!r}")
