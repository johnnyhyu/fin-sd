"""Unified provider client backed by LiteLLM.

LiteLLM normalizes the call/response shape across Anthropic, OpenAI (Chat Completions and
Responses), and Google (Vertex and AI Studio), and absorbs the provider-specific quirks
that previously required per-adapter handling (Anthropic's `temperature` deprecation on
Opus 4.7+, OpenAI's `/v1/responses` requirement for tool-calling reasoning models,
Gemini's Vertex auth, etc.).

We translate the harness's normalized message format to OpenAI Chat Completions shape
(LiteLLM's lingua franca) and back. `parse_model_id` validates `provider:snapshot` form
and warns on floating aliases.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from abc import ABC, abstractmethod
from typing import Any, Literal

import litellm

from big_finance_harness.types import (
    Message,
    ModelResponse,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)

# Global LiteLLM config: drop unsupported params silently (so e.g. `temperature` is
# omitted on Opus 4.7) and quiet the chatty debug logging.
litellm.drop_params = True
litellm.suppress_debug_info = True

ThinkingLevel = Literal["off", "low", "medium", "high"]

_DATE_SUFFIX_RE = re.compile(r"-20\d{2}-?\d{2}-?\d{2}$")

_THINKING_BUDGETS: dict[ThinkingLevel, int] = {
    "off": 0,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
}


class FloatingAliasWarning(UserWarning):
    """Emitted when a model snapshot has no date suffix."""


# Snapshots that have already triggered the floating-alias warning. We fire once per
# snapshot per process so a 928×6 run doesn't spam stderr with thousands of lines.
_WARNED_FLOATING_ALIASES: set[str] = set()


def parse_model_id(model_id: str) -> tuple[str, str]:
    """Parse `provider:snapshot` form. Returns (provider, snapshot)."""
    if ":" not in model_id:
        raise ValueError(f"model id must be of the form 'provider:snapshot' (got {model_id!r})")
    provider, snapshot = model_id.split(":", 1)
    if provider not in {
        "openrouter",
        "anthropic",
        "openai",
        "google",
        "vertex",
        "vertex-anthropic",
        "gateway",
        "local",
    }:
        raise ValueError(f"unsupported provider: {provider}")
    # Local self-hosted vLLM routes name a checkpoint (e.g. `local:epoch_1`), not a
    # dated vendor snapshot, so the floating-alias warning doesn't apply to them.
    if provider == "local":
        return provider, snapshot
    if not _DATE_SUFFIX_RE.search(snapshot) and snapshot not in _WARNED_FLOATING_ALIASES:
        _WARNED_FLOATING_ALIASES.add(snapshot)
        warnings.warn(
            f"snapshot {snapshot!r} has no date suffix; the trace will still capture "
            "the resolved snapshot returned by the provider. For the paper's headline "
            "table you may want to record a dated snapshot explicitly.",
            FloatingAliasWarning,
            stacklevel=2,
        )
    return provider, snapshot


def _to_litellm_model(provider: str, snapshot: str) -> str:
    """Translate `provider:snapshot` to LiteLLM's expected model identifier."""
    if provider == "openrouter":
        # OpenRouter. Snapshot is the OpenRouter slug `<vendor>/<model>` (e.g.
        # `anthropic/claude-opus-4.7`, `google/gemini-3.1-pro-preview`,
        # `openai/gpt-5.5`). One `OPENROUTER_API_KEY` gives access to every vendor,
        # so the harness needs no per-provider keys. LiteLLM reads the key from env.
        return f"openrouter/{snapshot}"
    if provider == "anthropic":
        return snapshot  # `claude-opus-4-7`, `claude-sonnet-4-6`, etc.
    if provider == "openai":
        return snapshot  # `gpt-5.5`, `gpt-5.4`, etc.
    if provider == "google":
        return f"gemini/{snapshot}"  # AI Studio path; needs GEMINI_API_KEY
    if provider == "vertex":
        return f"vertex_ai/{snapshot}"  # Vertex path for Gemini; needs ADC
    if provider == "vertex-anthropic":
        # Anthropic-on-Vertex. LiteLLM routes `vertex_ai/claude-*` snapshots to the
        # Vertex Anthropic endpoint. Vertex auto-routes to Provisioned Throughput
        # capacity when the project has PT configured for that model+location.
        return f"vertex_ai/{snapshot}"
    if provider == "gateway":
        # Vercel AI Gateway. Snapshot is `<lab>/<model>` (e.g. `deepseek/deepseek-v4-pro`,
        # `moonshotai/kimi-k2.6`). One key gives access to DeepSeek, Moonshot, Alibaba,
        # MiniMax, Z.ai, Google open-weight models. Needs VERCEL_AI_GATEWAY_API_KEY.
        return f"vercel_ai_gateway/{snapshot}"
    if provider == "local":
        # Local self-hosted OpenAI-compatible server (vLLM). LiteLLM's `hosted_vllm/`
        # prefix routes to an arbitrary `api_base` (passed per call in
        # LiteLLMClient.chat), so `local:epoch_1` served at http://localhost:8000/v1
        # becomes `hosted_vllm/epoch_1`.
        return f"hosted_vllm/{snapshot}"
    raise ValueError(f"unsupported provider: {provider}")


# Per-provider Vertex region defaults. Anthropic on Vertex is only served in specific
# regions (us-east5 supports Opus 4.7+; check the Vertex docs for the snapshot you're
# calling). Gemini defaults to `global` so callers can route to whichever region has
# capacity for the model they want. Both are overridable via env vars below.
def _vertex_location_for(provider: str) -> str:
    if provider == "vertex-anthropic":
        return os.environ.get("VERTEX_ANTHROPIC_LOCATION", "us-east5")
    return os.environ.get("VERTEXAI_LOCATION", "global")


def _replay_reasoning() -> bool:
    """Should assistant turns be sent back with their reasoning attached?

    Off by default; set BFH_REPLAY_REASONING=1 to turn it on.

    A harmony assistant turn is a sequence of channelled messages, and the
    `analysis` channel — the reasoning — is part of it. The Chat Completions wire
    format has no field for that, so a normal replay sends back only the final text
    and the tool calls; vLLM then rebuilds the conversation with the analysis
    message MISSING from every prior turn. The model sees a history in which it
    apparently answered without thinking, which is not the distribution it was
    trained on.

    Sending `reasoning_content` back lets vLLM reconstruct the analysis message, so
    the replayed context matches what the model actually produced.

    It is opt-in rather than default for three reasons: it is a server-specific
    field that only vLLM's harmony path consumes (hosted providers ignore or reject
    it), replaying reasoning grows the prompt on every step of a 50-step run, and —
    the reason it is a knob rather than a fix — turning it on CHANGES THE
    CONDITIONING, so a run with it set is not comparable to the recorded runs
    without it. Flip it for both arms of an A/B or neither.

    Read at call time, not import time, so a test (or a caller that sets the
    variable after import) sees the current value.
    """
    return (os.environ.get("BFH_REPLAY_REASONING") or "").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
    )


def _to_oai_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    """Translate normalized messages into OpenAI Chat Completions shape."""
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        if m.role == "user":
            text = "".join(b.text for b in m.content if isinstance(b, TextBlock))
            if text:
                out.append({"role": "user", "content": text})
        elif m.role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for b in m.content:
                if isinstance(b, TextBlock):
                    text_parts.append(b.text)
                elif isinstance(b, ToolUseBlock):
                    tool_calls.append(
                        {
                            "id": b.id,
                            "type": "function",
                            "function": {
                                "name": b.name,
                                "arguments": json.dumps(b.input, ensure_ascii=False),
                            },
                        }
                    )
            entry: dict[str, Any] = {"role": "assistant"}
            if text_parts:
                entry["content"] = "".join(text_parts)
            if tool_calls:
                entry["tool_calls"] = tool_calls
            if "content" not in entry and "tool_calls" not in entry:
                continue
            # Checked AFTER the drop test above: reasoning alone is not a turn worth
            # replaying, and an entry carrying only `reasoning_content` is not a valid
            # assistant message.
            if m.reasoning and _replay_reasoning():
                entry["reasoning_content"] = m.reasoning
            out.append(entry)
        elif m.role == "tool":
            for b in m.content:
                if isinstance(b, ToolResultBlock):
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": b.tool_use_id,
                            "content": (f"[ERROR] {b.content}" if b.is_error else b.content),
                        }
                    )
    return out


# Harmony control tokens that vLLM's `openai` tool parser sometimes leaves glued to the
# function name it extracts (e.g. `final_answer<|channel|>commentary`). Left as-is the
# agent sees an unknown tool and burns a step self-correcting — and if the leak lands on
# the terminal `final_answer` call late in a run, the answer is lost entirely.
_TOOL_NAME_JUNK_RE = re.compile(r"<\|.*$|\s+$")
# Harmony addresses tools as `to=functions.web_search`; a partial parse can leave the
# namespace on the extracted name. Some routes use `tools.` or a bare `functions:`.
_TOOL_NAME_NAMESPACE_RE = re.compile(r"^(?:functions|tools)[.:]")
# Trailing punctuation the parser picks up from surrounding markup (`web_search?`,
# `python_exec.`, `"fetch_url"`). Observed on gpt-oss traces as `search?`, `functions?`.
_TOOL_NAME_TRAILING_RE = re.compile(r'[\s?.,:;!"\')\]]+$')


def _clean_tool_name(name: str | None) -> str:
    """Strip transport artifacts from a provider-reported tool name.

    Deliberately limited to *corruption* — control tokens, namespace prefixes, trailing
    punctuation. We do not map a name onto the nearest real tool: a model that invents
    `search` instead of `web_search` made a tool-calling error, and silently repairing it
    would make the scaffold, not the model, responsible for the recovery. Those names
    fall through to `agent._UnknownTool`, which names the valid tools back to the model.
    """
    cleaned = _TOOL_NAME_JUNK_RE.sub("", (name or "").strip())
    cleaned = _TOOL_NAME_NAMESPACE_RE.sub("", cleaned)
    return _TOOL_NAME_TRAILING_RE.sub("", cleaned)


def _extract_reasoning(msg: Any) -> str | None:
    """Pull the provider's chain of thought off a LiteLLM response message.

    Reasoning models return an empty `content` on every intermediate turn and put the
    analyst work in a separate field. LiteLLM normalizes that to `reasoning_content`,
    but not every route does: some set `reasoning`, and some stash it under
    `provider_specific_fields`. We check all three.

    Without this the harness records `assistant_text=''` for every step of a gpt-oss or
    o-series run, and the judge — which grades rubric lines like "Identifies DAY as the
    ticker" against the trace — sees tool calls with no stated reasoning behind them.
    """
    for attr in ("reasoning_content", "reasoning"):
        value = getattr(msg, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    extra = getattr(msg, "provider_specific_fields", None) or {}
    if isinstance(extra, dict):
        for key in ("reasoning_content", "reasoning"):
            value = extra.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def _to_oai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


class ModelClient(ABC):
    """Adapter interface — kept for testability. The real implementation is
    `LiteLLMClient`; tests can swap in a stub."""

    snapshot: str

    @abstractmethod
    async def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        temperature: float | None = None,
        thinking: ThinkingLevel = "off",
        max_output_tokens: int = 65536,
        tool_choice: str = "auto",
    ) -> ModelResponse: ...


class LiteLLMClient(ModelClient):
    """Concrete client using LiteLLM for all providers."""

    # Built-in retry count passed to litellm. 12 retries with default exponential backoff
    # gives a cumulative wait of ~10 min, which clears any per-minute Vertex token quota
    # window even under sustained multi-model load.
    NUM_RETRIES = 12

    def __init__(
        self,
        model_id: str,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.model_id = model_id
        provider, snapshot = parse_model_id(model_id)
        self.provider = provider
        self.snapshot = snapshot
        self._litellm_model = _to_litellm_model(provider, snapshot)
        self.num_retries = self.NUM_RETRIES
        # Explicit endpoint override for `local:` (self-hosted vLLM) routes. Hosted
        # providers leave these None and let LiteLLM resolve credentials from env.
        self.api_base = api_base
        self.api_key = api_key
        # Server-specific request-body fields forwarded verbatim (see
        # `config.VLLM_HARMONY_EXTRA_BODY` for why local gpt-oss routes need one).
        self.extra_body = extra_body or None

    async def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        temperature: float | None = None,
        thinking: ThinkingLevel = "off",
        max_output_tokens: int = 65536,
        tool_choice: str = "auto",
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self._litellm_model,
            "messages": _to_oai_messages(system, messages),
            "tools": _to_oai_tools(tools),
            # "auto" normally; the agent switches to "required" to pull a model out of
            # an empty-turn stall (see agent.MAX_EMPTY_TURNS).
            "tool_choice": tool_choice,
            "max_tokens": max_output_tokens,
            # Built-in retry with exponential backoff for RateLimitError, Timeout,
            # InternalServerError, and other transient classes. 12 retries × default
            # backoff = ~10 min cumulative, which clears any per-minute Vertex quota
            # window even under sustained multi-model load.
            "num_retries": self.num_retries,
            # Per-call timeout. Without this, a single hung API request can freeze the
            # entire orchestrator (we observed DeepSeek V4-Pro hang on a question for
            # 30+ min). 30 min is generous for high-effort reasoning on hard questions
            # within max_steps=50 — some thinking calls legitimately run 5-15 min, this
            # gives belt-and-suspenders headroom while still bounding worst-case hangs.
            "request_timeout": 1800,
        }
        # Local self-hosted vLLM routes carry their endpoint + key explicitly (hosted
        # providers resolve these from env inside LiteLLM).
        if self.api_base is not None:
            kwargs["api_base"] = self.api_base
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.extra_body is not None:
            kwargs["extra_body"] = dict(self.extra_body)
        # Temperature defaults to None — let each vendor use its own default. Anthropic
        # 4.7+ deprecates it; OpenAI reasoning models fix it at 1.0; Haiku/Gemini default
        # to 1.0. Pass only if the caller explicitly sets it (e.g. for ablations).
        if temperature is not None:
            kwargs["temperature"] = temperature
        # Privacy: don't let OpenAI retain proprietary benchmark questions for training.
        if self.provider == "openai":
            kwargs["store"] = False
        # Vertex paths need explicit project + location passed per call.
        # `X-Vertex-AI-LLM-Request-Type: dedicated` forces Provisioned Throughput-only
        # routing: when PT capacity is exhausted, Vertex returns 429 instead of silently
        # falling back to pay-as-you-go. Our `num_retries` with exponential backoff
        # handles the queueing. The header is honored for both Gemini and Claude on
        # Vertex (per https://docs.cloud.google.com/vertex-ai/generative-ai/docs/
        # provisioned-throughput/use-provisioned-throughput). Set
        # `VERTEX_DISABLE_DEDICATED=1` to drop the header and allow PAYG fallback —
        # appropriate when running without provisioned capacity.
        if self.provider in ("vertex", "vertex-anthropic"):
            project = os.environ.get("VERTEXAI_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
            if project:
                kwargs["vertex_project"] = project
            kwargs["vertex_location"] = _vertex_location_for(self.provider)
            if not os.environ.get("VERTEX_DISABLE_DEDICATED"):
                kwargs["extra_headers"] = {"X-Vertex-AI-LLM-Request-Type": "dedicated"}
        if thinking != "off":
            # LiteLLM passes provider-specific thinking kwargs through. For OpenAI
            # reasoning models it sends `reasoning_effort`; for Anthropic it sends
            # `thinking={"type": "enabled", "budget_tokens": N}`; for Gemini it sends
            # `thinking_budget`. With `drop_params=True` set globally, params not
            # supported by the target are silently dropped.
            kwargs["reasoning_effort"] = thinking
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": _THINKING_BUDGETS[thinking],
            }

        try:
            response = await litellm.acompletion(**kwargs)
        except (litellm.BadRequestError, litellm.InternalServerError) as e:
            # Newer Anthropic models (e.g. Opus 4.7 with adaptive thinking) deprecate
            # `temperature`. Vertex sometimes wraps the deprecation as a 500 instead of
            # a 400, so we catch both. LiteLLM's drop_params mapping may lag the API.
            msg = str(e).lower()
            if "temperature" in msg and "deprecated" in msg:
                kwargs.pop("temperature", None)
                response = await litellm.acompletion(**kwargs)
            else:
                raise

        choice = response.choices[0]
        msg = choice.message

        tool_calls: list[ToolUseBlock] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {"_unparsed_arguments": tc.function.arguments}
            tool_calls.append(
                ToolUseBlock(id=tc.id, name=_clean_tool_name(tc.function.name), input=args)
            )

        finish = choice.finish_reason or ""
        stop_map = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
        }
        stop_reason = stop_map.get(finish, "other")

        usage = getattr(response, "usage", None)
        # LiteLLM stamps `_hidden_params["response_cost"]` with a per-call USD estimate
        # based on its bundled price database. We read this rather than maintaining our
        # own price table since LiteLLM's table is updated alongside vendor releases.
        hidden = getattr(response, "_hidden_params", {}) or {}
        cost = hidden.get("response_cost")

        reasoning_tokens: int | None = None
        cached_tokens: int | None = None
        if usage is not None:
            ctd = getattr(usage, "completion_tokens_details", None)
            if ctd is not None:
                rt = getattr(ctd, "reasoning_tokens", None)
                if rt is not None:
                    reasoning_tokens = int(rt)
            ptd = getattr(usage, "prompt_tokens_details", None)
            if ptd is not None:
                ct = getattr(ptd, "cached_tokens", None)
                if ct is not None:
                    cached_tokens = int(ct)
            # Some Anthropic responses surface cache hits flat on usage rather than
            # nested under prompt_tokens_details.
            if cached_tokens is None:
                ari = getattr(usage, "cache_read_input_tokens", None)
                if ari is not None:
                    cached_tokens = int(ari)

        # Capture the full normalized response for replay/audit. We dump the litellm
        # response (which is OpenAI-shape) plus `_hidden_params` so cost, latency
        # metadata, and provider-specific fields all survive.
        raw_response: dict[str, Any] = {}
        try:
            if hasattr(response, "model_dump"):
                raw_response["response"] = response.model_dump(mode="json")
            else:
                raw_response["response"] = json.loads(json.dumps(response, default=str))
        except Exception as e:  # noqa: BLE001
            raw_response["response_error"] = f"{type(e).__name__}: {e}"
        raw_response["hidden_params"] = {
            k: v for k, v in hidden.items() if k != "additional_headers"
        }

        return ModelResponse(
            text=msg.content or "",
            reasoning_content=_extract_reasoning(msg),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            resolved_model=getattr(response, "model", None),
            cost_usd=float(cost) if cost is not None else None,
            request_id=getattr(response, "id", None),
            reasoning_tokens=reasoning_tokens,
            cached_tokens=cached_tokens,
            raw_response=raw_response,
        )
