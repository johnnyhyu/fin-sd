import pytest

from big_finance_harness.models.base import parse_model_id


def test_accepts_dated_anthropic_snapshot():
    assert parse_model_id("anthropic:claude-opus-4-7-20260416") == (
        "anthropic",
        "claude-opus-4-7-20260416",
    )


def test_accepts_dated_openai_snapshot():
    assert parse_model_id("openai:gpt-5.2-2026-01-15") == ("openai", "gpt-5.2-2026-01-15")


def test_warns_on_floating_alias():
    from big_finance_harness.models.base import FloatingAliasWarning

    with pytest.warns(FloatingAliasWarning, match="no date suffix"):
        provider, snapshot = parse_model_id("anthropic:claude-opus-4-7")
    assert provider == "anthropic"
    assert snapshot == "claude-opus-4-7"


def test_accepts_preview_alias_with_warning():
    from big_finance_harness.models.base import FloatingAliasWarning

    with pytest.warns(FloatingAliasWarning):
        provider, snapshot = parse_model_id("google:gemini-3.1-pro-preview")
    assert snapshot == "gemini-3.1-pro-preview"


def test_accepts_openrouter_slug():
    from big_finance_harness.models.base import _to_litellm_model

    provider, snapshot = parse_model_id("openrouter:anthropic/claude-opus-4.7")
    assert provider == "openrouter"
    assert snapshot == "anthropic/claude-opus-4.7"
    assert _to_litellm_model(provider, snapshot) == "openrouter/anthropic/claude-opus-4.7"


def test_accepts_local_vllm_route_without_warning():
    import warnings

    from big_finance_harness.models.base import _to_litellm_model

    # A local checkpoint name has no date suffix but must NOT trigger the floating
    # alias warning (it's a self-hosted route, not a vendor snapshot).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        provider, snapshot = parse_model_id("local:epoch_1")
    assert provider == "local"
    assert snapshot == "epoch_1"
    assert _to_litellm_model(provider, snapshot) == "hosted_vllm/epoch_1"


def test_local_client_carries_api_base_and_key():
    from big_finance_harness.models import make_client

    client = make_client("local:epoch_1", api_base="http://localhost:8000/v1", api_key="tok")
    assert client.api_base == "http://localhost:8000/v1"
    assert client.api_key == "tok"


def test_local_client_forwards_extra_body():
    from big_finance_harness.models import make_client

    client = make_client(
        "local:epoch_1",
        api_base="http://localhost:8000/v1",
        extra_body={"stop_token_ids": [200012, 200002]},
    )
    assert client.extra_body == {"stop_token_ids": [200012, 200002]}


def test_hosted_client_has_no_extra_body_by_default():
    from big_finance_harness.models import make_client

    client = make_client("openrouter:openai/gpt-oss-20b")
    assert client.extra_body is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        # vLLM's `openai` tool parser leaks harmony control tokens into the name.
        ("final_answer<|channel|>commentary", "final_answer"),
        ("web_search ", "web_search"),
        ("python_exec", "python_exec"),
        (None, ""),
    ],
)
def test_cleans_tool_names(raw, expected):
    from big_finance_harness.models.base import _clean_tool_name

    assert _clean_tool_name(raw) == expected


def test_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unsupported provider"):
        parse_model_id("cohere:command-r-2026-01-01")


def test_rejects_missing_colon():
    with pytest.raises(ValueError, match="provider:snapshot"):
        parse_model_id("claude-opus-4-7-20260416")


class _FakeMessage:
    """Shaped like a LiteLLM response message."""

    def __init__(self, **fields):
        self.content = fields.pop("content", "")
        for k, v in fields.items():
            setattr(self, k, v)


def test_extracts_reasoning_content():
    from big_finance_harness.models.base import _extract_reasoning

    msg = _FakeMessage(content="", reasoning_content="Need to check the 10-K.")
    assert _extract_reasoning(msg) == "Need to check the 10-K."


def test_extracts_reasoning_from_the_reasoning_alias():
    from big_finance_harness.models.base import _extract_reasoning

    assert _extract_reasoning(_FakeMessage(content="", reasoning="alt field")) == "alt field"


def test_extracts_reasoning_from_provider_specific_fields():
    from big_finance_harness.models.base import _extract_reasoning

    msg = _FakeMessage(content="", provider_specific_fields={"reasoning_content": "nested"})
    assert _extract_reasoning(msg) == "nested"


def test_reasoning_absent_or_blank_is_none():
    from big_finance_harness.models.base import _extract_reasoning

    assert _extract_reasoning(_FakeMessage(content="hi")) is None
    assert _extract_reasoning(_FakeMessage(content="hi", reasoning_content="   ")) is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Harmony namespace prefixes left behind by a partial tool-call parse.
        ("functions.web_search", "web_search"),
        ("functions:fetch_url", "fetch_url"),
        ("tools.python_exec", "python_exec"),
        # Trailing punctuation scraped in from surrounding markup. Observed verbatim in
        # gpt-oss-20b traces as `search?` and `functions?`.
        ("web_search?", "web_search"),
        ("python_exec.", "python_exec"),
        ('"fetch_url"', '"fetch_url'),
        # A genuinely invented name is normalized but NOT remapped onto a real tool —
        # that is the model's error to recover from, not the scaffold's to hide.
        ("search?", "search"),
        ("search_query", "search_query"),
    ],
)
def test_normalizes_tool_name_transport_artifacts(raw, expected):
    from big_finance_harness.models.base import _clean_tool_name

    assert _clean_tool_name(raw) == expected


def _assistant_turn():
    from big_finance_harness.types import Message, TextBlock, ToolUseBlock

    return [
        Message(role="user", content=[TextBlock(text="q")]),
        Message(
            role="assistant",
            content=[
                TextBlock(text="Checking the 10-K."),
                ToolUseBlock(id="t1", name="fetch_url", input={"url": "https://x"}),
            ],
            reasoning="The filing lists operating income in the income statement.",
        ),
    ]


def test_reasoning_is_not_replayed_by_default(monkeypatch):
    """Off by default: replaying reasoning changes the conditioning, so a run with it on
    is not comparable to the recorded runs without it."""
    from big_finance_harness.models.base import _to_oai_messages

    monkeypatch.delenv("BFH_REPLAY_REASONING", raising=False)
    out = _to_oai_messages("sys", _assistant_turn())
    assistant = [m for m in out if m["role"] == "assistant"][0]
    assert "reasoning_content" not in assistant
    # The rest of the turn is unaffected either way.
    assert assistant["content"] == "Checking the 10-K."
    assert assistant["tool_calls"][0]["function"]["name"] == "fetch_url"


def test_reasoning_is_replayed_when_enabled(monkeypatch):
    """With BFH_REPLAY_REASONING=1 the analysis channel goes back to the server, so
    vLLM's harmony path can rebuild the assistant turn the model actually produced
    instead of one that answered without thinking."""
    from big_finance_harness.models.base import _to_oai_messages

    monkeypatch.setenv("BFH_REPLAY_REASONING", "1")
    out = _to_oai_messages("sys", _assistant_turn())
    assistant = [m for m in out if m["role"] == "assistant"][0]
    assert assistant["reasoning_content"].startswith("The filing lists operating income")


def test_replay_reasoning_flag_is_read_at_call_time(monkeypatch):
    """Read per call, not at import: a value set after import must still take effect."""
    from big_finance_harness.models.base import _replay_reasoning

    for value, expected in [("1", True), ("true", True), ("0", False), ("", False)]:
        monkeypatch.setenv("BFH_REPLAY_REASONING", value)
        assert _replay_reasoning() is expected, value
    monkeypatch.delenv("BFH_REPLAY_REASONING", raising=False)
    assert _replay_reasoning() is False


def test_replay_never_emits_a_reasoning_only_assistant_message(monkeypatch):
    """A turn with no text and no tool calls is dropped entirely; attaching reasoning to
    it would put an assistant message on the wire with no content at all."""
    from big_finance_harness.models.base import _to_oai_messages
    from big_finance_harness.types import Message

    monkeypatch.setenv("BFH_REPLAY_REASONING", "1")
    out = _to_oai_messages(
        "sys", [Message(role="assistant", content=[], reasoning="thinking, but silent")]
    )
    assert [m for m in out if m["role"] == "assistant"] == []
