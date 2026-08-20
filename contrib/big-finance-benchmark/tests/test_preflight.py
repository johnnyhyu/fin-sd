"""Preflight must refuse to start a run whose results would be invalid."""

from __future__ import annotations

import pytest

from big_finance_harness.config import ResolvedModel
from big_finance_harness.preflight import assert_ready, check_environment, probe_local_route

_ALL_KEYS = (
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "VERCEL_AI_GATEWAY_API_KEY",
    "VERTEXAI_PROJECT",
    "GOOGLE_CLOUD_PROJECT",
    "SERP_API_KEY",
    "TAVILY_API_KEY",
    "SEC_EDGAR_USER_AGENT",
    "BFH_SKIP_PREFLIGHT",
)


@pytest.fixture
def clean_env(monkeypatch):
    """conftest stubs the tool keys for every test; clear them here so preflight sees a
    genuinely empty environment."""
    for key in _ALL_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_missing_everything_reports_every_problem(clean_env):
    problems = check_environment(["openrouter:openai/gpt-oss-20b"])
    assert any("OPENROUTER_API_KEY" in p for p in problems)
    assert any("SERP_API_KEY" in p for p in problems)
    assert any("SEC_EDGAR_USER_AGENT" in p for p in problems)


def test_a_fully_configured_openrouter_run_passes(clean_env):
    clean_env.setenv("OPENROUTER_API_KEY", "sk-or-x")
    clean_env.setenv("TAVILY_API_KEY", "tv-x")
    clean_env.setenv("SEC_EDGAR_USER_AGENT", "Me me@example.com")
    assert check_environment(["openrouter:openai/gpt-oss-20b"]) == []


def test_judges_are_checked_too(clean_env):
    clean_env.setenv("SERP_API_KEY", "s")
    clean_env.setenv("SEC_EDGAR_USER_AGENT", "Me me@example.com")
    problems = check_environment([], ["openrouter:anthropic/claude-opus-4.7"])
    assert len(problems) == 1
    assert "OPENROUTER_API_KEY" in problems[0]


def test_grade_only_runs_do_not_require_tool_keys(clean_env):
    clean_env.setenv("OPENROUTER_API_KEY", "sk-or-x")
    assert check_environment([], ["openrouter:x/y"], require_tool_keys=False) == []


def test_local_routes_need_no_provider_key(clean_env):
    """A `local:` route carries its credentials in the config, not the environment."""
    clean_env.setenv("SERP_API_KEY", "s")
    clean_env.setenv("SEC_EDGAR_USER_AGENT", "Me me@example.com")
    assert check_environment(["local:epoch_1"]) == []


def test_unreachable_local_server_is_reported():
    model = ResolvedModel(
        label="local-model1",
        model_id="local:epoch_1",
        # Port 1 is reserved and never listening, so this fails fast.
        api_base="http://127.0.0.1:1/v1",
        api_key="token-abc123",
    )
    problems = probe_local_route(model, timeout_s=1.0)
    assert len(problems) == 1
    assert "cannot reach the local server" in problems[0]


def test_local_server_serving_the_wrong_model_is_reported(httpx_mock):
    httpx_mock.add_response(
        url="http://127.0.0.1:8000/v1/models",
        json={"data": [{"id": "epoch_2"}]},
    )
    model = ResolvedModel(
        label="local-model1",
        model_id="local:epoch_1",
        api_base="http://127.0.0.1:8000/v1",
        api_key="token-abc123",
    )
    problems = probe_local_route(model)
    assert len(problems) == 1
    assert "--served-model-name" in problems[0]


def test_local_server_serving_the_right_model_passes(httpx_mock):
    httpx_mock.add_response(
        url="http://127.0.0.1:8000/v1/models",
        json={"data": [{"id": "epoch_1"}]},
    )
    model = ResolvedModel(
        label="local-model1",
        model_id="local:epoch_1",
        api_base="http://127.0.0.1:8000/v1",
        api_key="token-abc123",
    )
    assert probe_local_route(model) == []


def test_assert_ready_aborts_with_an_actionable_message(clean_env):
    with pytest.raises(SystemExit) as exc:
        assert_ready([ResolvedModel(label="m", model_id="openrouter:openai/gpt-oss-20b")])
    message = str(exc.value)
    assert "preflight failed" in message
    assert "BFH_SKIP_PREFLIGHT=1" in message


def test_preflight_can_be_bypassed(clean_env):
    clean_env.setenv("BFH_SKIP_PREFLIGHT", "1")
    assert_ready([ResolvedModel(label="m", model_id="openrouter:openai/gpt-oss-20b")])
