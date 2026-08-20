"""Catalog resolution: how a `llms:` entry becomes `make_client` arguments."""

from __future__ import annotations

from big_finance_harness.config import (
    VLLM_HARMONY_EXTRA_BODY,
    InferenceConfig,
    LLMSpec,
)


def test_local_route_defaults_to_harmony_stop_tokens():
    # Without these, vLLM's chat-completions path runs gpt-oss past `<|call|>` and 500s
    # parsing its own output on every tool-enabled request.
    resolved = LLMSpec(model_id="epoch_1", base_url="http://localhost:8000/v1").resolve(
        "local-model1"
    )
    assert resolved.model_id == "local:epoch_1"
    assert resolved.extra_body == VLLM_HARMONY_EXTRA_BODY


def test_local_route_extra_body_is_overridable():
    spec = LLMSpec(
        model_id="epoch_1",
        base_url="http://localhost:8000/v1",
        extra_body={"stop_token_ids": [7]},
    )
    assert spec.resolve("local-model1").extra_body == {"stop_token_ids": [7]}


def test_local_route_extra_body_can_be_emptied():
    spec = LLMSpec(model_id="epoch_1", base_url="http://localhost:8000/v1", extra_body={})
    assert spec.resolve("local-model1").extra_body == {}


def test_hosted_route_sends_no_extra_body():
    resolved = LLMSpec(model_id="openrouter:openai/gpt-oss-20b").resolve("gpt-oss-20b")
    assert resolved.model_id == "openrouter:openai/gpt-oss-20b"
    assert resolved.extra_body is None
    assert resolved.api_base is None


def test_inference_temperature_defaults_to_none():
    # None means "send no temperature", which is what every run before this knob existed
    # did. Kept as the default so hosted routes still use each vendor's own default.
    cfg = InferenceConfig(model_name="m", dataset="d", run_id="r")
    assert cfg.temperature is None


def test_inference_temperature_zero_survives_the_config():
    # `0` must reach the client as 0.0, not be swallowed as falsy anywhere in between —
    # a local gpt-oss route with no temperature samples at vLLM's default 1.0, which is
    # what makes a 50-item single-trial comparison measure the sampler.
    cfg = InferenceConfig(model_name="m", dataset="d", run_id="r", temperature=0)
    assert cfg.temperature == 0.0
