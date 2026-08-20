"""Single LiteLLM-backed client. Use `make_client(model_id)` to obtain one."""

from big_finance_harness.models.base import (
    FloatingAliasWarning,
    LiteLLMClient,
    ModelClient,
    parse_model_id,
)


def make_client(
    model_id: str,
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    extra_body: dict | None = None,
) -> ModelClient:
    return LiteLLMClient(
        model_id, api_base=api_base, api_key=api_key, extra_body=extra_body
    )


__all__ = [
    "FloatingAliasWarning",
    "LiteLLMClient",
    "ModelClient",
    "make_client",
    "parse_model_id",
]
