"""Config-driven run surface for the Big Finance harness.

Mirrors FinanceReasoning/MMLU-Pro: a single `config/config.yaml` is the source of
truth, consumed by `inference.py` (generate traces) and `evaluation.py` (judge
them). Both are invoked as:

    python inference.py  --config config/config.yaml
    python evaluation.py --config config/config.yaml

The `llms:` catalog maps a friendly MODEL_KEY to a route. Hosted models use the
harness's `provider:snapshot` id (e.g. `openrouter:anthropic/claude-opus-4.7`)
and resolve credentials from the environment inside LiteLLM. A `base_url` turns an
entry into a local self-hosted vLLM route (OpenAI-compatible): `model_id` is the
served checkpoint name and `api_key` falls back to ${VLLM_API_KEY}
(default 'token-abc123'), exactly like FinanceReasoning's `local-model1`.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

# A string that is *nothing but* a single ${VAR} / $VAR placeholder. Only these are
# expanded, so arbitrary config text containing a literal '$' is left untouched.
_ENV_PLACEHOLDER = re.compile(r"^\$\{?\w+\}?$")

# Load the unified repo-root .env (three levels up: big_finance_harness/ ->
# big-finance-benchmark/ -> repo root) so OPENROUTER_API_KEY / VLLM_API_KEY are
# available when configs resolve their `${...}` placeholders below.
#
# Narrowed from a bare `except Exception: pass`. Swallowing everything meant that if
# python-dotenv were missing or the file unreadable, every credential silently resolved
# to unset and the failure surfaced hundreds of API calls later as empty traces. Only
# the import is optional now, and it says so.
_ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is a hard dependency
    warnings.warn(
        "python-dotenv is not installed; the repo-root .env will not be loaded and "
        "credentials must already be exported. `pip install -e .` installs it.",
        RuntimeWarning,
        stacklevel=2,
    )
else:
    load_dotenv(_ROOT_ENV)
    load_dotenv()  # also honor a .env in the current working directory


def _expand_env(obj):
    """Recursively expand ${VAR} / $VAR placeholders in string values. Only strings
    that are exactly a placeholder are expanded; other text is returned verbatim."""
    if isinstance(obj, str):
        return os.path.expandvars(obj) if _ENV_PLACEHOLDER.match(obj) else obj
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj


# Harmony's "assistant is done acting" tokens: `<|call|>` (200012, the model just emitted
# a tool call) and `<|return|>` (200002, the model finished its turn).
#
# vLLM registers these as default stop tokens for gpt-oss, but its chat-completions path
# passes the *request's* `stop_token_ids` (which defaults to `[]`) straight into
# SamplingParams and never merges the server defaults in. So a plain
# /v1/chat/completions call runs past `<|call|>`, the model keeps writing another harmony
# message, and vLLM's own output parser dies with
# `HarmonyError: Unexpected token 12606 while expecting start token 200006` — a 500 on
# *every* tool-enabled request, i.e. every request this harness makes. Sending the stop
# tokens explicitly is the fix.
#
# These ids only exist in the harmony (o200k_harmony) vocabulary, so they are inert for a
# local server hosting anything else. Set `extra_body: {}` on a catalog entry to opt out.
VLLM_HARMONY_EXTRA_BODY: dict = {"stop_token_ids": [200012, 200002]}


class ResolvedModel(BaseModel):
    """A catalog entry resolved into `make_client` arguments."""

    label: str
    model_id: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    extra_body: Optional[dict] = None


class LLMSpec(BaseModel):
    model_id: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    display: Optional[str] = None
    # Raw request-body fields forwarded verbatim to the server (LiteLLM `extra_body`).
    # The escape hatch for server-specific knobs the OpenAI schema has no field for —
    # notably vLLM's `stop_token_ids`, which a gpt-oss route needs (see below).
    extra_body: Optional[dict] = None

    def resolve(self, label: str) -> ResolvedModel:
        base_url = (self.base_url or "").strip()
        if base_url:
            # Local self-hosted vLLM (OpenAI-compatible). The harness client id is
            # `local:<checkpoint>`; strip any accidental provider prefix on model_id.
            name = self.model_id.split(":", 1)[-1]
            key = (self.api_key or "").strip()
            if not key or key.startswith("${"):
                key = os.environ.get("VLLM_API_KEY", "token-abc123").strip()
            return ResolvedModel(
                label=label,
                model_id=f"local:{name}",
                api_base=base_url,
                api_key=key,
                extra_body=(
                    dict(VLLM_HARMONY_EXTRA_BODY)
                    if self.extra_body is None
                    else self.extra_body
                ),
            )
        # Hosted route: `provider:snapshot`, credentials resolved from env by LiteLLM.
        return ResolvedModel(
            label=label,
            model_id=self.model_id,
            extra_body=self.extra_body,
        )


class InferenceConfig(BaseModel):
    model_name: str
    data_dir: str = "./data"
    dataset: str
    output_dir: str = "./runs"
    run_id: str
    kind: str = "headline"
    sample_n: Optional[int] = None
    sample_seed: int = 0
    n_trials: int = 1
    concurrency: int = 3
    thinking: str = "off"
    # Sampling temperature sent with every request. `None` sends NO temperature field.
    #
    # This exists because the README claimed `temperature=0` while nothing in the
    # harness ever sent one. For hosted routes that is harmless — each vendor
    # applies its own default — but a local vLLM route falls back to **1.0**, so
    # every past 50-item single-trial A/B was measuring the sampler at least as much
    # as the weights. Two arms of the same checkpoint could differ by more than the
    # effect being looked for.
    #
    # The default stays `None` rather than becoming 0.0: that is what every already
    # recorded run did, and silently changing it would make new numbers
    # incomparable with the ones on disk. Set it explicitly (`--temperature 0`, or
    # `temperature: 0` in the config) for any comparison that needs to be
    # reproducible — and note `0` must survive as 0.0 rather than being swallowed as
    # falsy, which is what `is not None` checks downstream are protecting.
    temperature: Optional[float] = None
    max_steps: int = 50
    max_output_tokens: int = 65536
    token_budget: Optional[int] = None
    resume: bool = True


class EvaluationConfig(BaseModel):
    model_name: str
    data_dir: str = "./data"
    dataset: str
    output_dir: str = "./runs"
    run_id: str
    judges: list[str] = [
        "openrouter:google/gemini-3.1-pro-preview",
        "openrouter:anthropic/claude-opus-4.7",
    ]
    grade_concurrency: int = 2
    resume: bool = True


class Config(BaseModel):
    inference: InferenceConfig
    evaluation: EvaluationConfig
    llms: dict[str, LLMSpec]

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "Config":
        with open(yaml_path, "r") as f:
            raw = _expand_env(yaml.safe_load(f))
        return cls(
            inference=InferenceConfig(**raw["inference"]),
            evaluation=EvaluationConfig(**raw["evaluation"]),
            llms={k: LLMSpec(**v) for k, v in raw["llms"].items()},
        )

    def resolve_model(self, model_name: str) -> ResolvedModel:
        if model_name not in self.llms:
            raise KeyError(
                f"model '{model_name}' is not in the llms: catalog "
                f"(have: {sorted(self.llms)})"
            )
        return self.llms[model_name].resolve(model_name)

    def dataset_path(self, section: InferenceConfig | EvaluationConfig) -> Path:
        """Resolve `<data_dir>/<dataset>.jsonl`, or `<dataset>` if it already points
        at an existing file (an explicit path in the config)."""
        direct = Path(section.dataset)
        if direct.suffix and direct.exists():
            return direct
        return Path(section.data_dir) / f"{section.dataset}.jsonl"
