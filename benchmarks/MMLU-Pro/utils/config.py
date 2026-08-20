from typing import Optional, Literal
from pydantic import BaseModel, field_validator, model_validator, computed_field
from .prompts import MODEL_PROMPT_DICT
import yaml
import os
import sys
import re

# A string that is *nothing but* a single ${VAR} / $VAR placeholder. Only these
# are expanded, so arbitrary config text containing a literal '$' (prompts, stop
# tokens, ...) is left untouched.
_ENV_PLACEHOLDER = re.compile(r"^\$\{?\w+\}?$")

# Captured BEFORE .env is loaded, so it means "the operator set this for this
# command" and not "the repo-root .env has a default". Only the former is a real
# choice of server; the latter is the stale pre-shift port that made an out-of-band
# benchmark talk to somebody else's model. See _resolve_routing.
_EXPLICIT_VLLM_BASE_URL = (os.environ.get("VLLM_BASE_URL") or "").strip()

# Load the unified repo-root .env (two levels up: MMLU-Pro/utils/ -> repo root)
# so OPENROUTER_API_KEY / VLLM_API_KEY are available when configs resolve their
# `${...}` placeholders below.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    load_dotenv()  # also honor a .env in the current working directory
except Exception:
    pass

# Endpoint discovery lives in the pipeline package at the repo root. Optional:
# this harness must still run standalone, where nothing publishes endpoints and
# the configured URL is the only answer there is.
try:
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from pipeline import endpoint as _endpoint
except Exception:
    _endpoint = None


def _expand_env(obj):
    """Recursively expand ${VAR} / $VAR placeholders in string values using the
    process environment (populated from the repo-root .env above). Lets config
    YAML reference secrets by name instead of hard-coding them. Only strings that
    are exactly a placeholder are expanded; other text is returned verbatim so a
    literal '$' in prompts/stop tokens is never mangled."""
    if isinstance(obj, str):
        return os.path.expandvars(obj) if _ENV_PLACEHOLDER.match(obj) else obj
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj


class PromptConfig(BaseModel):
    prompt_type: str

    @field_validator("prompt_type")
    def validate_prompt_type(cls, v):
        if v not in MODEL_PROMPT_DICT.keys():
            raise ValueError(f"Invalid prompt type: {v}")
        return v

    @computed_field(return_type=dict)
    def template(self):
        return MODEL_PROMPT_DICT[self.prompt_type]


class LLMConfig(BaseModel):
    model_id: str
    support_system_role: bool
    reasoner: bool
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    sampling_args: dict = {}
    max_retries: int = 3
    # Hard per-request wall-clock cap (seconds). A single hung upstream request
    # (common with reasoning models that never close the stream) would otherwise
    # block the whole asyncio.gather forever — the progress bar freezes one short
    # of the total. On timeout the request is cancelled and retried like any other
    # error. Generous enough for a full max_tokens reasoning completion.
    request_timeout: float = 900.0
    # All hosted calls route through OpenRouter (openai-compatible API); local
    # self-hosted calls route through vLLM (also openai-compatible). Only the
    # 'openai' client style is supported.
    api_style: Literal['openai'] = 'openai'
    rpm: int = 60  # requests per minute
    max_concurrency: int = 60  # max simultaneous in-flight requests

    @model_validator(mode="after")
    def _resolve_routing(self):
        """Enforce that every LLM is served by OpenRouter or a local vLLM
        endpoint, and fill the OpenRouter/vLLM api_key from the environment when
        the YAML leaves it as an unexpanded placeholder or blank."""
        url = (self.base_url or "").strip()
        if not url:
            raise ValueError(
                f"LLM '{self.model_id}' has no base_url. Point it at OpenRouter "
                "(https://openrouter.ai/api/v1) or a local vLLM server."
            )
        is_openrouter = "openrouter.ai" in url
        is_local_vllm = any(h in url for h in ("localhost", "127.0.0.1", "0.0.0.0"))
        if not (is_openrouter or is_local_vllm):
            raise ValueError(
                f"LLM '{self.model_id}' base_url '{url}' is not allowed. All calls "
                "must go through OpenRouter or a local vLLM endpoint."
            )
        # A local vLLM route is resolved against the server that is actually live,
        # so a run always talks to ITS OWN model. The YAML pins localhost:8000, but
        # the pipeline moves each concurrent instance to a private port AFTER launch
        # (vllm_server._assign_instance_port) — so the configured port is a request,
        # not a fact, and this file cannot know the answer. Measured: 4 parallel runs
        # on 8000-8003 all sent their requests to 8000, so three of them benchmarked
        # the FIRST run's model and wrote the answers out under their own model's
        # name. That is the 20B/120B result mix-up: nothing errors, the numbers are
        # just someone else's.
        #
        # $VLLM_BASE_URL set for this command wins. Otherwise the live endpoint is
        # looked up (pipeline.endpoint), which is what repairs the shift for a
        # benchmark launched out of band — inheriting the pipeline's environment is
        # not something an operator in a second shell can do. Several live servers
        # and no explicit choice raises rather than guessing.
        if is_local_vllm and _endpoint is not None:
            url = self.base_url = _endpoint.resolve(url, _EXPLICIT_VLLM_BASE_URL)

        # An unexpanded "${VAR}" (env var missing) or blank key -> pull from env.
        key = (self.api_key or "").strip()
        if not key or key.startswith("${"):
            if is_openrouter:
                key = os.environ.get("OPENROUTER_API_KEY", "").strip()
                if not key:
                    raise ValueError(
                        f"LLM '{self.model_id}' routes through OpenRouter but "
                        "OPENROUTER_API_KEY is unset (set it in the repo-root .env)."
                    )
            else:  # local vLLM
                key = os.environ.get("VLLM_API_KEY", "token-abc123").strip()
            self.api_key = key
        return self


class SamplingEvalConfig(BaseModel):
    """Multi-sample evaluation: draw `num_samples` completions per question at
    (`temperature`, `top_p`) and report avg@N ± CI, pass@k, and mean per-token
    entropy. When disabled the pipeline behaves exactly as before (single greedy
    completion per question)."""

    enabled: bool = False
    num_samples: int = 16
    pass_at_k: list[int] = [1, 4, 8, 16]
    temperature: float = 0.7
    top_p: float = 0.95
    ci_confidence: float = 0.95  # two-sided confidence level for the ± interval
    compute_entropy: bool = True
    top_logprobs: int = 20  # OpenAI/vLLM cap the returned top_logprobs at 20

    @field_validator("pass_at_k")
    def validate_pass_at_k(cls, v):
        if any(k <= 0 for k in v):
            raise ValueError("pass_at_k values must be positive")
        return v


class InferenceConfig(BaseModel):
    model_name: str
    llms: dict[str, LLMConfig]
    data_dir: str = "./data"
    output_dir: str = "./results"
    dataset: str = "MMLU-Pro"
    subset: str  # a category name (e.g. "law") or "all"
    prompt: PromptConfig
    num_shots: int = 5  # few-shot CoT exemplars drawn from the validation split
    sampling_eval: Optional[SamplingEvalConfig] = None

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, "r") as f:
            config_dict = _expand_env(yaml.safe_load(f))
        llms = {}
        for key, value in config_dict["llms"].items():
            llms[key] = LLMConfig(**value)
        config_dict['inference']['llms'] = llms
        if config_dict.get("sampling_eval") is not None:
            config_dict['inference']['sampling_eval'] = config_dict["sampling_eval"]
        config = cls(**config_dict["inference"])
        return config

    @computed_field(return_type=str)
    def save_path(self):
        return os.path.join(
            self.output_dir,
            self.dataset,
            self.subset,
            self.prompt.prompt_type,
            self.model_name,
        )

    @computed_field(return_type=str)
    def data_file(self):
        return os.path.join(
            self.data_dir,
            self.dataset,
            f"{self.subset}.json",
        )

    @computed_field(return_type=str)
    def shots_file(self):
        # Few-shot exemplars (grouped by category) prepared from the validation
        # split by prepare_data.py.
        return os.path.join(
            self.data_dir,
            self.dataset,
            "validation.json",
        )


class EvaluationConfig(BaseModel):
    result_dir: str = './results'
    model_name: str
    dataset: str = "MMLU-Pro"
    subset: str
    prompt_type: str
    llms: dict[str, LLMConfig]
    sampling_eval: Optional[SamplingEvalConfig] = None

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, "r") as f:
            config_dict = _expand_env(yaml.safe_load(f))
        llms = {}
        for key, value in config_dict["llms"].items():
            llms[key] = LLMConfig(**value)
        config_dict["evaluation"]["llms"] = llms
        if config_dict.get("sampling_eval") is not None:
            config_dict["evaluation"]["sampling_eval"] = config_dict["sampling_eval"]
        config = cls(**config_dict["evaluation"])
        return config

    @field_validator("prompt_type")
    def validate_prompt_type(cls, v):
        if v not in MODEL_PROMPT_DICT.keys():
            raise ValueError(f"Invalid prompt type: {v}")
        return v

    @computed_field(return_type=str)
    def save_path(self):
        return os.path.join(
            self.result_dir,
            self.dataset,
            self.subset,
            self.prompt_type,
            self.model_name,
        )

    @computed_field(return_type=str)
    def evaluation_file(self):
        return os.path.join(
            self.save_path,
            "evaluation.json",
        )

    @computed_field(return_type=str)
    def inference_file(self):
        return os.path.join(
            self.save_path,
            "inference.json",
        )
