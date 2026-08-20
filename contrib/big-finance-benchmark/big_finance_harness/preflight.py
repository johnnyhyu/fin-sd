"""Startup checks that turn silent, expensive misconfiguration into a fast failure.

The failure this exists to prevent actually happened, and is preserved in
`runs/headline/`: a 150-trace run of `gpt-oss-20b` completed in 533s and reported
`"n_errors": 0` in its manifest. Every one of the 150 traces was worthless — `web_search`
had raised `ValueError: web_search requires either SERP_API_KEY ... in the environment`
193 times, 148 traces ended with no answer at all, and the grade phase then spent real
judge tokens scoring 0.0% final-answer accuracy. Nothing in the pipeline said anything
was wrong until a human read the traces.

Two checks, run before any model is called:

  - `check_environment` — credentials for every route in the run, plus the tool keys the
    agent needs. Missing tool keys are the dangerous case, because the run *succeeds*
    without them.
  - `probe_local_route` — for self-hosted vLLM routes, confirm the server is up and
    actually serving the model name the config asks for. A typo'd
    `--served-model-name` otherwise surfaces as a 404 on every question.

Set `BFH_SKIP_PREFLIGHT=1` to bypass (e.g. offline replay of an existing run).
"""

from __future__ import annotations

import os

import httpx

from big_finance_harness.config import ResolvedModel

# Env var each hosted provider prefix needs. `local:` is handled separately — its
# credentials come from the config entry, not the environment.
_PROVIDER_KEYS: dict[str, tuple[str, ...]] = {
    "openrouter": ("OPENROUTER_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "gateway": ("VERCEL_AI_GATEWAY_API_KEY",),
    "vertex": ("VERTEXAI_PROJECT", "GOOGLE_CLOUD_PROJECT"),
    "vertex-anthropic": ("VERTEXAI_PROJECT", "GOOGLE_CLOUD_PROJECT"),
}


def _provider_of(model_id: str) -> str:
    return model_id.split(":", 1)[0] if ":" in model_id else ""


def check_environment(
    model_ids: list[str],
    judge_ids: list[str] | None = None,
    *,
    require_tool_keys: bool = True,
) -> list[str]:
    """Return human-readable problems with the current environment. Empty == ready."""
    problems: list[str] = []

    for model_id in sorted(set(model_ids) | set(judge_ids or [])):
        provider = _provider_of(model_id)
        candidates = _PROVIDER_KEYS.get(provider)
        if not candidates:
            continue
        if not any(os.environ.get(name) for name in candidates):
            problems.append(
                f"{model_id}: needs {' or '.join(candidates)} in the environment "
                "(the repo-root .env is loaded automatically)"
            )

    if require_tool_keys:
        if not (os.environ.get("SERP_API_KEY") or os.environ.get("TAVILY_API_KEY")):
            problems.append(
                "web_search: needs SERP_API_KEY (SerpAPI) or TAVILY_API_KEY (Tavily). "
                "Without one, every web_search call errors and the agent answers from "
                "memory or not at all — the run completes but the numbers are void."
            )
        if not os.environ.get("SEC_EDGAR_USER_AGENT"):
            problems.append(
                "edgar_search / fetch_url on sec.gov: needs "
                "SEC_EDGAR_USER_AGENT='Your Name your@email.com'. SEC blocks requests "
                "without contact info, so filing retrieval fails on every question."
            )

    return problems


def probe_local_route(model: ResolvedModel, timeout_s: float = 10.0) -> list[str]:
    """Check that a `local:` route's server is up and serving the configured model."""
    if not model.api_base:
        return []
    served_name = model.model_id.split(":", 1)[-1]
    url = model.api_base.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {model.api_key}"} if model.api_key else {}
    try:
        resp = httpx.get(url, headers=headers, timeout=timeout_s)
        resp.raise_for_status()
        served = [m.get("id") for m in (resp.json().get("data") or [])]
    except Exception as e:  # noqa: BLE001 — any failure to reach the server is fatal
        return [
            f"{model.label}: cannot reach the local server at {model.api_base} "
            f"({type(e).__name__}: {e}). Start it with `vllm serve ... "
            "--enable-auto-tool-choice --tool-call-parser openai`."
        ]
    if served_name not in served:
        return [
            f"{model.label}: server at {model.api_base} is up but does not serve "
            f"{served_name!r} (it serves: {served or 'nothing'}). `model_id` in the "
            "config must equal the server's --served-model-name."
        ]
    return []


def run_preflight(
    models: list[ResolvedModel],
    judge_ids: list[str] | None = None,
    *,
    require_tool_keys: bool = True,
) -> list[str]:
    """Full preflight for a set of resolved models. Returns problems; empty == ready."""
    if os.environ.get("BFH_SKIP_PREFLIGHT"):
        return []
    problems = check_environment(
        [m.model_id for m in models], judge_ids, require_tool_keys=require_tool_keys
    )
    for model in models:
        problems.extend(probe_local_route(model))
    return problems


def assert_ready(
    models: list[ResolvedModel],
    judge_ids: list[str] | None = None,
    *,
    require_tool_keys: bool = True,
) -> None:
    """Run preflight and abort with an actionable message if anything is missing."""
    problems = run_preflight(models, judge_ids, require_tool_keys=require_tool_keys)
    if not problems:
        return
    lines = "\n".join(f"  - {p}" for p in problems)
    raise SystemExit(
        "preflight failed — refusing to start a run that would produce invalid "
        f"results:\n{lines}\n\nFix the above, or set BFH_SKIP_PREFLIGHT=1 to override."
    )
