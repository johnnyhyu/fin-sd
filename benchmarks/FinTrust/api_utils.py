"""Shared helpers for the FinTrust api_call.py scripts."""
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load the repo-root .env before building clients. This runs at import time,
# which is *before* each api_call.py's own load_dotenv(), so keys are available
# when MODEL_CATALOG is constructed regardless of the (often missing) per-module
# .env path the scripts pass.
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# The run configuration (model catalog + run/postprocess defaults). Mirrors
# FinanceReasoning: a single config/config.yaml is the source of truth. inference.py
# / evaluation.py pass their resolved --config path to subprocesses via the
# FINTRUST_CONFIG env var so every process builds the same MODEL_CATALOG.
DEFAULT_CONFIG_PATH = Path(
    os.environ.get("FINTRUST_CONFIG") or (REPO_ROOT / "config" / "config.yaml")
)

# Global concurrency cap shared across every dataset when run through inference.py.
# One process, one asyncio.Semaphore(GLOBAL_PARALLEL) gates every API call, so the
# true number of in-flight requests never exceeds this regardless of how many
# datasets are running. config.yaml's run.global_parallel is the real control;
# this env-derived value is only the fallback default when config omits it.
GLOBAL_PARALLEL = int(os.environ.get("GLOBAL_PARALLEL", 32))

# A string that is *nothing but* a single ${VAR} / $VAR placeholder. Only these
# are expanded, so literal '$' in other config text is left untouched. Mirrors
# FinanceReasoning/utils/config.py.
_ENV_PLACEHOLDER = re.compile(r"^\$\{?\w+\}?$")


def _expand_env(obj):
    """Recursively expand ${VAR} / $VAR placeholders (only whole-string ones)
    using the process environment populated from the repo-root .env above."""
    if isinstance(obj, str):
        return os.path.expandvars(obj) if _ENV_PLACEHOLDER.match(obj) else obj
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj


def load_config(path=None):
    """Load and env-expand the run config YAML. Returns {} if the file is absent
    (callers then fall back to their built-in defaults)."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return _expand_env(yaml.safe_load(f) or {})


def retry_async(retries=10, initial_delay=1, backoff_factor=2,
                allowed_exceptions=(Exception,)):
    """Retry an async function with exponential backoff and jitter.

    Shared across all datasets so retry behavior (and the fix for
    deep-inception, which previously had no retry at all) stays consistent.
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            attempt = 0
            delay = initial_delay
            while attempt < retries:
                try:
                    return await func(*args, **kwargs)
                except allowed_exceptions as e:
                    attempt += 1
                    if attempt >= retries:
                        raise
                    sleep_time = delay * (backoff_factor ** (attempt - 1))
                    sleep_time = sleep_time * (1 + random.uniform(-0.1, 0.1))
                    print(f"Retry {attempt}/{retries} after error: {e}. "
                          f"Sleeping {sleep_time:.1f}s...")
                    await asyncio.sleep(sleep_time)
        return wrapper
    return decorator


def load_jsonl_or_json(file_path):
    """Load a dataset that may be either JSON or JSONL.

    Shared across the dataset api_call.py scripts, which previously each carried
    a byte-for-byte (or near-identical) copy of this. A `.jsonl` extension is
    read line-by-line; anything else is tried as a single JSON document first,
    then falls back to line-delimited parsing before raising a clear error.
    """
    with open(file_path, encoding="utf-8") as f:
        if file_path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        else:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                lines = [line for line in f if line.strip()]
                try:
                    return [json.loads(line) for line in lines]
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to parse file {file_path}. The file is neither valid JSON nor JSONL.\nError: {e}"
                    )


def standalone_semaphore():
    """Semaphore for a single-dataset (standalone) api_call.py run.

    Uses MAX_PARALLEL when set, otherwise falls back to GLOBAL_PARALLEL.
    """
    size = int(os.environ.get("MAX_PARALLEL", GLOBAL_PARALLEL))
    return asyncio.Semaphore(size)


# AsyncOpenAI clients are cached by (base_url, api_key) so every OpenRouter model
# shares one client (as before) while a local vLLM route gets its own.
_CLIENT_CACHE: dict[tuple[str, str], AsyncOpenAI] = {}


def _resolve_api_key(base_url, api_key):
    """Fill a blank/unexpanded api_key from the environment, matching the route.

    Mirrors FinanceReasoning/utils/config.py: OpenRouter routes require
    OPENROUTER_API_KEY (fail loudly if unset — an empty key otherwise surfaces as
    a misleading 'Connection error.' on every request); local vLLM routes fall
    back to VLLM_API_KEY (default 'token-abc123')."""
    key = (api_key or "").strip()
    is_openrouter = "openrouter.ai" in base_url
    if not key or key.startswith("${"):
        if is_openrouter:
            key = os.environ.get("OPENROUTER_API_KEY", "").strip()
            if not key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY is empty or unset. Set it in the repo-root "
                    ".env (OpenRouter routes require it)."
                )
        else:  # local vLLM or other openai-compatible endpoint
            key = os.environ.get("VLLM_API_KEY", "token-abc123").strip()
    return key


def _get_client(base_url, api_key):
    key = _resolve_api_key(base_url, api_key)
    cache_key = (base_url, key)
    client = _CLIENT_CACHE.get(cache_key)
    if client is None:
        client = _CLIENT_CACHE[cache_key] = AsyncOpenAI(api_key=key, base_url=base_url)
    return client


def build_catalog(llms):
    """Build a MODEL_CATALOG ({key: {display, client, model}}) from a config
    `llms:` mapping. base_url defaults to OpenRouter; each distinct (base_url,
    key) shares one AsyncOpenAI client. The 'model'/'client' keys match what the
    per-dataset api_call.py scripts consume, so nothing downstream changes."""
    catalog = {}
    for key, spec in llms.items():
        model_id = spec.get("model_id") or spec.get("model")
        if not model_id:
            raise ValueError(f"llms['{key}'] is missing 'model_id'.")
        base_url = (spec.get("base_url") or OPENROUTER_BASE_URL).strip()
        catalog[key] = {
            "display": spec.get("display", key),
            "client": _get_client(base_url, spec.get("api_key")),
            "model": model_id,
        }
    return catalog


# MODEL_CATALOG is populated from config/config.yaml (the source of truth). It is
# mutated in place — never rebound — so per-dataset modules that did
# `from api_utils import MODEL_CATALOG` at import time see the same live dict.
MODEL_CATALOG: dict[str, dict] = {}


def set_catalog_from_config(config):
    """Repopulate MODEL_CATALOG in place from a loaded config dict's `llms:`."""
    llms = (config or {}).get("llms")
    if not llms:
        raise RuntimeError(
            f"No 'llms:' catalog found in the run config ({DEFAULT_CONFIG_PATH}). "
            "Define your models there — it is the source of truth."
        )
    MODEL_CATALOG.clear()
    MODEL_CATALOG.update(build_catalog(llms))
    return MODEL_CATALOG


# Build the default catalog at import from config/config.yaml (or FINTRUST_CONFIG
# if the launcher pointed subprocesses at a different file). Errors (missing key
# or missing config) are deferred: a launcher that actually calls models
# (inference.py) rebuilds via set_catalog_from_config() and gets the loud error
# there, while tools that only import api_utils for its helpers (evaluation.py)
# don't need a valid catalog and shouldn't fail to import.
_CATALOG_INIT_ERROR = None
try:
    set_catalog_from_config(load_config())
except Exception as _e:  # noqa: BLE001
    _CATALOG_INIT_ERROR = _e


def contains_error(obj):
    """Recursively check whether a result record holds any 'ERROR:' string.

    Works uniformly across modules regardless of which key(s) hold the model
    response (item['response'], item['answer'], nested dicts, robustness' six
    keys, etc.).
    """
    if isinstance(obj, str):
        return obj.startswith("ERROR:")
    if isinstance(obj, dict):
        return any(contains_error(v) for v in obj.values())
    if isinstance(obj, list):
        return any(contains_error(v) for v in obj)
    return False


def report_failures(results, out_path, exit_on_failure=True):
    """Print an end-of-run summary and, by default, exit non-zero if any item
    failed after all retries.

    A failed item is any result that still contains an 'ERROR:' string anywhere
    in its structure. Without this, a run that hit rate limits / connection
    errors would still print a plain success message and exit 0, silently
    baking 'ERROR:' strings into the output JSON.
    """
    failed = [r for r in results if contains_error(r)]
    total = len(results)
    print(f"Results written to: {out_path}")
    if failed:
        print(
            f"⚠️  {len(failed)}/{total} items FAILED after all retries "
            f"(response contains 'ERROR:'). The output file includes those failed "
            f"responses; re-run just the failures with:\n"
            f"    MODEL_KEY=<key> python retry_errors.py <module_dir> {out_path}",
            file=sys.stderr,
        )
        if exit_on_failure:
            sys.exit(1)
    else:
        print(f"✅ All {total} items completed with no errors.")
    return failed
