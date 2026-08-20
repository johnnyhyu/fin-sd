"""Runtime contamination guard for the agent's retrieval surface.

The benchmark ships a public 50-item subset to Hugging Face and mirrors the paper on
arXiv. Both carry the *reference answers and the rubric text*. An agent with a live web
search will find them: in a 3-question smoke run, `gpt-oss-20b` searched for the question
text, got `RogoAI/big-finance-benchmark` back as the top organic result, fetched the
dataset viewer, and read rubric lines like `{"text": "Identify PSN has NCI", "points": 1}`
straight out of the page. The judge caught it — "it merely retrieved the benchmark dataset
containing the rubric" — and scored the final answer *correct* with zero rubric points.

That is not a measurement of financial research ability, so we exclude the benchmark's own
publication surface from both tools:

  - `web_search` drops matching results before the model ever sees them, and reports how
    many were dropped so the exclusion is visible in the trace.
  - `fetch_url` refuses matching URLs with an explanatory `ToolError`.

Scope is deliberately narrow — only the benchmark's own artifacts, matched on
distinctive tokens (`big-finance-benchmark`, `bigfinancebench`, the arXiv id). Primary
sources, aggregators, and financial-data sites are untouched. Set
`BFH_ALLOW_BENCHMARK_SOURCES=1` to disable the guard (for debugging the guard itself;
never for a scored run).
"""

from __future__ import annotations

import os
import re
from urllib.parse import unquote

# URLs that serve the dataset, this harness repo, or the paper. Matched against the
# lowercased URL. Hugging Face is reachable under several hostnames (`huggingface.co`,
# `hf.co`, `datasets-server.huggingface.co`, `cdn-lfs*.hf.co`), so we match on the
# repo path rather than enumerating hosts.
_LEAK_URL_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Hugging Face dataset repo, any host/route (viewer, raw, resolve, api, parquet).
    re.compile(r"rogoai/big-finance-benchmark"),
    re.compile(r"big-finance-benchmark.*\.(?:jsonl|parquet|csv)"),
    # GitHub source repo and its raw/API mirrors.
    re.compile(r"rogo-technologies/big-finance-benchmark"),
    # The paper, on arXiv and the usual mirrors that reproduce its full text. No
    # trailing word boundary — arXiv appends a version suffix (`/pdf/2606.03829v1`).
    re.compile(r"2606\.03829"),
    re.compile(r"bigfinancebench"),
)

# Search-result text that identifies a hit as the benchmark itself even when the URL is a
# mirror we do not enumerate. Matched against `title + snippet`, lowercased. Kept to
# distinctive multi-word tokens so an ordinary filing or news article cannot trip it.
_LEAK_TEXT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"rogoai/big-finance"),
    re.compile(r"big-finance-benchmark"),
    re.compile(r"bigfinancebench"),
    re.compile(r"\b2606\.03829\b"),
)

_ENV_DISABLE = "BFH_ALLOW_BENCHMARK_SOURCES"

LEAK_FETCH_MESSAGE = (
    "fetch_url refuses {url!r}: it is part of this benchmark's own published material "
    "(dataset, harness repo, or paper), which contains the reference answers and rubric. "
    "Answer from primary sources instead — SEC filings, company press releases, and "
    "investor-relations pages."
)


def guard_enabled() -> bool:
    return not os.environ.get(_ENV_DISABLE)


def is_benchmark_source(url: str) -> bool:
    """True if `url` points at the benchmark's own dataset, repo, or paper."""
    if not guard_enabled():
        return False
    # Unquote first: the Hugging Face datasets-server API addresses the repo as a query
    # parameter with the slash percent-encoded
    # (`?dataset=RogoAI%2Fbig-finance-benchmark`), which the raw patterns would miss.
    u = unquote(url or "").lower()
    return any(p.search(u) for p in _LEAK_URL_PATTERNS)


def is_benchmark_result(result: dict[str, str]) -> bool:
    """True if a `{title, url, snippet}` search result is the benchmark itself."""
    if not guard_enabled():
        return False
    if is_benchmark_source(result.get("url", "")):
        return True
    blob = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
    return any(p.search(blob) for p in _LEAK_TEXT_PATTERNS)


def filter_search_results(
    results: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    """Drop benchmark-self results. Returns `(kept, n_dropped)`."""
    kept = [r for r in results if not is_benchmark_result(r)]
    return kept, len(results) - len(kept)
