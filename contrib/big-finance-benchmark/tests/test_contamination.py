"""The retrieval surface must not hand the model this benchmark's own answer key.

Regression tests for the leak observed in a live gpt-oss-20b run: a `web_search` for the
question text returned `RogoAI/big-finance-benchmark` on Hugging Face as the top organic
result, and `fetch_url` on the dataset viewer returned the rubric verbatim.
"""

from __future__ import annotations

import json

import pytest
from pytest_httpx import HTTPXMock

from big_finance_harness.contamination import (
    filter_search_results,
    is_benchmark_result,
    is_benchmark_source,
)
from big_finance_harness.tools.base import ToolError
from big_finance_harness.tools.fetch_url import FetchUrlTool
from big_finance_harness.tools.web_search import WebSearchTool, _SerpApiBackend

LEAKY_URLS = [
    "https://huggingface.co/datasets/RogoAI/big-finance-benchmark/viewer/",
    "https://huggingface.co/datasets/RogoAI/big-finance-benchmark/resolve/main/x.jsonl",
    "https://hf.co/datasets/RogoAI/big-finance-benchmark",
    "https://datasets-server.huggingface.co/rows?dataset=RogoAI%2Fbig-finance-benchmark",
    "https://github.com/Rogo-Technologies/big-finance-benchmark/blob/main/data/x.jsonl",
    "https://raw.githubusercontent.com/Rogo-Technologies/big-finance-benchmark/main/d.jsonl",
    "https://arxiv.org/abs/2606.03829",
    "https://arxiv.org/pdf/2606.03829v1",
]

SAFE_URLS = [
    "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm",
    "https://investor.shakeshack.com/news/press-releases/detail/1/q4-2024",
    "https://huggingface.co/datasets/openai/gsm8k",
    "https://arxiv.org/abs/2401.00001",
    "https://finance.yahoo.com/quote/AAPL/",
]


@pytest.mark.parametrize("url", LEAKY_URLS)
def test_benchmark_urls_are_recognized(url):
    assert is_benchmark_source(url) is True


@pytest.mark.parametrize("url", SAFE_URLS)
def test_ordinary_sources_are_not_blocked(url):
    assert is_benchmark_source(url) is False


def test_leak_detected_from_result_text_even_on_an_unknown_mirror():
    """URL matching alone misses mirrors, so title/snippet are checked too."""
    assert is_benchmark_result(
        {
            "title": "RogoAI/big-finance-benchmark - Datasets at Hugging Face",
            "url": "https://some-unknown-mirror.example/x",
            "snippet": "50 rows, public_release split",
        }
    )


def test_ordinary_finance_result_survives_the_text_check():
    assert not is_benchmark_result(
        {
            "title": "Shake Shack Announces Fourth Quarter and Fiscal Year 2024",
            "url": "https://investor.shakeshack.com/news",
            "snippet": "Total revenue of $1,252.6 million, up 15.2%.",
        }
    )


def test_filter_reports_how_many_were_dropped():
    kept, dropped = filter_search_results(
        [
            {"title": "10-K", "url": "https://www.sec.gov/x", "snippet": "revenue"},
            {"title": "ds", "url": LEAKY_URLS[0], "snippet": "rubric"},
        ]
    )
    assert dropped == 1
    assert len(kept) == 1
    assert kept[0]["url"] == "https://www.sec.gov/x"


def test_guard_can_be_disabled_for_debugging(monkeypatch):
    monkeypatch.setenv("BFH_ALLOW_BENCHMARK_SOURCES", "1")
    assert is_benchmark_source(LEAKY_URLS[0]) is False


@pytest.mark.asyncio
async def test_web_search_withholds_benchmark_results(httpx_mock: HTTPXMock):
    import re

    httpx_mock.add_response(
        url=re.compile(r"^https://serpapi\.com/search\.json"),
        method="GET",
        json={
            "organic_results": [
                {
                    "title": "RogoAI/big-finance-benchmark · Datasets at Hugging Face",
                    "link": LEAKY_URLS[0],
                    "snippet": "Identify PSN has NCI ...",
                },
                {
                    "title": "Parsons 10-K",
                    "link": "https://www.sec.gov/Archives/edgar/data/x.htm",
                    "snippet": "Non-controlling interest ...",
                },
            ]
        },
    )
    tool = WebSearchTool(backend=_SerpApiBackend("k", 30.0))
    parsed = json.loads(await tool.run({"query": "PSN NCI merger synergies"}))
    assert len(parsed["results"]) == 1
    assert parsed["results"][0]["title"] == "Parsons 10-K"
    assert parsed["excluded_benchmark_results"] == 1
    assert "reference answers" in parsed["note"]


@pytest.mark.asyncio
async def test_fetch_url_refuses_the_dataset():
    tool = FetchUrlTool()
    with pytest.raises(ToolError, match="benchmark's own published material"):
        await tool.run({"url": LEAKY_URLS[0]})


@pytest.mark.asyncio
async def test_fetch_url_refuses_a_redirect_into_the_dataset(httpx_mock: HTTPXMock):
    """A `resolve/` URL redirects to a CDN host; the same trick reaches the dataset via
    any open redirector, so every hop is checked, not just the first."""
    httpx_mock.add_response(
        url="https://example.com/go",
        status_code=302,
        headers={"location": LEAKY_URLS[1]},
    )
    tool = FetchUrlTool()
    with pytest.raises(ToolError, match="benchmark's own published material"):
        await tool.run({"url": "https://example.com/go"})
