import json

import pytest
from pytest_httpx import HTTPXMock

from big_finance_harness.tools.edgar_search import (
    SUBMISSIONS_URL,
    TICKERS_URL,
    EdgarSearchTool,
)
import big_finance_harness.tools.edgar_search as edgar_module


@pytest.fixture(autouse=True)
def _reset_ticker_cache():
    edgar_module._TICKER_CACHE = None
    yield
    edgar_module._TICKER_CACHE = None


@pytest.mark.asyncio
async def test_edgar_search_filters_by_form_type(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url=TICKERS_URL,
        json={
            "0": {"ticker": "AAPL", "cik_str": 320193, "title": "Apple Inc."},
            "1": {"ticker": "MSFT", "cik_str": 789019, "title": "Microsoft Corp."},
        },
    )
    httpx_mock.add_response(
        url=SUBMISSIONS_URL.format(cik="320193"),
        json={
            "filings": {
                "recent": {
                    "form": ["10-K", "10-Q", "8-K", "10-K"],
                    "filingDate": [
                        "2023-11-03",
                        "2023-08-04",
                        "2023-05-04",
                        "2022-10-28",
                    ],
                    "accessionNumber": [
                        "0000320193-23-000106",
                        "0000320193-23-000077",
                        "0000320193-23-000064",
                        "0000320193-22-000108",
                    ],
                    "primaryDocument": [
                        "aapl-20230930.htm",
                        "aapl-20230701.htm",
                        "aapl-20230401.htm",
                        "aapl-20220924.htm",
                    ],
                }
            }
        },
    )
    tool = EdgarSearchTool()
    raw = await tool.run({"ticker": "AAPL", "form_type": "10-K"})
    parsed = json.loads(raw)
    assert parsed["ticker"] == "AAPL"
    assert parsed["cik"] == "320193"
    assert all(f["form"] == "10-K" for f in parsed["filings"])
    assert len(parsed["filings"]) == 2
    assert (
        parsed["filings"][0]["primary_document_url"]
        == "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
    )


@pytest.mark.asyncio
async def test_edgar_search_unknown_ticker_raises(httpx_mock: HTTPXMock):
    from big_finance_harness.tools.base import ToolError

    httpx_mock.add_response(
        url=TICKERS_URL,
        json={"0": {"ticker": "AAPL", "cik_str": 320193, "title": "Apple Inc."}},
    )
    tool = EdgarSearchTool()
    with pytest.raises(ToolError, match="unknown ticker"):
        await tool.run({"ticker": "ZZZZZZ"})


def test_edgar_tool_constructs_without_a_user_agent(monkeypatch):
    """`default_tools()` builds every tool. Raising here killed `inference.py` before it
    could emit a manifest or report anything useful, purely on an unset env var."""
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    tool = EdgarSearchTool()
    assert tool.name == "edgar_search"
    assert tool.spec.input_schema["required"] == ["ticker"]


@pytest.mark.asyncio
async def test_edgar_run_still_refuses_without_a_user_agent(monkeypatch):
    """Deferring the check must not weaken it — SEC requires contact info per request."""
    from big_finance_harness.tools.base import ToolError

    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    with pytest.raises(ToolError, match="SEC_EDGAR_USER_AGENT"):
        await EdgarSearchTool().run({"ticker": "AAPL"})


@pytest.mark.asyncio
async def test_unknown_ticker_error_tells_the_model_how_to_recover(httpx_mock: HTTPXMock):
    """Delisted and renamed issuers (R1, 2U, QVC, CTLP) miss this lookup even though
    their filings are on EDGAR. Traces show the model retrying the same call instead of
    routing around it."""
    from big_finance_harness.tools.base import ToolError

    httpx_mock.add_response(
        url=TICKERS_URL,
        json={"0": {"ticker": "AAPL", "cik_str": 320193, "title": "Apple Inc."}},
    )
    with pytest.raises(ToolError, match="delisted, renamed, or acquired"):
        await EdgarSearchTool().run({"ticker": "QVC"})


@pytest.mark.asyncio
async def test_out_of_range_limit_is_clamped_not_rejected(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url=TICKERS_URL,
        json={"0": {"ticker": "AAPL", "cik_str": 320193, "title": "Apple Inc."}},
    )
    httpx_mock.add_response(
        url=SUBMISSIONS_URL.format(cik="320193"),
        json={
            "filings": {
                "recent": {
                    "form": ["10-K", "10-Q"],
                    "filingDate": ["2023-11-03", "2023-08-04"],
                    "accessionNumber": ["0000320193-23-000106", "0000320193-23-000077"],
                    "primaryDocument": ["a.htm", "b.htm"],
                }
            }
        },
    )
    parsed = json.loads(await EdgarSearchTool().run({"ticker": "AAPL", "limit": 9999}))
    assert len(parsed["filings"]) == 2
