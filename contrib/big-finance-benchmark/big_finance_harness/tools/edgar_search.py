from __future__ import annotations

import json
import os
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from big_finance_harness.tools.base import Tool, ToolError, require_str

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
DEFAULT_TIMEOUT_S = 20.0
DEFAULT_LIMIT = 20

# Cached at module load; the SEC ticker file is small (~1MB) and stable enough that
# refetching once per process is fine.
_TICKER_CACHE: dict[str, str] | None = None


class EdgarSearchTool(Tool):
    name = "edgar_search"
    description = (
        "List recent SEC EDGAR filings for a public company by ticker. Returns a list of "
        "filings with form type (10-K, 10-Q, 8-K, etc.), filing date, accession number, "
        "and the URL of the primary document. Use `fetch_url` to read a filing's "
        "contents."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "U.S. listed-equity ticker symbol (e.g. AAPL, MSFT).",
            },
            "form_type": {
                "type": "string",
                "description": ("Optional form filter (e.g. '10-K', '10-Q', '8-K', 'DEF 14A')."),
            },
            "limit": {
                "type": "integer",
                "description": "Maximum filings to return. Default 20.",
                "minimum": 1,
                "maximum": 50,
            },
        },
        "required": ["ticker"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        user_agent: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        # The User-Agent is resolved lazily, matching `WebSearchTool`'s deferred backend
        # selection. Raising in `__init__` made `default_tools()` — and therefore
        # `inference.py`'s manifest emission and `run_eval_set.py`'s startup — die with a
        # bare ValueError before either could report anything useful, purely because an
        # env var was unset. The requirement itself is unchanged: `run()` still refuses
        # to call SEC without contact info.
        self._user_agent_override = user_agent
        self.timeout_s = timeout_s

    @property
    def user_agent(self) -> str:
        ua = self._user_agent_override or os.environ.get("SEC_EDGAR_USER_AGENT")
        if not ua:
            raise ToolError(
                "SEC EDGAR requires a User-Agent header with contact info. "
                "Set SEC_EDGAR_USER_AGENT='Your Name your@email.com' in the "
                "environment (or the repo-root .env)."
            )
        return ua

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(httpx.HTTPError),
        reraise=True,
    )
    async def _get(self, url: str) -> httpx.Response:
        async with httpx.AsyncClient(
            timeout=self.timeout_s,
            headers={"User-Agent": self.user_agent, "Accept": "application/json"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp

    async def _ticker_to_cik(self, ticker: str) -> str:
        global _TICKER_CACHE
        if _TICKER_CACHE is None:
            resp = await self._get(TICKERS_URL)
            raw: dict[str, dict[str, Any]] = resp.json()
            _TICKER_CACHE = {row["ticker"].upper(): str(row["cik_str"]) for row in raw.values()}
        cik = _TICKER_CACHE.get(ticker.upper())
        if not cik:
            # `company_tickers.json` indexes *currently registered* filers only. Traces
            # show this miss on delisted or renamed issuers (R1, 2U, QVC, CTLP, MARK) —
            # cases where the filings do exist on EDGAR under a CIK the ticker no longer
            # maps to. Say so, so the model routes around it instead of retrying the
            # same lookup.
            raise ToolError(
                f"unknown ticker: {ticker}. This lookup covers currently-registered SEC "
                "filers by ticker only; delisted, renamed, or acquired issuers will not "
                "resolve. Try `web_search` for the company's CIK or former ticker, then "
                "`fetch_url` on its EDGAR filing index at "
                "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=<cik>&type=<form>"
            )
        return cik

    async def run(self, args: dict[str, Any]) -> str:
        ticker = require_str(args, "ticker", 'expected {"ticker": "AAPL"}')
        form_type = args.get("form_type")
        try:
            limit = int(args.get("limit") or DEFAULT_LIMIT)
        except (TypeError, ValueError):
            limit = DEFAULT_LIMIT
        # Clamp rather than reject: an out-of-range `limit` is not worth a wasted step.
        limit = max(1, min(limit, 50))

        try:
            cik = await self._ticker_to_cik(ticker)
            resp = await self._get(SUBMISSIONS_URL.format(cik=cik))
        except httpx.HTTPError as e:
            raise ToolError(f"edgar_search failed: {e}") from e

        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        out: list[dict[str, Any]] = []
        for form, date, acc, doc in zip(forms, dates, accessions, primary_docs):
            if form_type and form.upper() != form_type.upper():
                continue
            acc_clean = acc.replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{doc}"
            out.append(
                {
                    "form": form,
                    "filing_date": date,
                    "accession": acc,
                    "primary_document_url": url,
                }
            )
            if len(out) >= limit:
                break

        return json.dumps(
            {
                "ticker": ticker.upper(),
                "cik": cik,
                "filings": out,
            },
            ensure_ascii=False,
        )
