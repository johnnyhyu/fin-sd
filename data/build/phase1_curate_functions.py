#!/usr/bin/env python3
"""
Phase 1 — curate the function library.

Chains four steps, turning the raw article functions into the filtered library
that later phases build problems around:

    functions-article-all.json
      → grade_functions      (LLM difficulty grade 1-8)  → functions-graded.json
      → filter_grade3plus    (keep grade >= 3)           → functions-hard.json
      → filter_grade5        (LLM viability judge)        → functions-hard-filtered.json
      → build_function_library (join back to source)     → function-library.json

Usage:
    OPENROUTER_API_KEY=sk-or-... python scripts/phase1_curate_functions.py
    python scripts/phase1_curate_functions.py --only build_function_library
    python scripts/phase1_curate_functions.py --grade-start-from function_042
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import aiohttp
from openai import OpenAI

# ── Shared config ────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
# This file lives in data/build/, so the corpus directory is its parent.
_DATA_DIR = Path(__file__).parent.parent

# The 3,133-function library released with FinanceReasoning (paper §4.1), read
# from the vendored benchmark rather than duplicated into data/. The harness sits
# under benchmarks/ in the released layout and at the top level in the original
# working repo, so accept either.
_FR_FUNCTIONS = Path("FinanceReasoning") / "data" / "functions" / "functions-article-all.json"
ARTICLE_ALL_FILE = next(
    (c for c in (_ROOT / "benchmarks" / _FR_FUNCTIONS, _ROOT / _FR_FUNCTIONS) if c.exists()),
    _ROOT / "benchmarks" / _FR_FUNCTIONS,
)
GRADED_FILE = _DATA_DIR / "functions-graded.json"
HARD_FILE = _DATA_DIR / "functions-hard.json"
HARD_FILTERED_FILE = _DATA_DIR / "functions-hard-filtered.json"
LIBRARY_FILE = _DATA_DIR / "function-library.json"

_env_file = _ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


def _id_key(function_id: str) -> int:
    m = re.search(r"\d+$", function_id)
    return int(m.group()) if m else 0


def _backoff(attempt: int, base: float = 2.0, cap: float = 60.0) -> float:
    """Exponential backoff with full jitter."""
    ceiling = min(cap, base * (2 ** (attempt - 1)))
    return random.uniform(0, ceiling)


def _require_api_key() -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        sys.exit("Error: OPENROUTER_API_KEY environment variable not set.")
    return api_key


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — grade_functions
# ─────────────────────────────────────────────────────────────────────────────
GRADE_MODEL = "openai/gpt-oss-120b"
GRADE_CONCURRENCY = 64
GRADE_RETRY_ATTEMPTS = 5

GRADE_SYSTEM_PROMPT = """
You are an expert test-writer. You will receive an isolated, annotated financial formula as input. Your task is to evaluate the hidden cognitive complexity of a typical, real-world application of this formula.

The Evaluation Protocol (Think First)
Before assigning a level, you must mentally execute these two steps:
1. Identify the Real-World Friction: What do the variables actually represent? Are they typically handed to a practitioner as explicit numbers, or do they require forecasting, historical data analysis, or market estimation?
2. Anchor to the Professional Standard: Do not grade based on a simplified textbook "plug-and-play" exercise. Grade based on the typical professional implementation of this formula in corporate finance, trading, or portfolio management.

Complexity Rubric

1: Plug-and-Play (Deterministic Inputs)
Criteria: The variables are almost always known, fixed, historical facts or explicit contract terms. There is zero estimation, forecasting, or latency required to populate the formula.
Typical Concepts: Simple interest, basic future value of a single cash flow, spot-rate currency conversion.

2: Standard Static Benchmarks
Criteria: Uses readily available, standardized market data or fixed constants. No complex extraction or variable shifting.
Typical Concepts: Basic CAPM (where beta, Rf, and Rm are standard index outputs), ordinary annuity formulas.

3: Operational & Horizon Alignments
Criteria: Formulas where the variables are easy to find, but structurally messy to align (e.g., structural mismatch between compounding periods and payment frequencies).
Typical Concepts: Annual Effective Rate conversions, basic yield-to-maturity approximations, loan modifications.

4: Binary Adjustments & Structural Wrinkles
Criteria: Simple-looking formulas that require a discrete behavioral shift or structural pivot based on timing, contract types, or day-count conventions.
Typical Concepts: Annuity due vs. Ordinary annuity, adjusting day-counts for leap years in fixed income, simple fractional bet-sizing with asymmetric payouts.

5: Latent Input Derivation (The Deceptive Formulas)
Criteria: The formula itself is brief/elegant, but the inputs are latent. To populate even a single variable, a practitioner must first build a separate predictive model, analyze historical distributions, or estimate hidden probabilities.
Typical Concepts: The Kelly Criterion (requires estimating the true probability of winning p), basic project NPV (requires forecasting multi-year free cash flows).

6: Path-Dependent & Iterative Workflows
Criteria: Applying the formula requires maintaining a state or recalculating inputs sequentially over multiple time horizons or shifting phases.
Typical Concepts: Multi-stage dividend discount models, dynamic asset allocation rebalancing loops, multi-period capital budgeting.

7: Non-Linear & Algorithmic Formulation
Criteria: The formula contains complex calculus, continuous distributions, or highly sensitive mathematical relationships where a tiny input variance drastically alters the output.
Typical Concepts: Black-Scholes options pricing (Greeks/Volatility surfaces), basic Markowitz portfolio optimization, complex variable-rate debt amortization.

8: Multi-Asset Constraints & Stochastic Systems
Criteria: The formula is part of a broader systemic calculation involving simultaneous optimization, correlation matrices, complex tax/regulatory logic, or high numeric precision.
Typical Concepts: Simultaneous multi-asset Kelly betting (dealing with co-dependencies), exact bond portfolio immunization matching, tracking error optimization under constraints.

Respond with ONLY a JSON object:
{"grade": <integer 1-8>, "reasoning": "<one sentence>"}
"""


def _grade_user_prompt(entry: dict) -> str:
    return (
        f'Article: {entry["article_title"]}\n\n'
        f'Function signature and docstring:\n```python\n{entry["function"]}\n```\n\n'
        "Grade how difficult it would be to solve a problem built around this function from the docstring alone."
    )


async def _grade_one(
    session: aiohttp.ClientSession,
    entry: dict,
    api_key: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://ctgt.ai",
    }
    payload = {
        "model": GRADE_MODEL,
        "messages": [
            {"role": "system", "content": GRADE_SYSTEM_PROMPT},
            {"role": "user", "content": _grade_user_prompt(entry)},
        ],
        "temperature": 0.1,
        "max_tokens": 120,
    }

    for attempt in range(1, GRADE_RETRY_ATTEMPTS + 1):
        sleep_for = 0.0
        try:
            async with semaphore:
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status in RETRYABLE_STATUSES:
                        retry_after = resp.headers.get("Retry-After")
                        sleep_for = float(retry_after) if retry_after else _backoff(attempt)
                        label = "rate limit" if resp.status == 429 else f"server error {resp.status}"
                        print(f"  [{label}] {entry['function_id']} — waiting {sleep_for:.1f}s (attempt {attempt})", flush=True)
                    else:
                        resp.raise_for_status()
                        data = await resp.json()
                        raw = data["choices"][0]["message"]["content"].strip()
                        m = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
                        parsed = json.loads(m.group() if m else raw)
                        return {
                            "id": _id_key(entry["function_id"]),
                            "function_id": entry["function_id"],
                            "article_title": entry["article_title"],
                            "grade": int(parsed["grade"]),
                            "reasoning": parsed.get("reasoning", ""),
                        }
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            if attempt == GRADE_RETRY_ATTEMPTS:
                print(f"  [parse error] {entry['function_id']}: {e}", flush=True)
                return {
                    "id": _id_key(entry["function_id"]),
                    "function_id": entry["function_id"],
                    "article_title": entry["article_title"],
                    "grade": -1,
                    "reasoning": f"parse error: {e}",
                }
            sleep_for = _backoff(attempt)
        except Exception as e:
            if attempt == GRADE_RETRY_ATTEMPTS:
                print(f"  [error] {entry['function_id']}: {e}", flush=True)
                return {
                    "id": _id_key(entry["function_id"]),
                    "function_id": entry["function_id"],
                    "article_title": entry["article_title"],
                    "grade": -1,
                    "reasoning": f"error: {e}",
                }
            sleep_for = _backoff(attempt)

        if sleep_for:
            await asyncio.sleep(sleep_for)


def _save_grades(results: list[dict]) -> None:
    sorted_results = sorted((r for r in results if r["grade"] != -1), key=lambda r: r["id"])
    with open(GRADED_FILE, "w") as f:
        json.dump(sorted_results, f, indent=2)


async def _grade_functions_async(start_from: str | None) -> None:
    api_key = _require_api_key()

    with open(ARTICLE_ALL_FILE) as f:
        all_entries = json.load(f)

    all_entries.sort(key=lambda e: _id_key(e["function_id"]))

    existing: dict[str, dict] = {}
    if GRADED_FILE.exists():
        with open(GRADED_FILE) as f:
            for row in json.load(f):
                if row.get("grade", -1) != -1:
                    if "id" not in row:
                        row["id"] = _id_key(row["function_id"])
                    existing[row["function_id"]] = row
        print(f"Resuming: {len(existing)} already graded, {len(all_entries) - len(existing)} remaining.")

    todo = sorted(
        (e for e in all_entries if e["function_id"] not in existing),
        key=lambda e: _id_key(e["function_id"]),
    )

    if start_from:
        start_n = _id_key(start_from)
        todo = [e for e in todo if _id_key(e["function_id"]) >= start_n]
        print(f"Starting from {start_from!r} (#{start_n}): {len(todo)} functions to grade.")

    print(f"Grading {len(todo)} functions with model={GRADE_MODEL}, concurrency={GRADE_CONCURRENCY}...")

    results: list[dict] = list(existing.values())
    completed = 0
    semaphore = asyncio.Semaphore(GRADE_CONCURRENCY)

    async with aiohttp.ClientSession() as session:
        tasks = [_grade_one(session, entry, api_key, semaphore) for entry in todo]

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            grade_str = str(result["grade"]) if result["grade"] != -1 else "ERR"
            print(f"  [{completed}/{len(todo)}] {result['function_id']} → grade={grade_str}", flush=True)

            if completed % 50 == 0:
                _save_grades(results)

    _save_grades(results)
    errors = sum(1 for r in results if r["grade"] == -1)
    print(f"\nDone. {len(results)} graded, {errors} errors. Output: {GRADED_FILE}")


def grade_functions(start_from: str | None = None) -> None:
    asyncio.run(_grade_functions_async(start_from))


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — filter_grade3plus
# ─────────────────────────────────────────────────────────────────────────────
def filter_grade3plus() -> None:
    with open(GRADED_FILE) as f:
        problems = json.load(f)

    filtered = [p for p in problems if p["grade"] >= 3]

    with open(HARD_FILE, "w") as f:
        json.dump(filtered, f, indent=2)

    print(f"Kept {len(filtered)} / {len(problems)} problems (grade >= 3) → {HARD_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — filter_grade5
# ─────────────────────────────────────────────────────────────────────────────
GRADE5_MODEL = "deepseek/deepseek-v4-flash"
GRADE5_SYSTEM_PROMPT = """
You are an expert academic evaluator specializing in university-level Finance and Corporate Valuation testing. Your task is to evaluate a specific financial formula or metric and grade its operational viability for a timed, closed-book exam.

Analyze the given financial formula/concept and determine if a student could realistically derive the inputs themselves during a test (assume a calculator and Excel are available).

### Grading Criteria

* **Return 0 (Complex / Unsuitable for Independent Input Generation):**
    * The inputs required for the formula cannot realistically be derived by hand or from a simple prompt table in a timely manner.
    * Determining the inputs requires subjective estimation, complex multi-stage modeling, curve construction (e.g., yield curves), or extensive forecasting.
    * *Examples:* Pre-money valuation (requires complex cap-tables or multi-year discounted cash flow forecasting), multi-asset portfolio optimization, or implied volatility.

* **Return 1 (Viable for Independent Input Generation):**
    * It is highly feasible to design a test problem where the student determines or extracts the inputs themselves in a few minutes.
    * Inputs can be derived from standard financial statement snippets, a simple historical data table, or basic algebraic isolation.
    * *Examples:* Weighted Average Cost of Capital (WACC), standard Option Pricing (where inputs like spot price, strike, and risk-free rate are easily provided), Market Share, Discount Rates, or Inflation adjustments (with CPI).

### Output Format
Return exactly a single digit: `0` or `1`. Do not include JSON formatting, markdown block indicators, spaces, or any explanatory text.
""".strip()


def filter_grade5() -> None:
    _require_api_key()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    with open(HARD_FILE) as f:
        items = json.load(f)

    grade5_items = [(i, item) for i, item in enumerate(items) if item.get("grade") == 5]
    grade5_total = len(grade5_items)
    lock = threading.Lock()
    keep_set: set[int] = set()
    counters = {"processed": 0, "kept": 0}

    def evaluate(idx, item):
        response = client.chat.completions.create(
            model=GRADE5_MODEL,
            messages=[
                {"role": "system", "content": GRADE5_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(item, indent=2)},
            ],
        )
        return idx, response.choices[0].message.content.strip() == "1"

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(evaluate, idx, item): idx for idx, item in grade5_items}
        for future in as_completed(futures):
            idx, keep = future.result()
            with lock:
                counters["processed"] += 1
                if keep:
                    counters["kept"] += 1
                    keep_set.add(idx)
                print(
                    f"[{counters['processed']}/{grade5_total}] kept={counters['kept']} "
                    f"filtered={counters['processed'] - counters['kept']}",
                    end="\r",
                    flush=True,
                )

    print()

    output = [item for i, item in enumerate(items) if item.get("grade") != 5 or i in keep_set]

    with open(HARD_FILTERED_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Done. {len(output)} items saved to {HARD_FILTERED_FILE.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — build_function_library
# ─────────────────────────────────────────────────────────────────────────────
def build_function_library() -> None:
    with open(HARD_FILTERED_FILE) as f:
        filtered = json.load(f)

    with open(ARTICLE_ALL_FILE) as f:
        all_functions = json.load(f)

    lookup = {item["function_id"]: item["function"] for item in all_functions}

    library = []
    for item in filtered:
        fid = item["function_id"]
        if fid in lookup:
            library.append({"function_id": fid, "function": lookup[fid]})
        else:
            print(f"Warning: {fid} not found in {ARTICLE_ALL_FILE.name}")

    with open(LIBRARY_FILE, "w") as f:
        json.dump(library, f, indent=2)

    print(f"Wrote {len(library)} entries to {LIBRARY_FILE.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase entry point
# ─────────────────────────────────────────────────────────────────────────────
STEPS = ["grade_functions", "filter_grade3plus", "filter_grade5", "build_function_library"]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Phase 1 — curate the function library.")
    parser.add_argument("--only", choices=STEPS, help="Run a single step instead of the whole phase.")
    parser.add_argument("--grade-start-from", metavar="FUNCTION_ID",
                        help="grade_functions: skip all function_ids that sort before this value.")
    args = parser.parse_args(argv)

    steps = [args.only] if args.only else STEPS
    for step in steps:
        print(f"\n=== Phase 1 · {step} ===", flush=True)
        if step == "grade_functions":
            grade_functions(args.grade_start_from)
        elif step == "filter_grade3plus":
            filter_grade3plus()
        elif step == "filter_grade5":
            filter_grade5()
        elif step == "build_function_library":
            build_function_library()


if __name__ == "__main__":
    main()
