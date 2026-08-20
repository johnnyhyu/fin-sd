#!/usr/bin/env python3
"""
Phase 2 — generate & extract problems.

    function-library.json
      → generate_problemset  (LLM writes NUM_OUTPUTS problems/function) → extra_problemset.json
      → extract_problems     (regex parse into flat records)           → problemset.json

Usage:
    OPENROUTER_API_KEY=sk-or-... python scripts/phase2_generate_problems.py
    python scripts/phase2_generate_problems.py --only extract_problems
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

import aiohttp

# ── Shared config ────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
# This file lives in data/build/, so the corpus directory is its parent.
_DATA_DIR = Path(__file__).parent.parent

LIBRARY_FILE = _DATA_DIR / "function-library.json"
RAW_PROBLEMSET_FILE = _DATA_DIR / "extra_problemset.json"
PROBLEMSET_FILE = _DATA_DIR / "problemset.json"

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
    ceiling = min(cap, base * (2 ** (attempt - 1)))
    return random.uniform(0, ceiling)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — generate_problemset
# ─────────────────────────────────────────────────────────────────────────────
GENERATE_MODEL = "openai/gpt-5.5"
NUM_OUTPUTS = 3
GENERATE_CONCURRENCY = 16
GENERATE_RETRY_ATTEMPTS = 5

GENERATE_SYSTEM_PROMPT = """
### SYSTEM ROLE
You are an expert Financial Engineering Professor. Your task is to design a highly precise, multi-step financial reasoning problem built entirely around the mechanics of a specific, hidden financial function provided to you.

CRITICAL RULE: The student does not know which function you have been given, nor do they have access to it. You must never reference "the provided function," "the core function," or "the formula below" in the user-facing problem. The student must solve the problem using standard financial theory, definitions, and formulas native to the domain.

### TEST DESIGN PRINCIPLES
The problem must test strict financial logic and mathematical reasoning. It must avoid subjective corporate strategy and instead lead to a single, completely unambiguous, easily verifiable final numerical answer.

To achieve this, structure the problem into three distinct logical phases:

1. Phase 1: Input Derivation (The Setup)
Do not provide the function's inputs explicitly. Instead, describe a realistic market scenario, a timeline, or a set of balance sheet items that forces the user to apply financial definitions to calculate the exact inputs required.

2. Phase 2: Core Function Execution
The user must naturally arrive at the need to calculate the intermediate metric governed by your hidden function, using the precise variables they derived in Phase 1.

3. Phase 3: The Verifiable Closing Mechanism (The Bottom Line)
The intermediate metric from Phase 2 cannot be the final answer. To create a crisp, verifiable endpoint, you must design a final step native to this function's domain. Choose the most logical option below based on the function provided:
* Threshold Comparison: Compare the function's output against a provided benchmark (e.g., hurdle rate, cost of capital, market index) to find an exact spread or a definitive binary decision.
* Valuation Impact: Use the output to calculate a final dollar-value impact (e.g., net profit, portfolio value change, tax liability, or arbitrage gain).
* Optimization/Sensitivity: Ask for the exact change in the final output if one baseline market variable shifts by a specific amount.

### OUTPUT FORMAT
Your output must strictly follow this structure:

#### 1. The Financial Problem (User-Facing)
* Scenario: [A concise, clear financial narrative containing all raw data, dates, and market conditions.]
* The Core Question: [A single, explicit question asking for the final verifiable value, specifying required units and rounding, e.g., "What is the net dollar arbitrage profit, rounded to the nearest whole dollar?"]

#### 2. Verification Metadata (For Automated Grading)
* Final Answer: [The exact numerical value or explicit binary choice, e.g., "6.25%" or "$450" or "Project B"].
"""


class TokenTracker:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def add(self, usage: dict):
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens

    def summary(self) -> str:
        return (
            f"Tokens used — prompt: {self.prompt_tokens:,}  "
            f"completion: {self.completion_tokens:,}  "
            f"total: {self.total_tokens:,}"
        )


async def _generate_call_once(
    session: aiohttp.ClientSession,
    entry: dict,
    api_key: str,
    semaphore: asyncio.Semaphore,
    tracker: TokenTracker,
) -> str | None:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://ctgt.ai",
    }
    messages = [
        {"role": "system", "content": GENERATE_SYSTEM_PROMPT},
        {"role": "user", "content": entry["function"]},
    ]

    payload = {
        "model": GENERATE_MODEL,
        "messages": messages,
        "reasoning": {"effort": "high"},
    }

    for attempt in range(1, GENERATE_RETRY_ATTEMPTS + 1):
        sleep_for = 0.0
        try:
            async with semaphore:
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    if resp.status in RETRYABLE_STATUSES:
                        retry_after = resp.headers.get("Retry-After")
                        sleep_for = float(retry_after) if retry_after else _backoff(attempt)
                        label = "rate limit" if resp.status == 429 else f"server error {resp.status}"
                        print(f"  [{label}] {entry['function_id']} — waiting {sleep_for:.1f}s (attempt {attempt})", flush=True)
                    else:
                        if resp.status != 200:
                            body = await resp.text()
                            print(f"  [http {resp.status}] {entry['function_id']}: {body[:300]}", flush=True)
                            resp.raise_for_status()
                        data = await resp.json()
                        tracker.add(data.get("usage") or {})
                        content = data["choices"][0]["message"].get("content")
                        if content is None:
                            content = (data["choices"][0]["message"].get("reasoning_content")
                                       or data["choices"][0]["message"].get("reasoning") or "")
                            print(f"  [warn] null content for {entry['function_id']}, used fallback field", flush=True)
                        return content.strip()
        except Exception as e:
            if attempt == GENERATE_RETRY_ATTEMPTS:
                print(f"  [error] {entry['function_id']}: {type(e).__name__}: {e}", flush=True)
                return None
            sleep_for = _backoff(attempt)

        if sleep_for:
            await asyncio.sleep(sleep_for)

    return None


async def _generate_process_one(
    session: aiohttp.ClientSession,
    entry: dict,
    api_key: str,
    semaphore: asyncio.Semaphore,
    tracker: TokenTracker,
) -> dict:
    outputs = await asyncio.gather(
        *[_generate_call_once(session, entry, api_key, semaphore, tracker) for _ in range(NUM_OUTPUTS)]
    )
    return {
        "id": _id_key(entry["function_id"]),
        "function_id": entry["function_id"],
        "outputs": list(outputs),
    }


def _save_raw_problemset(results: list[dict]) -> None:
    sorted_results = sorted(results, key=lambda r: r.get("id", 0))
    with open(RAW_PROBLEMSET_FILE, "w") as f:
        json.dump(sorted_results, f, indent=2)


async def _generate_problemset_async() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        sys.exit("Error: OPENROUTER_API_KEY environment variable not set.")

    with open(LIBRARY_FILE) as f:
        all_entries = json.load(f)

    all_entries.sort(key=lambda e: _id_key(e["function_id"]))

    existing: dict[str, dict] = {}
    if RAW_PROBLEMSET_FILE.exists():
        with open(RAW_PROBLEMSET_FILE) as f:
            for row in json.load(f):
                outputs = row.get("outputs", [])
                if len(outputs) == NUM_OUTPUTS and all(o is not None for o in outputs):
                    existing[row["function_id"]] = row
        print(f"Resuming: {len(existing)} already processed, {len(all_entries) - len(existing)} remaining.")

    todo = [e for e in all_entries if e["function_id"] not in existing]

    print(f"Processing {len(todo)} functions with model={GENERATE_MODEL}, concurrency={GENERATE_CONCURRENCY}...")

    results: list[dict] = list(existing.values())
    completed = 0
    semaphore = asyncio.Semaphore(GENERATE_CONCURRENCY)
    tracker = TokenTracker()

    async with aiohttp.ClientSession() as session:
        tasks = [_generate_process_one(session, entry, api_key, semaphore, tracker) for entry in todo]

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            errs = sum(1 for o in result.get("outputs", []) if o is None)
            status = "OK" if errs == 0 else f"ERR({errs}/{NUM_OUTPUTS})"
            print(f"  [{completed}/{len(todo)}] {result['function_id']} → {status}", flush=True)

            if completed % 50 == 0:
                _save_raw_problemset(results)
                print(tracker.summary(), flush=True)

    _save_raw_problemset(results)
    errors = sum(1 for r in results for o in r.get("outputs", []) if o is None)
    print(f"\nDone. {len(results)} processed, {errors} failed outputs. Output: {RAW_PROBLEMSET_FILE}")
    print(tracker.summary())


def generate_problemset() -> None:
    asyncio.run(_generate_problemset_async())


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — extract_problems
# ─────────────────────────────────────────────────────────────────────────────
PROBLEM_SECTION = re.compile(
    r"####\s*1\.\s*The Financial Problem.*?\n(.*?)(?=####\s*2\.|\Z)",
    re.DOTALL | re.IGNORECASE,
)
ANSWER_PATTERN = re.compile(r"\*\s*Final Answer:\s*(.+)", re.IGNORECASE)


def _extract(text: str):
    problem_match = PROBLEM_SECTION.search(text)
    answer_match = ANSWER_PATTERN.search(text)
    if not problem_match or not answer_match:
        return None
    problem = problem_match.group(1).strip()
    answer = answer_match.group(1).strip()
    return problem, answer


def extract_problems() -> None:
    with open(RAW_PROBLEMSET_FILE) as f:
        raw = json.load(f)

    try:
        with open(PROBLEMSET_FILE) as f:
            problems = json.load(f)
    except FileNotFoundError:
        problems = []
    existing = {(p["id"], p["problem_index"]) for p in problems}
    skipped = 0

    for item in raw:
        for idx, output in enumerate(item["outputs"]):
            if not isinstance(output, str):
                skipped += 1
                continue
            result = _extract(output)
            if result is None:
                skipped += 1
                print(f"WARNING: could not parse id={item['id']} output[{idx}]", file=sys.stderr)
                continue
            problem, answer = result
            problem_index = idx + 3 if (item["id"], idx) in existing else idx
            problems.append(
                {
                    "id": item["id"],
                    "function_id": item["function_id"],
                    "problem_index": problem_index,
                    "problem": problem,
                    "answer": answer,
                }
            )

    with open(PROBLEMSET_FILE, "w") as f:
        json.dump(problems, f, indent=2)

    print(f"Extracted {len(problems)} problems ({skipped} skipped) -> {PROBLEMSET_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase entry point
# ─────────────────────────────────────────────────────────────────────────────
STEPS = ["generate_problemset", "extract_problems"]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Phase 2 — generate & extract problems.")
    parser.add_argument("--only", choices=STEPS, help="Run a single step instead of the whole phase.")
    args = parser.parse_args(argv)

    steps = [args.only] if args.only else STEPS
    for step in steps:
        print(f"\n=== Phase 2 · {step} ===", flush=True)
        if step == "generate_problemset":
            generate_problemset()
        elif step == "extract_problems":
            extract_problems()


if __name__ == "__main__":
    main()
