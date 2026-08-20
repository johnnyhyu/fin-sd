#!/usr/bin/env python3
"""
Phase 3 — solve, grade, and assemble the training set.

    problemset.json
      → run_inference     (model answers each problem)          → answers.json
      → grade_answers     (LLM equivalence judge vs. expected)  → grades.json
      → create_trainingset (keep the ones the model got wrong)  → trainingset.json

The training set is the problems the answering model got *wrong* (grade == 0):
those are the ones worth distilling against a teacher.

Usage:
    OPENROUTER_API_KEY=... python scripts/phase3_build_trainingset.py
    python scripts/phase3_build_trainingset.py --model openai/gpt-oss-120b --concurrency 32
    python scripts/phase3_build_trainingset.py --only create_trainingset
"""

import argparse
import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import aiohttp
import requests

# ── Shared config ────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
# This file lives in data/build/, so the corpus directory is its parent.
_DATA_DIR = Path(__file__).parent.parent

PROBLEMSET_FILE = _DATA_DIR / "problemset.json"
ANSWERS_FILE = _DATA_DIR / "answers.json"
GRADES_FILE = _DATA_DIR / "grades.json"
TRAININGSET_FILE = _DATA_DIR / "trainingset.json"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_env_file = _ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — run_inference
# ─────────────────────────────────────────────────────────────────────────────
INFERENCE_MODEL = "openai/gpt-oss-120b"
INFERENCE_CONCURRENCY = 32
INFERENCE_RETRY_STATUSES = {429, 500, 502, 503, 504}
INFERENCE_MAX_RETRIES = 5
INFERENCE_RETRY_BASE_DELAY = 1.0

INFERENCE_SYSTEM_PROMPT = """
Act as a senior financial analyst and expert. Before providing your final output, you must explicitly think through the problem step-by-step. Select and apply the most relevant advanced reasoning techniques from the framework below to guarantee absolute accuracy:

1. Systematic Analysis (SA): Deconstruct the problem's structure, identifying all inputs, variables, variables over time, and core financial objectives.
2. Method Reuse (MR): Map the problem to established financial models or formulas (e.g., NPV, DCF, Black-Scholes, CAPM, Portfolio Optimization) where applicable.
3. Divide and Conquer (DC): Break complex multi-stage financial calculations into isolated, sequential sub-steps.
4. Self-Refinement (SR): Audit your intermediate calculations. Cross-check for mathematical consistency and logical fallacies before proceeding.
5. Context Identification (CI): Align the solution with industry-specific realities (e.g., tax implications, compounding frequencies, macroeconomic assumptions).
6. Emphasizing Constraints (EC): Strictly adhere to specified constraints, including rounding rules, decimal precision, percentages, and currency units.

CRITICAL OUTPUT REQUIREMENT: You must output ONLY the final answer. Do not include any introductory text, reasoning, step-by-step explanation, or concluding remarks. Your entire response must consist of exactly and only the final answer itself (whether it is a number, word, or phrase), with no other text whatsoever.
"""


def _answer_id(problem: dict) -> int:
    return problem["id"] * 6 + problem["problem_index"]


async def _inference_call(
    session: aiohttp.ClientSession,
    problem: dict,
    model: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    aid = _answer_id(problem)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages = []
    if INFERENCE_SYSTEM_PROMPT.strip():
        messages.append({"role": "system", "content": INFERENCE_SYSTEM_PROMPT})
    messages.append({"role": "user", "content": problem["problem"]})

    payload = {"model": model, "messages": messages}

    async with semaphore:
        for attempt in range(INFERENCE_MAX_RETRIES + 1):
            try:
                async with session.post(OPENROUTER_URL, headers=headers, json=payload) as resp:
                    if resp.status in INFERENCE_RETRY_STATUSES and attempt < INFERENCE_MAX_RETRIES:
                        delay = INFERENCE_RETRY_BASE_DELAY * (2 ** attempt)
                        text = await resp.text()
                        print(f"HTTP {resp.status} for id={aid} (attempt {attempt+1}/{INFERENCE_MAX_RETRIES+1}), retrying in {delay:.1f}s: {text[:200]}", file=sys.stderr)
                        await asyncio.sleep(delay)
                        continue
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"HTTP {resp.status} for id={aid}: {text}")
                    data = await resp.json()
                    break
            except aiohttp.ClientError as exc:
                if attempt < INFERENCE_MAX_RETRIES:
                    delay = INFERENCE_RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"Network error for id={aid} (attempt {attempt+1}/{INFERENCE_MAX_RETRIES+1}), retrying in {delay:.1f}s: {exc}", file=sys.stderr)
                    await asyncio.sleep(delay)
                else:
                    raise

    content = data["choices"][0]["message"]["content"]
    return {
        "answer_id": aid,
        "id": problem["id"],
        "problem_index": problem["problem_index"],
        "problem": problem["problem"],
        "expected_answer": problem.get("answer"),
        "llm_answer": content,
    }


def _load_answers(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    with open(path) as f:
        entries = json.load(f)
    return {e["answer_id"]: e for e in entries}


def _save_answers(path: Path, results: dict[int, dict]) -> None:
    sorted_entries = sorted(results.values(), key=lambda e: e["answer_id"])
    with open(path, "w") as f:
        json.dump(sorted_entries, f, indent=2)


async def _run_inference_async(model: str, concurrency: int, overwrite: bool) -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Error: OPENROUTER_API_KEY environment variable not set.")

    with open(PROBLEMSET_FILE) as f:
        problems = json.load(f)

    existing = {} if overwrite else _load_answers(ANSWERS_FILE)
    pending = [p for p in problems if _answer_id(p) not in existing]

    print(f"Total problems: {len(problems)}")
    print(f"Already answered: {len(existing)}")
    print(f"To process: {len(pending)}")
    if not pending:
        print("Nothing to do.")
        return

    results = dict(existing)
    save_lock = asyncio.Lock()
    completed = 0

    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency)

    async with aiohttp.ClientSession(connector=connector) as session:
        async def process(problem: dict) -> None:
            nonlocal completed
            try:
                result = await _inference_call(session, problem, model, api_key, semaphore)
            except Exception as exc:
                print(f"Error on id={problem['id']} problem_index={problem['problem_index']}: {exc}", file=sys.stderr)
                raise
            async with save_lock:
                results[result["answer_id"]] = result
                _save_answers(ANSWERS_FILE, results)
                completed += 1
                print(f"[{completed}/{len(pending)}] answer_id={result['answer_id']} saved")

        await asyncio.gather(*[process(p) for p in pending])

    print(f"\nDone. {len(results)} answers written to {ANSWERS_FILE}")


def run_inference(model: str = INFERENCE_MODEL, concurrency: int = INFERENCE_CONCURRENCY, overwrite: bool = False) -> None:
    asyncio.run(_run_inference_async(model, concurrency, overwrite))


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — grade_answers
# ─────────────────────────────────────────────────────────────────────────────
JUDGE_MODEL = "deepseek/deepseek-v4-flash"

JUDGE_SYSTEM_PROMPT = """
You are a strict math grader. Your sole task is to determine if the numerical value in the LLM Answer is mathematically equivalent to the Expected Answer within a strict tolerance.

Follow these steps executionally:
1. EXTRACT & NORMALIZE: Isolate the core numerical values from both the Expected Answer and the LLM Answer.
   - Strip away all currency symbols ($), commas, markdown, and spaces.
   - Ignore scale words like "million", "billion", or "B". e.g. 1.5 billion = 1500000000 OR 1.5.
2. CALCULATE: Let E be the normalized Expected Answer and L be the normalized LLM Answer. Calculate the error using the formula:
   $$\text{Error } = \frac{|L - E|}{|E|} \times 100$$
3. EVALUATE:
   - If the Error is less than or equal to 0.2, they are equivalent.
   - If the Error is greater than 0.2, they are NOT equivalent.
   e.g. 1001 and 1000 is equivalent (0.1 error), but 1003 is not (0.3 error).

Output Format:
Reply with exactly '1' if they are equivalent, or '0' if they are not. Do not include any other text, explanation, or markdown in your final response. Only output the single digit.
"""


def _check_equivalence(expected: str, llm: str, api_key: str, max_retries: int = 3) -> int:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Expected answer: {expected}\nLLM answer: {llm}"},
        ],
        "max_tokens": 1024,
        "temperature": 0,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            return 1 if content.startswith("1") else 0
        except (requests.RequestException, KeyError):
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def grade_answers() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Error: OPENROUTER_API_KEY environment variable not set.")

    with open(ANSWERS_FILE) as f:
        answers = json.load(f)

    grades: dict[str, int] = {}
    if GRADES_FILE.exists():
        with open(GRADES_FILE) as f:
            grades = json.load(f)
        print(f"Resuming: {len(grades)} already graded, loaded from {GRADES_FILE}")

    total = len(answers)
    pending = [item for item in answers if str(item["answer_id"]) not in grades]
    done_count = total - len(pending)
    correct_count = sum(grades.values())

    if done_count:
        print(f"Skipping {done_count} already-graded items, {len(pending)} remaining.")

    if not pending:
        print("All items already graded.")
    else:
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {
                executor.submit(_check_equivalence, item["expected_answer"], item["llm_answer"], api_key): item
                for item in pending
            }
            for fut in as_completed(futures):
                item = futures[fut]
                label = fut.result()

                grades[str(item["answer_id"])] = label
                with open(GRADES_FILE, "w") as f:
                    json.dump(grades, f)

                done_count += 1
                if label == 1:
                    correct_count += 1
                status = "CORRECT" if label == 1 else "WRONG"
                pct = done_count / total * 100
                print(
                    f"[{done_count}/{total} {pct:.0f}%] [{status}]"
                    f" expected={item['expected_answer']!r} | llm={item['llm_answer']!r}"
                )

    accuracy = correct_count / total if total else 0.0
    print(f"\nAccuracy: {correct_count}/{total} = {accuracy:.2%}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — create_trainingset
# ─────────────────────────────────────────────────────────────────────────────
def create_trainingset() -> None:
    with open(GRADES_FILE) as f:
        grades = json.load(f)

    with open(PROBLEMSET_FILE) as f:
        problems = json.load(f)

    lookup = {(p["id"], p["problem_index"]): p for p in problems}

    trainingset = []
    missing = []

    for grade_id_str, label in grades.items():
        if label != 0:
            continue
        grade_id = int(grade_id_str)
        problem_id = grade_id // 6
        problem_index = grade_id % 6
        key = (problem_id, problem_index)
        if key not in lookup:
            missing.append(grade_id_str)
            continue
        p = lookup[key]
        trainingset.append({"problem": p["problem"], "answer": p["answer"]})

    with open(TRAININGSET_FILE, "w") as f:
        json.dump(trainingset, f, indent=2)

    print(f"Saved {len(trainingset)} entries to {TRAININGSET_FILE.name}")
    if missing:
        print(f"Warning: {len(missing)} grade IDs had no matching problem: {missing[:10]}{'...' if len(missing) > 10 else ''}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase entry point
# ─────────────────────────────────────────────────────────────────────────────
STEPS = ["run_inference", "grade_answers", "create_trainingset"]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Phase 3 — solve, grade, and assemble the training set.")
    parser.add_argument("--only", choices=STEPS, help="Run a single step instead of the whole phase.")
    parser.add_argument("--model", default=INFERENCE_MODEL, help="run_inference: OpenRouter model ID.")
    parser.add_argument("--concurrency", type=int, default=INFERENCE_CONCURRENCY, help="run_inference: parallel requests.")
    parser.add_argument("--overwrite", action="store_true", help="run_inference: ignore existing answers.")
    args = parser.parse_args(argv)

    steps = [args.only] if args.only else STEPS
    for step in steps:
        print(f"\n=== Phase 3 · {step} ===", flush=True)
        if step == "run_inference":
            run_inference(args.model, args.concurrency, args.overwrite)
        elif step == "grade_answers":
            grade_answers()
        elif step == "create_trainingset":
            create_trainingset()


if __name__ == "__main__":
    main()
