"""
Call an LLM via OpenRouter to validate answers in problemset.json.
Sends each problem + expected answer to the LLM and saves feedback.

Usage:
    OPENROUTER_API_KEY=... python scripts/validation.py [--model MODEL] [--concurrency N]
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import aiohttp

# ── Paste your system prompt here ──────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an expert Financial Engineering Professor known for your rigorous academic standards and meticulous grading. 
Your task is to evaluate a financial engineering exam question against a student's final answer. Note that you only have the student's final result/value, not their step-by-step calculations or reasoning.
[CRITICAL INSTRUCTION: You must solve the question completely and independently in your head before looking at the student's final answer. Do NOT output your independent step-by-step math or solution in the final response. Use your internal reasoning to derive the exact answer, then use that hidden result to verify the student's submission.]

You must structure your entire response using the template below. Do not deviate from this format.
---
Verdict: [Correct / Incorrect / Partially Correct due to rounding or minor variations]
Analysis: [State whether the student's final number/metric matches your hidden, independently derived result. If it differs, deduce what mathematical or conceptual error likely led to their specific final output.]
Ambiguities: [Quote the exact phrasing or variables from the question that are unclear or invite alternative interpretations, or state "None" if the question is perfectly precise.]
"""
# ───────────────────────────────────────────────────────────────────────────────

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-fable-5"
DEFAULT_CONCURRENCY = 32

ROOT = Path(__file__).parent.parent
PROBLEMSET_PATH = ROOT / "data" / "trainingset.json"
FEEDBACK_PATH = ROOT / "data" / "feedback.json"

_env_file = ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())


def build_user_message(problem: dict) -> str:
    return f"Problem:\n{problem['problem']}\n\nAnswer:\n{problem['answer']}"


async def call_openrouter(
    session: aiohttp.ClientSession,
    problem: dict,
    model: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    eid = problem["entry_id"]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages = []
    if SYSTEM_PROMPT.strip():
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": build_user_message(problem)})

    payload = {"model": model, "messages": messages}

    async with semaphore:
        async with session.post(OPENROUTER_URL, headers=headers, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"HTTP {resp.status} for entry_id={eid}: {text}")
            data = await resp.json()

    content = data["choices"][0]["message"]["content"]
    return {
        "entry_id": eid,
        "problem": problem["problem"],
        "answer": problem["answer"],
        "feedback": content,
    }


def load_existing(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    with open(path) as f:
        entries = json.load(f)
    return {e["entry_id"]: e for e in entries}


def save_all(path: Path, results: dict[int, dict]) -> None:
    sorted_entries = sorted(results.values(), key=lambda e: e["entry_id"])
    with open(path, "w") as f:
        json.dump(sorted_entries, f, indent=2)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Validate answers via OpenRouter LLM")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model ID")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing feedback")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Error: OPENROUTER_API_KEY environment variable not set.")

    with open(PROBLEMSET_PATH) as f:
        raw = json.load(f)
    problems = [{**p, "entry_id": i} for i, p in enumerate(raw)]

    existing = {} if args.overwrite else load_existing(FEEDBACK_PATH)
    pending = [p for p in problems if p["entry_id"] not in existing]

    print(f"Total problems: {len(problems)}")
    print(f"Already validated: {len(existing)}")
    print(f"To process: {len(pending)}")
    if not pending:
        print("Nothing to do.")
        return

    results = dict(existing)
    save_lock = asyncio.Lock()
    completed = 0

    semaphore = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency)

    async with aiohttp.ClientSession(connector=connector) as session:
        async def process(problem: dict) -> None:
            nonlocal completed
            result = await call_openrouter(session, problem, args.model, api_key, semaphore)
            async with save_lock:
                results[result["entry_id"]] = result
                save_all(FEEDBACK_PATH, results)
                completed += 1
                print(f"[{completed}/{len(pending)}] entry_id={result['entry_id']} saved")

        await asyncio.gather(*[process(p) for p in pending])

    print(f"\nDone. {len(results)} entries written to {FEEDBACK_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
