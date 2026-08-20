"""Reference-solution dataset build.

Has the hint model (`opsd.config.HINT_MODEL`, served via OpenRouter) write a full
reference solution for *every* problem in the training set, and stores them as a
resumable JSONL that `opsd.train` conditions the teacher on.

Each call uses the shared inference system prompt, so the reference solutions are
in the `<reasoning>…</reasoning><answer>…</answer>` format the student is trained
to produce. Reasoning-model hint models (the default `openai/gpt-oss-120b`) return
the worked solution on the separate `reasoning`/`reasoning_content` channel and
leave `content` holding just the final answer, so both halves are read back and
re-emitted as one canonical tagged trace (see `compose_reference`) — otherwise the
"reference solution" opsd conditions its teacher on is a bare number, and there is
no solution to distil. On an answer mismatch (checked with the pipeline's grader) the problem
is retried with the ground-truth answer appended to the prompt — the model can then
write a solution that reaches the known answer — up to REFERENCE_MAX_ATTEMPTS. The
first matching trace is kept; if none match, the last (answer-appended) trace is
kept anyway so every problem yields a reference.

The output is resumable: problems already present in the JSONL are skipped, so a
partial or interrupted run can simply be re-invoked.

    python -m opsd.generate_data                 # build/resume with config defaults
    python -m opsd.generate_data --limit 20      # only the first 20 problems (smoke test)
"""
from __future__ import annotations

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pipeline.eval_script import run_eval
from pipeline.utils import logger, openrouter_call_with_reasoning, setup_logging

from . import config

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_REASONING_RE = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.DOTALL | re.IGNORECASE)
_write_lock = threading.Lock()


def extract_answer(completion: str) -> str:
    """Pull the final answer out of an `<answer>…</answer>` block.

    Falls back to the last non-empty line if the tag is absent, so a
    slightly-off-format response can still be graded.
    """
    m = _ANSWER_RE.search(completion)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in completion.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def compose_reference(content: str, native_reasoning: str = "") -> str:
    """Build the reference solution from one teacher response.

    A reasoning-model hint model (the default `openai/gpt-oss-120b`) returns its
    WORKED SOLUTION on the separate `reasoning`/`reasoning_content` channel and
    leaves `content` holding little more than the final answer — so reading the
    content alone yields references like "25000", and opsd's teacher is then
    conditioned on a bare answer instead of a solution to distil from. Both halves
    are therefore re-emitted as one canonical `<reasoning>…</reasoning>
    <answer>…</answer>` trace, the format the system prompt asks for (the same
    treatment sft.generate_data gives its distillation traces).

    A response that already carries the tags inline keeps them; when the model
    produced both halves they are merged into ONE `<reasoning>` block, in
    generation order (the native channel comes first).
    """
    content = (content or "").strip()
    native_reasoning = (native_reasoning or "").strip()

    tagged = _REASONING_RE.search(content)
    if tagged and not native_reasoning:
        return content
    if not tagged and not native_reasoning:
        return content  # nothing but an answer — best effort, flagged by the caller

    reasoning = "\n\n".join(
        part for part in (native_reasoning, tagged.group(1).strip() if tagged else "") if part
    )
    answer = extract_answer(content) or content
    return f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>\n{answer}\n</answer>"


def load_trainingset(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list of problems, got {type(data).__name__}")
    return data


def _already_done(out_path: Path) -> set:
    """Return the set of problem `number`s already written to the JSONL."""
    done: set = set()
    if not out_path.exists():
        return done
    for line in out_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(json.loads(line)["number"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def _user_message(problem: str, ground_truth: str, reveal_answer: bool) -> str:
    """Build the reference-build user turn.

    On retries we reveal the ground-truth answer so the model can write a solution
    that actually reaches it (the answer is otherwise never shown to the teacher).
    """
    if not reveal_answer:
        return problem
    return (
        f"{problem}\n\n"
        f"The correct final answer is: {ground_truth}. Produce a complete reference "
        f"solution that reasons step by step and arrives at this answer."
    )


def generate_one(item: dict, *, max_attempts: int) -> dict:
    """Generate a reference solution for one problem.

    Attempt 1 sees only the problem. If the answer doesn't match, subsequent
    attempts append the ground-truth answer to the prompt. Returns the first
    matching trace, or the last attempt if none matched (so every problem yields
    a reference — flagged `correct=False`).
    """
    problem, ground_truth = item["problem"], str(item.get("answer", ""))

    last_completion = ""
    last_predicted = ""
    for attempt in range(1, max(1, max_attempts) + 1):
        reveal = attempt > 1  # bare problem first, then reveal the answer on retries
        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": _user_message(problem, ground_truth, reveal)},
        ]
        out = openrouter_call_with_reasoning(
            messages,
            model=config.HINT_MODEL,
            temperature=config.HINT_TEMPERATURE,
            max_tokens=config.HINT_MAX_TOKENS,
        )
        # Grade the final answer (which lives in the content), but keep the WORKED
        # SOLUTION — which a reasoning model returns on its own channel — as the
        # reference; see compose_reference.
        completion = compose_reference(out["content"], out["reasoning"])
        predicted = extract_answer(out["content"]) or extract_answer(completion)
        last_completion, last_predicted = completion, predicted
        if "<reasoning>" not in completion:
            logger.warning(
                "Problem %s: reference has no worked solution, only an answer "
                "(attempt %d/%d) — the teacher will be conditioned on that alone.",
                item.get("number"), attempt, max_attempts,
            )

        if run_eval(predicted, ground_truth):
            return _record(item, completion, predicted, correct=True)
        logger.info(
            "Problem %s: reference answer '%s' != ground truth '%s' (attempt %d/%d)%s",
            item.get("number"), predicted, ground_truth, attempt, max_attempts,
            " — retrying with answer revealed" if attempt < max_attempts else "",
        )

    logger.warning(
        "Problem %s: no answer-matching reference in %d attempts — keeping last trace.",
        item.get("number"), max_attempts,
    )
    return _record(item, last_completion, last_predicted, correct=False)


def _record(item: dict, completion: str, predicted: str, correct: bool) -> dict:
    return {
        "number": item.get("number"),
        "problem": item["problem"],
        "answer": item.get("answer", ""),
        "predicted_answer": predicted,
        "reference_solution": completion,
        "correct": correct,
        "hint_model": config.HINT_MODEL,
    }


def _append(out_path: Path, record: dict) -> None:
    with _write_lock:
        with out_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")


def generate(
    trainingset_path: str | None = None,
    out_path: str | None = None,
    *,
    limit: int | None = None,
) -> str:
    """Build the reference-solution dataset. Returns the output path."""
    trainingset_path = trainingset_path or config.TRAININGSET_PATH
    out_path = Path(out_path or config.REFERENCE_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    problems = load_trainingset(trainingset_path)
    if limit is not None:
        problems = problems[:limit]

    done = _already_done(out_path)
    todo = [p for p in problems if p.get("number") not in done]
    logger.info(
        "Building references for %s: %d problems total, %d already done, %d to generate "
        "(hint_model=%s, max_attempts=%d).",
        trainingset_path, len(problems), len(done), len(todo),
        config.HINT_MODEL, config.REFERENCE_MAX_ATTEMPTS,
    )
    if not todo:
        logger.info("Nothing to do — reference set already complete at %s.", out_path)
        return str(out_path)

    kept = matched = 0
    with ThreadPoolExecutor(max_workers=config.GEN_CONCURRENCY) as pool:
        futures = {
            pool.submit(generate_one, item, max_attempts=config.REFERENCE_MAX_ATTEMPTS): item
            for item in todo
        }
        for fut in as_completed(futures):
            item = futures[fut]
            try:
                record = fut.result()
            except Exception as exc:  # one bad problem shouldn't kill the run
                logger.error("Problem %s failed: %s", item.get("number"), exc)
                continue
            _append(out_path, record)
            kept += 1
            matched += int(bool(record.get("correct")))
            logger.info("Wrote %d / %d references (%d answer-matched)", kept, len(todo), matched)

    logger.info(
        "Reference build complete: %d written (%d answer-matched, %d best-effort) → %s",
        kept, matched, kept - matched, out_path,
    )
    return str(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build hint-model reference solutions for opsd.")
    ap.add_argument("--trainingset", default=config.TRAININGSET_PATH)
    ap.add_argument("--out", default=config.REFERENCE_PATH)
    ap.add_argument("--limit", type=int, default=None, help="Only the first N problems.")
    args = ap.parse_args()

    setup_logging()
    generate(args.trainingset, args.out, limit=args.limit)


if __name__ == "__main__":
    main()
