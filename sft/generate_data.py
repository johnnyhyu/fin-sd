"""Distillation data generation.

Has the teacher (`sft.config.TEACHER_MODEL`, i.e. the pipeline's hint model,
served via OpenRouter) solve every problem in the training set, and writes the
resulting (prompt, solution) pairs to a JSONL file that `sft.train` consumes.

Each teacher call uses the shared inference system prompt, so the traces are in
the `<reasoning>…</reasoning><answer>…</answer>` format the student is trained to
produce. Reasoning-model teachers put their chain-of-thought on the separate
`reasoning`/`reasoning_content` channel instead of inside the content, so both
halves are read back and re-emitted as one canonical tagged trace — the student
has to be supervised on the reasoning, not just the final answer. With rejection
sampling on (the default) a trace is only kept if its final answer matches the
ground-truth answer, checked with the pipeline's grader.

The output is resumable: problems already present in the JSONL are skipped, so a
partial or interrupted run can simply be re-invoked.

    python -m sft.generate_data                 # generate/resume with config defaults
    python -m sft.generate_data --limit 20      # only the first 20 problems (smoke test)
    python -m sft.generate_data --no-reject      # keep every trace, no answer check
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
    slightly-off-format teacher response can still be graded.
    """
    m = _ANSWER_RE.search(completion)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in completion.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def split_trace(content: str, native_reasoning: str = "") -> tuple[str, str]:
    """Split one teacher response into (reasoning, final answer).

    `content` is the assistant message content, `native_reasoning` the separate
    reasoning channel (empty for teachers that inline their `<reasoning>` block).
    Untagged content is split at its last non-empty line, so an off-format
    response still yields reasoning instead of collapsing to the answer alone.
    """
    content = content.strip()
    native_reasoning = native_reasoning.strip()

    reasoning_m = _REASONING_RE.search(content)
    answer_m = _ANSWER_RE.search(content)

    if reasoning_m:
        reasoning = reasoning_m.group(1).strip()
    elif answer_m:
        reasoning = content[: answer_m.start()].strip()
    else:
        reasoning = ""

    if answer_m:
        answer = answer_m.group(1).strip()
    elif reasoning_m:
        answer = content[reasoning_m.end():].strip()
    else:
        lines = [ln for ln in content.splitlines() if ln.strip()]
        answer = lines[-1].strip() if lines else ""
        if not reasoning:
            reasoning = "\n".join(lines[:-1]).strip()

    # The native channel is emitted before the content, so keep that order when
    # a teacher produced both (mirrors inference_script.run_inference).
    if native_reasoning:
        reasoning = f"{native_reasoning}\n\n{reasoning}".strip()
    return reasoning, answer


def format_completion(reasoning: str, answer: str) -> str:
    """Render a (reasoning, answer) pair in the format config.SYSTEM_PROMPT asks for."""
    return (
        f"<reasoning>\n{reasoning}\n</reasoning>\n"
        f"<answer>\n{answer}\n</answer>"
    )


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


def generate_one(item: dict, *, reject: bool, max_attempts: int) -> dict | None:
    """Generate (and optionally verify) one teacher solution.

    Returns a record dict on success, or None if every attempt was rejected —
    either for a wrong final answer (when `reject` is True) or for coming back
    without any reasoning (when config.REQUIRE_REASONING is True).
    """
    problem, ground_truth = item["problem"], item.get("answer", "")
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]

    # Both gates retry: a wrong answer and a reasoning-free trace are each
    # grounds for resampling the teacher.
    attempts = max_attempts if (reject or config.REQUIRE_REASONING) else 1
    for attempt in range(1, attempts + 1):
        out = openrouter_call_with_reasoning(
            messages,
            model=config.TEACHER_MODEL,
            temperature=config.TEACHER_TEMPERATURE,
            max_tokens=config.TEACHER_MAX_TOKENS,
        )
        reasoning, predicted = split_trace(out["content"], out["reasoning"])

        # A reasoning-free trace teaches the student to answer without thinking,
        # which is the opposite of the point of this dataset; an answer-free one
        # has nothing to grade. Either is worth resampling.
        if not reasoning:
            logger.warning(
                "Problem %s: teacher returned no reasoning (attempt %d/%d)%s",
                item.get("number"), attempt, attempts,
                "" if config.REQUIRE_REASONING else " — keeping it anyway "
                                                     "(SFT_REQUIRE_REASONING=0).",
            )
            if config.REQUIRE_REASONING:
                continue
        if not predicted:
            logger.warning(
                "Problem %s: teacher returned no final answer (attempt %d/%d).",
                item.get("number"), attempt, attempts,
            )
            continue

        if not reject:
            return _record(item, reasoning, predicted, correct=None)

        if run_eval(predicted, ground_truth):
            return _record(item, reasoning, predicted, correct=True)
        logger.info(
            "Problem %s: teacher answer '%s' != ground truth '%s' (attempt %d/%d)",
            item.get("number"), predicted, ground_truth, attempt, attempts,
        )

    logger.warning(
        "Problem %s: no usable teacher trace in %d attempts — dropping.",
        item.get("number"), attempts,
    )
    return None


def _record(item: dict, reasoning: str, predicted: str, correct: bool | None) -> dict:
    """Build the JSONL record: `completion` is the full tagged trace to train on."""
    return {
        "number": item.get("number"),
        "problem": item["problem"],
        "answer": item.get("answer", ""),
        "predicted_answer": predicted,
        "reasoning": reasoning,
        "completion": format_completion(reasoning, predicted),
        "correct": correct,
        "teacher_model": config.TEACHER_MODEL,
    }


def _append(out_path: Path, record: dict) -> None:
    with _write_lock:
        with out_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")


def generate(
    trainingset_path: str | None = None,
    out_path: str | None = None,
    *,
    reject: bool | None = None,
    limit: int | None = None,
) -> str:
    """Generate the distillation dataset. Returns the output path."""
    trainingset_path = trainingset_path or config.TRAININGSET_PATH
    out_path = Path(out_path or config.DISTILL_PATH)
    reject = config.REJECT_SAMPLING if reject is None else reject
    out_path.parent.mkdir(parents=True, exist_ok=True)

    problems = load_trainingset(trainingset_path)
    if limit is not None:
        problems = problems[:limit]

    done = _already_done(out_path)
    todo = [p for p in problems if p.get("number") not in done]
    logger.info(
        "Distilling %s: %d problems total, %d already done, %d to generate "
        "(teacher=%s, reject_sampling=%s, require_reasoning=%s).",
        trainingset_path, len(problems), len(done), len(todo), config.TEACHER_MODEL,
        reject, config.REQUIRE_REASONING,
    )
    if not todo:
        logger.info("Nothing to do — distillation set already complete at %s.", out_path)
        return str(out_path)

    kept = dropped = 0
    with ThreadPoolExecutor(max_workers=config.GEN_CONCURRENCY) as pool:
        futures = {
            pool.submit(
                generate_one, item, reject=reject, max_attempts=config.REJECT_MAX_ATTEMPTS
            ): item
            for item in todo
        }
        for fut in as_completed(futures):
            item = futures[fut]
            try:
                record = fut.result()
            except Exception as exc:  # one bad problem shouldn't kill the run
                logger.error("Problem %s failed: %s", item.get("number"), exc)
                dropped += 1
                continue
            if record is None:
                dropped += 1
                continue
            _append(out_path, record)
            kept += 1
            logger.info("Kept %d / %d (dropped %d)", kept, len(todo), dropped)

    logger.info("Distillation complete: %d kept, %d dropped → %s", kept, dropped, out_path)
    return str(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate teacher distillation data for SFT.")
    ap.add_argument("--trainingset", default=config.TRAININGSET_PATH)
    ap.add_argument("--out", default=config.DISTILL_PATH)
    ap.add_argument("--limit", type=int, default=None, help="Only the first N problems.")
    ap.add_argument("--no-reject", action="store_true", help="Keep every trace (skip answer check).")
    args = ap.parse_args()

    setup_logging()
    generate(
        args.trainingset,
        args.out,
        reject=False if args.no_reject else None,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
