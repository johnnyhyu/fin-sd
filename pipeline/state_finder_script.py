"""
state_finder_script.py — truncates a reasoning chain at the point of the mistake.

Given the original reasoning and the verbatim mistake sentence extracted by hint_script,
truncates the reasoning just before that sentence using fuzzy string matching.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from rapidfuzz import fuzz

from .utils import logger, FILE_LOG_LOCK

# Each run_state_finder call appends one JSON record (full input reasoning +
# truncated output) here, for debugging the truncation step.
_LOG_PATH = Path(__file__).parent.parent / "data" / "reasoning_log.jsonl"

# Minimum partial_ratio (0-100) at which we trust the hint model's quote to name a
# real position in the trace. The quote is supposed to be VERBATIM (both hint
# prompts say so twice), so a genuine match scores in the high 90s; the tolerance
# exists for whitespace and unicode-punctuation drift, not for paraphrase. Below
# this the alignment is arbitrary — it still returns SOME offset, and the pipeline
# would then truncate at a point unrelated to the mistake and distil the teacher's
# continuation of the wrong prefix. That corrupts the training signal without ever
# raising, so the item is dropped instead.
_MIN_MATCH_SCORE = 70.0

# Fallback floor for the head-of-quote rescue below. Only the START of the quote
# decides the cut point (we return everything before dest_start), so a quote whose
# head is verbatim localises the mistake correctly no matter how far its tail
# drifts. partial_ratio averages that drift over the whole needle, which pushes
# genuine hits under _MIN_MATCH_SCORE whenever the hint model merges two sentences
# or elides with "...". Scoring the head alone separates those from real misses:
# over data/reasoning_log.jsonl (1224 quotes) this recovers 10 of the 34 sub-70
# items and admits none of the bad ones — quotes lifted from the prompt scaffolding
# ("Student's Answer (incorrect): 103") score ~50 on their head too, because they
# do not appear in the trace at any length.
_MIN_HEAD_MATCH_SCORE = 90.0

# Head candidates: the quote's first sentence (if at least this long — shorter
# leading fragments carry too little signal to place a cut) …
_MIN_HEAD_CHARS = 40
# … and, failing that, its first N characters.
_HEAD_PREFIX_CHARS = 60

# Sentence boundary inside a quote: terminal punctuation followed by whitespace,
# or an explicit elision marker (the model writes "..." where it skipped text).
_SENTENCE_END = re.compile(r"[.!?](?=\s)|\.\.\.|…")

# Minimum number of characters that must FOLLOW the cut point. The truncated
# prefix is handed to the teacher to continue, so a cut this close to the end
# leaves nothing to rewrite: the quote named the final-answer line ("103",
# "Thus answer: -1605.", "I'll output '$1.90'.") rather than the mistake, and the
# whole flawed chain would be distilled as correct prefix. In the log every cut
# leaving under ~150 chars is that failure mode, while real late mistakes
# ("Salvage: 160,000.") leave 240+; arithmetic slips in the final line — the other
# legitimate source of late cuts — are already filtered by SKIP_ARITHMETIC_ERRORS.
_MIN_TAIL_CHARS = 150


def _save_record(reasoning: str, hint: str, quoted_sentence: str, truncated: str) -> None:
    """Append a JSON record of one state-finder call to _LOG_PATH and log it."""
    logger.info(
        "State finder: %d-char reasoning truncated to %d chars.",
        len(reasoning), len(truncated),
    )
    logger.debug("State finder full reasoning:\n%s", reasoning)
    logger.debug("State finder truncated reasoning:\n%s", truncated)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "correct": False,
        "hint": hint,
        "quoted_sentence": quoted_sentence,
        "full_reasoning": reasoning,
        "truncated_reasoning": truncated,
        "full_len": len(reasoning),
        "truncated_len": len(truncated),
    }
    try:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Lock so concurrent worker threads can't interleave multi-KB lines.
        with FILE_LOG_LOCK, _LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line)
    except Exception as exc:  # never let logging break the pipeline
        logger.warning("Failed to write state finder log to %s: %s", _LOG_PATH, exc)


def _head_candidates(needle: str) -> list[str]:
    """Return the leading fragments of needle to retry a failed match with.

    First the opening sentence (up to and including its terminator or elision
    marker), then a fixed-length prefix. Both are dropped if they are not shorter
    than needle itself — re-scoring the whole quote would just repeat the match
    that already failed.
    """
    candidates: list[str] = []
    for match in _SENTENCE_END.finditer(needle):
        if match.end() >= _MIN_HEAD_CHARS:
            candidates.append(needle[:match.end()])
            break
    candidates.append(needle[:_HEAD_PREFIX_CHARS])
    return [c for c in candidates if len(c) < len(needle)]


def _truncate_before_sentence(reasoning: str, quoted_sentence: str) -> str:
    """Find quoted_sentence in reasoning and return everything strictly before it.

    Uses rapidfuzz.fuzz.partial_ratio_alignment to handle minor transcription
    differences between the hint model's quote and the original text.
    On exact ties the leftmost window is returned, so a verbatim recurrence of
    the mistake sentence later in the chain cannot pull the cut point forward.

    Rejects anything below _MIN_MATCH_SCORE: partial_ratio always returns a best
    window, so a paraphrased or hallucinated quote yields a plausible-looking
    offset that has nothing to do with the mistake. A quote that fails that floor
    gets one retry on its head alone (see _MIN_HEAD_MATCH_SCORE), which recovers
    the common case of a verbatim opening followed by a merged or elided tail.
    """
    needle = quoted_sentence.strip()
    if not needle:
        raise RuntimeError("Hint returned an empty quoted_sentence.")

    alignment = fuzz.partial_ratio_alignment(needle, reasoning)
    score = 0.0 if alignment is None else alignment.score
    if alignment is not None and score >= _MIN_MATCH_SCORE:
        logger.debug("State finder: quote matched at offset %d (score %.1f).",
                     alignment.dest_start, score)
        return reasoning[:alignment.dest_start]

    for head in _head_candidates(needle):
        head_alignment = fuzz.partial_ratio_alignment(head, reasoning)
        if head_alignment is not None and head_alignment.score >= _MIN_HEAD_MATCH_SCORE:
            logger.info(
                "State finder: full quote scored %.1f (< %.0f) but its first %d chars "
                "matched at %.1f; cutting at offset %d.",
                score, _MIN_MATCH_SCORE, len(head), head_alignment.score,
                head_alignment.dest_start,
            )
            return reasoning[:head_alignment.dest_start]

    raise RuntimeError(
        f"Hint quoted_sentence does not match the original reasoning "
        f"(best partial_ratio {score:.1f} < {_MIN_MATCH_SCORE:.0f}, and its head "
        f"scored below {_MIN_HEAD_MATCH_SCORE:.0f}); the quote was paraphrased "
        f"rather than copied, so its position in the trace is not trustworthy: "
        f"{quoted_sentence!r}"
    )


def run_state_finder_with_sentence(reasoning: str, hint: str, quoted_sentence: str) -> tuple[str, str]:
    """Return (truncated_reasoning, quoted_sentence) for the given reasoning chain.

    Args:
        reasoning:        The full original reasoning chain from inference.
        hint:             Constructive hint generated by hint_script.
        quoted_sentence:  Verbatim sentence from the reasoning where the mistake
                          first occurred, as extracted by hint_script.

    Returns:
        Tuple of (truncated reasoning string, the quoted_sentence used).
    """
    truncated = _truncate_before_sentence(reasoning, quoted_sentence)
    if not truncated.strip():
        raise RuntimeError(
            "State finder truncation produced no content before the quoted sentence; "
            "the mistake may be at the very start of the reasoning chain."
        )
    tail_chars = len(reasoning) - len(truncated)
    if tail_chars < _MIN_TAIL_CHARS:
        raise RuntimeError(
            f"State finder kept {len(truncated)} of {len(reasoning)} chars: only "
            f"{tail_chars} chars follow the quoted sentence, under the "
            f"{_MIN_TAIL_CHARS}-char minimum. The quote named the final-answer line "
            f"rather than the mistake, so the teacher would be asked to re-emit an "
            f"answer instead of rewriting flawed reasoning: {quoted_sentence!r}"
        )
    _save_record(reasoning, hint, quoted_sentence, truncated)
    return truncated, quoted_sentence


def run_state_finder(reasoning: str, hint: str, quoted_sentence: str) -> str:
    """Return the reasoning chain truncated just before the mistake sentence.

    Args:
        reasoning:        The full original reasoning chain from inference.
        hint:             Constructive hint generated by hint_script.
        quoted_sentence:  Verbatim sentence from the reasoning where the mistake
                          first occurred, as extracted by hint_script.

    Returns:
        Truncated reasoning string containing only the correct prefix steps.
    """
    truncated, _ = run_state_finder_with_sentence(reasoning, hint, quoted_sentence)
    return truncated
