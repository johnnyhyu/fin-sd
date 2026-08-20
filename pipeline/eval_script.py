"""
eval_script.py — compares the model's generated answer against ground truth.

Uses OpenRouter so we get semantic equivalence checking (e.g. "$21,600" == "21600").
Returns True if correct, False otherwise.
"""
import re

from .utils import logger, openrouter_call
from . import config

_SYSTEM = r"""
You are a strict grader. Your sole task is to determine if the LLM Answer is equivalent to the Expected Answer.

First, classify the Expected Answer:

CASE A — the Expected Answer is numeric (a number, possibly with units, currency, or percent):
1. EXTRACT & NORMALIZE: Isolate the core numerical values from both the Expected Answer and the LLM Answer.
   - Strip away all currency symbols ($), commas, markdown, and spaces.
   - Ignore scale words like "million", "billion", or "B". e.g. 1.5 billion = 1500000000 OR 1.5.
2. CALCULATE: Let E be the normalized Expected Answer and L be the normalized LLM Answer. Calculate the error using the formula:
   $$\text{Error } = \frac{|L - E|}{|E|} \times 100$$
   - Special case: if E is exactly 0, do not divide by zero. Instead they are equivalent if and only if |L| <= 0.000001.
3. EVALUATE:
   - If the Error is less than or equal to 0.2, they are equivalent.
   - If the Error is greater than 0.2, they are NOT equivalent.
   e.g. 1001 and 1000 is equivalent (0.1 error), but 1003 is not (0.3 error).

CASE B — the Expected Answer is a word or phrase (not a number):
They are equivalent if and only if the LLM Answer expresses the same meaning as the Expected Answer (ignore case, punctuation, and phrasing differences; the LLM Answer must not contradict or omit the substance of the Expected Answer).

Output Format:
Reply with exactly '1' if they are equivalent, or '0' if they are not. Do not include any other text, explanation, or markdown in your final response. Only output the single digit.
"""


# The grader is told to reply with a bare '1' or '0', but models wrap it anyway —
# a markdown fence, a "**0**", a trailing period. A naive startswith('1') reads
# every one of those as INCORRECT, which is the expensive direction to be wrong in:
# the item is then hinted, probed and trained on, i.e. the pipeline distils a
# correction for an answer that was already right. Pull the last standalone 0/1 out
# of the reply instead (last, so any preamble the model emits before its verdict
# doesn't win) and treat a reply with no verdict at all as an explicit failure
# rather than a silent "wrong".
_VERDICT_RE = re.compile(r"[01]")


def run_eval(generated_answer: str, ground_truth: str) -> bool:
    """Return True if generated_answer matches ground_truth semantically.

    Raises RuntimeError if the grader's reply contains no 0/1 verdict — the caller
    (run_pipeline.process_item) turns that into a skipped item, which is right:
    an ungradeable response is not evidence that the answer was wrong.
    """
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"Expected answer: {ground_truth}\n"
                f"LLM answer: {generated_answer}"
            ),
        },
    ]
    result = openrouter_call(
        messages, model=config.EVAL_MODEL, temperature=0.0, max_tokens=8192
    )
    digits = _VERDICT_RE.findall(result)
    if not digits:
        raise RuntimeError(
            f"Grader ({config.EVAL_MODEL}) returned no 0/1 verdict: {result[:200]!r}"
        )
    if result.strip() not in ("0", "1"):
        logger.debug("Grader reply was not a bare digit (%r); read verdict as %s.",
                     result[:120], digits[-1])
    return digits[-1] == "1"