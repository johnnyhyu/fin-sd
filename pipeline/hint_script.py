"""
hint_script.py — generates a constructive pedagogical hint for an incorrect answer.

Given the problem, reasoning chain, wrong answer, and correct answer,
asks a hint model to pinpoint the mistake and guide the student without revealing the answer.
The hint model is the fixed OpenRouter config.HINT_MODEL by default, or the live
vLLM policy (the current student weights) when config.HINT_VIA_VLLM is set.
Returns both the hint text and the verbatim quoted sentence where the mistake occurred.

The amount of guidance the hint reveals is controlled by `mode`:
  • "full"    — a 2-3 sentence hint that states the correct premise/assumption (most guidance).
  • "partial" — names only the concept that was misused, without giving the correct interpretation.
  • "concept" — surfaces the relevant concept itself: the verbatim text from the problem the
                student overlooked, or the applicable concept from standard practice.
  • "vague"   — states only whether the mistake was arithmetic or conceptual (least guidance).
  • "custom"  — a self-contained prompt (its own Steps 1-3 and output format) that
                tightens the "full" guidance with explicit leakage-calibration and
                causal-first-error rules; see _CUSTOM_SYSTEM.

The meta-mode "curriculum" selects a concrete mode per epoch (partial for the
first half of training, full for the second); see resolve_hint_mode().
"""
import json

from .utils import openrouter_call, vllm_call
from . import config
from . import inference_script

# Hint verbosity modes (see module docstring). The default lives in config.HINT_MODE.
HINT_MODE_FULL = "full"
HINT_MODE_PARTIAL = "partial"
HINT_MODE_CONCEPT = "concept"
HINT_MODE_VAGUE = "vague"
HINT_MODE_CUSTOM = "custom"
VALID_HINT_MODES = (
    HINT_MODE_FULL, HINT_MODE_PARTIAL, HINT_MODE_CONCEPT, HINT_MODE_VAGUE, HINT_MODE_CUSTOM,
)

# Curriculum is a meta-mode: it isn't a verbosity level itself but selects one
# per epoch (partial for the first half of training, full for the second half).
# resolve_hint_mode() turns it into a concrete verbosity mode before run_hint().
HINT_MODE_CURRICULUM = "curriculum"
VALID_CONFIG_HINT_MODES = VALID_HINT_MODES + (HINT_MODE_CURRICULUM,)


def resolve_hint_mode(mode: str, epoch: int, num_epochs: int) -> str:
    """Resolve a (possibly meta) hint mode into a concrete verbosity mode.

    "curriculum" ramps guidance over training: the first half of epochs use
    "partial" hints and the second half use "full". epoch is 1-based. All other
    modes pass through unchanged.
    """
    if mode == HINT_MODE_CURRICULUM:
        return HINT_MODE_PARTIAL if epoch <= num_epochs / 2 else HINT_MODE_FULL
    return mode

# Shared preamble: the intro plus Step 1 (locate the first error) and Step 2
# (classify). Every mode uses this identical block; only Step 3 (how much the
# hint reveals) differs per mode — see _STEP3_INSTRUCTIONS.
_SYSTEM_PREAMBLE = """\
You are an expert tutor diagnosing a student's incorrect solution to a finance problem.

You will receive:
1. The problem statement
2. The student's step-by-step reasoning chain
3. The student's (wrong) final answer
4. The correct ground-truth answer

## Step 1 — Locate the first error
Compare the student's reasoning against the ground truth and find the EARLIEST point where their solution diverges. Later errors are often downstream consequences of this first one — always anchor your diagnosis to the first divergence.

## Step 2 — Classify the error
- **conceptual**: wrong formula, wrong assumption, misread constraint, wrong inputs applied to a correct formula, missed condition, or a sign/direction error. If the student computed correctly but set up the wrong thing, it is conceptual.
- **arithmetic**: the setup, formula, and inputs are all correct, but a number was computed, transcribed, or rounded incorrectly.
When in doubt, classify as conceptual."""

# Mode-specific Step 3 — the only part that varies between modes. Each block is
# inserted between the shared preamble above and the shared output format below,
# and controls how much guidance the "hint" field reveals (see module docstring).
_STEP3_INSTRUCTIONS = {
    HINT_MODE_FULL: """\
## Step 3 — Write the hint (2–3 sentences)

**Conceptual errors** — use this two-part structure:
1. Diagnose: name the specific step or quantity where the logic diverged.
   ("Your logic diverges when defining ..." / "Your calculation for X incorrectly applies ...")
2. Correct the premise as a factual assumption with a reason:
   "Assume Y because Z." — state the correct rule, constraint, or interpretation and WHY it holds.

Style examples of correct conceptual hints:
- "Your logic diverges when defining the observed and expected values for the chi-square test. Assume the test must be evaluated using the frequency counts of loans in each category because the chi-square statistic applies to discrete counts, not continuous dollar amounts."
- "Your calculation for vacancy loss incorrectly applies the percentage to the total gross income, including parking and laundry. Assume vacancy loss is calculated only on the gross scheduled rent from the apartment units, excluding other income sources."

Hard constraints for conceptual hints:
- State the rule; never command the math step. Do NOT write "recalculate X", "multiply X by Y", "solve for X", or any procedural instruction.
- Do not walk through the corrected computation.

**Arithmetic errors:**
- Confirm their setup/logic is correct, then point them to the specific step whose number is off.
  (e.g., "Your setup for Step 2 is correct, but double-check your arithmetic when compounding the interest.")
- Do not state the corrected number.

**All hints:**
- Never reveal or imply the correct final answer or any corrected intermediate value.
- Be concise and direct: 2–3 sentences, no praise, no filler.""",
    HINT_MODE_PARTIAL: """\
## Step 3 — Write the hint (one sentence)

**Conceptual errors:**
- Name ONLY the specific concept, formula, assumption, or constraint the student misused (e.g., "You misapplied the compounding-frequency convention").
- Do NOT state the correct premise, interpretation, or assumption, and do NOT explain how to fix it. Identify the misused concept and nothing more.

**Arithmetic errors:**
- State only that the setup is correct but a calculation/rounding slip occurred, and name the step where it happened. Do not perform or hint at the corrected math.

**All hints:**
- Never reveal or imply the correct final answer or any corrected intermediate value.
- Keep the hint to a single sentence, no praise, no filler.""",
    HINT_MODE_CONCEPT: """\
## Step 3 — Write the hint (one sentence or a short quoted excerpt)

Surface the relevant concept the student needed — NOT a diagnosis of their mistake.

**Conceptual errors:**
- If the problem statement itself contains the specific text the student overlooked or misread (a stated value, definition, constraint, or condition), quote that text VERBATIM from the problem.
- Otherwise, if the needed concept comes from standard financial practice rather than the problem text (a convention, formula, or definition the student was expected to know — e.g., a day-count convention, compounding-frequency rule, or tax treatment), state that standard concept plainly.
- Provide ONLY the relevant problem text or standard concept — do NOT diagnose the student's error, state the correct premise, or explain how to fix it.

**Arithmetic errors:**
- State only that the setup is correct but a calculation/rounding slip occurred, and name the step where it happened. Do not perform or hint at the corrected math.

**All hints:**
- Never reveal or imply the correct final answer or any corrected intermediate value.
- Keep the hint to a single sentence or a short quoted excerpt, no praise, no filler.""",
    HINT_MODE_VAGUE: """\
## Step 3 — Write the hint (fixed phrasing)

State ONLY the category of the error and nothing else:
- If the mistake is conceptual, the hint must be exactly: "Your mistake is conceptual."
- If the mistake is arithmetic/rounding, the hint must be exactly: "Your mistake is arithmetic."
- Do NOT identify the step, concept, formula, or assumption involved, and do NOT explain how to fix it.
- Never reveal or imply the correct final answer.""",
}

_OUTPUT_FORMAT = """\
## Output
Respond with ONLY this JSON object (no markdown fences, no extra text):
{
  "error_type": "conceptual" | "arithmetic",
  "hint": "<hint following the rules above>",
  "quoted_sentence": "<the single sentence copied VERBATIM from the student's reasoning where the error first occurs — exact characters, including any typos or notation; do not paraphrase, trim, or merge sentences>"
}
"""


# "custom" is a fully self-contained system prompt: unlike the other modes it is
# not a Step 3 block slotted into the shared preamble/output-format scaffold, but
# supplies its own Steps 1-3 and output format. It still emits the same JSON
# schema as _OUTPUT_FORMAT, so run_hint() parses it unchanged.
_CUSTOM_SYSTEM = """\
You are an expert tutor diagnosing a student's incorrect solution to a finance
problem. Your diagnosis will be delivered as a short hint to a smaller, less
capable model, which uses it to revise its own reasoning. Two consequences follow
and govern everything below: (a) the hint must stay inside the concepts,
quantities, and notation the student already used — do not introduce new formulas,
terminology, or solution methods the student did not invoke; and (b) the hint must
never let the correct answer be reconstructed.

You will receive:
1. The problem statement
2. The student's step-by-step reasoning chain
3. The student's (wrong) final answer
4. The correct ground-truth answer

Treat the ground-truth answer as authoritative even if you would approach the
problem differently.

Step 1 — Locate the first error
Compare the student's reasoning against the ground truth and find the EARLIEST
point where their solution diverges in a way that is causally responsible for the
wrong answer. Do NOT flag stylistic choices or valid alternative approaches that
would still reach the correct answer. Later errors are usually downstream
consequences of this first one — anchor your diagnosis to the first divergence
ONLY, and ignore every later error in your hint.

Step 2 — Classify the error
* conceptual: wrong formula, wrong assumption, misread constraint, wrong inputs
  applied to a correct formula, missed condition, or a sign/direction error. If the
  student computed correctly but set up the wrong thing, it is conceptual.
* arithmetic: the setup, formula, and inputs are all correct, but a number was
  computed, transcribed, or rounded incorrectly.
Distinguish these on their merits. Only when the evidence is genuinely ambiguous,
default to conceptual.

Step 3 — Write the hint (2–3 sentences)

Conceptual errors — use this two-part structure:
1. Diagnose: name the specific step or quantity where the logic diverged. ("Your
   logic diverges when defining ..." / "Your calculation for X incorrectly
   applies ...")
2. Correct the premise as a factual assumption with a reason: "Assume Y because Z."
   — state the correct rule, constraint, or interpretation and WHY it holds.

Style examples of correct conceptual hints:
* "Your logic diverges when defining the observed and expected values for the
  chi-square test. Assume the test must be evaluated using the frequency counts of
  loans in each category because the chi-square statistic applies to discrete
  counts, not continuous dollar amounts."
* "Your calculation for vacancy loss incorrectly applies the percentage to the
  total gross income, including parking and laundry. Assume vacancy loss is
  calculated only on the gross scheduled rent from the apartment units, excluding
  other income sources."

Leakage calibration — a stronger diagnosis makes it easy to leak. Contrast:
* LEAKY (do NOT do this): "Your discount rate is wrong; assume it should be the
  after-tax cost of debt of 8%." — hands over the corrected input.
* SAFE: "Your discount rate incorrectly uses the pre-tax cost of debt. Assume the
  cash flows must be discounted at the after-tax cost of debt because interest is
  tax-deductible, which lowers the effective financing cost." — states the rule and
  why, no number.

Hard constraints for conceptual hints:
* State the rule; never command the math step. Do NOT write "recalculate X",
  "multiply X by Y", "solve for X", or any procedural instruction.
* Do not walk through the corrected computation.
* If the corrected premise you are about to state is a single trivial operation
  away from the answer, generalize it up to the underlying rule instead.

Arithmetic errors:
* Confirm their setup/logic is correct, then point them to the specific step whose
  number is off. (e.g., "Your setup for Step 2 is correct, but double-check your
  arithmetic when compounding the interest.")
* Do not state the corrected number.

All hints:
* Use only the student's own concepts, quantities, and notation. Introduce nothing
  new — no formula, method, or term the student did not already use.
* Never reveal or imply the correct final answer or any corrected intermediate
  value. Do not state or imply its numeric value, sign, magnitude, or order of
  magnitude, and do not narrow the solution to a single obvious value.
* Be concise and direct: 2–3 sentences, no praise, no filler, no restatement of
  the problem.
* Before emitting, verify the hint does not allow the answer to be reconstructed in
  one step. If it does, generalize it further.

Quoted sentence:
* Copy the single sentence VERBATIM from the student's reasoning where the error
  first occurs — exact characters, including any typos, notation, or formatting. Do
  NOT normalize, correct, paraphrase, trim, or merge sentences, even where the text
  is obviously wrong. Reproduce it as written.

Output
Respond with ONLY this JSON object (no markdown fences, no extra text):
{
  "error_type": "conceptual" | "arithmetic",
  "hint": "<hint following the rules above>",
  "quoted_sentence": "<the single sentence copied VERBATIM from the student's reasoning where the error first occurs — exact characters, including any typos or notation; do not paraphrase, trim, or merge sentences>"
}"""


def _build_system(mode: str) -> str:
    """Assemble the system prompt for the requested hint mode."""
    if mode == HINT_MODE_CUSTOM:
        return _CUSTOM_SYSTEM
    return _SYSTEM_PREAMBLE + "\n\n" + _STEP3_INSTRUCTIONS[mode] + "\n\n" + _OUTPUT_FORMAT


# Classification value (error_type) that marks a calculation/rounding slip rather
# than a flawed setup. run_pipeline.py uses this to exclude such items from training.
ARITHMETIC_ERROR_TYPE = "arithmetic"


def run_hint(
    problem: str,
    reasoning: str,
    generated_answer: str,
    ground_truth: str,
    mode: str | None = None,
) -> tuple[str, str, str]:
    """Return (hint, quoted_sentence, error_type).

    quoted_sentence is copied verbatim from reasoning. error_type is the model's
    classification of the mistake — ARITHMETIC_ERROR_TYPE for a calculation/rounding
    slip, otherwise "conceptual" (the default if the field is missing/unrecognised).

    mode controls how much the hint reveals (see module docstring): "full",
    "partial", or "vague". Defaults to config.HINT_MODE.
    """
    if mode is None:
        mode = config.HINT_MODE
    if mode not in VALID_HINT_MODES:
        raise ValueError(
            f"hint_script: unknown hint mode {mode!r}; expected one of {VALID_HINT_MODES}"
        )
    user_content = (
        f"**Problem:**\n{problem}\n\n"
        f"**Student's Reasoning:**\n{reasoning}\n\n"
        f"**Student's Answer (incorrect):** {generated_answer}\n\n"
        f"**Correct Answer:** {ground_truth}\n\n"
    )
    messages = [
        {"role": "system", "content": _build_system(mode)},
        {"role": "user", "content": user_content},
    ]
    if config.HINT_VIA_VLLM:
        # Self-generated hint from the live served policy (the current student
        # weights), rather than the fixed external HINT_MODEL. Bounded by
        # MAX_NEW_TOKENS since the local server is capped at VLLM_MAX_MODEL_LEN.
        hint_model = inference_script.get_active_model() or config.VLLM_MODEL
        raw = vllm_call(
            messages, model=hint_model, temperature=0.3,
            max_tokens=config.MAX_NEW_TOKENS,
        )
    else:
        raw = openrouter_call(
            messages, model=config.HINT_MODEL, temperature=0.3, max_tokens=8192,
        )
    # Strip optional markdown code fences the model may wrap around the JSON.
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1]
        stripped = stripped.rsplit("```", 1)[0]
    try:
        data = json.loads(stripped)
        hint = str(data["hint"])
        quoted_sentence = str(data["quoted_sentence"])
    except Exception as exc:
        raise RuntimeError(
            f"hint_script: could not parse JSON response — {exc}\nRaw output:\n{raw}"
        ) from exc
    # error_type is advisory (used only to gate training), so normalise leniently
    # and default to "conceptual" rather than failing the whole item.
    error_type = str(data.get("error_type", "conceptual")).strip().lower()
    if error_type != ARITHMETIC_ERROR_TYPE:
        error_type = "conceptual"
    return hint, quoted_sentence, error_type
