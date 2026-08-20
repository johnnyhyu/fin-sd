"""Answer extraction and scoring for MMLU-Pro (10-way multiple choice, A-J).

The regex extractors mirror the tiered strategy in the original MMLU-Pro
`compute_accuracy.py`; the sample-statistics helpers (`pass_at_k`,
`compute_sample_statistics`) mirror the FinanceReasoning pipeline so multi-sample
evaluation reports the same avg@N ± CI / pass@k / entropy summary.
"""

import re
import random
import numpy as np
from scipy import stats

CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

# Tiered extraction: prefer the explicit "the answer is (X)" template, then an
# "Answer: X" line, then fall back to the last standalone capital letter A-J.
_PATTERNS = (
    r"answer is \(?([A-J])\)?",
    r"[aA]nswer:\s*\(?([A-J])\)?",
    r"\b([A-J])\b(?!.*\b[A-J]\b)",
)

# Seeded RNG so the random fallback for un-extractable answers is reproducible
# (matches the original compute_accuracy.py behavior).
_RNG = random.Random(12345)


def extract_answer(text: str):
    """Return the predicted choice letter (A-J) or None if nothing matches."""
    if not text:
        return None
    for pattern in _PATTERNS:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
    return None


def extract_answer_or_guess(text: str):
    """Extract the predicted letter, falling back to a seeded random guess when
    no letter can be parsed. Returns (letter, extracted) where `extracted` is
    False when the letter came from the random fallback."""
    pred = extract_answer(text)
    if pred is None:
        return _RNG.choice(CHOICES), False
    return pred, True


def get_acc(prediction, ground_truth) -> int:
    """1 when the predicted letter matches the ground-truth letter, else 0."""
    if prediction is None:
        return 0
    return int(str(prediction).strip().upper() == str(ground_truth).strip().upper())


def get_statistics(data):
    """Overall and per-category accuracy plus execution (extraction) rate."""
    total_acc = 0
    total_execution = 0
    total_tokens = 0
    per_category = {}
    for record in data:
        acc = record["result"]["acc"]
        execution = record["result"]["execution_rate"]
        total_acc += acc
        total_execution += execution
        total_tokens += record.get("completion_tokens") or 0
        cat = record.get("category", "unknown")
        bucket = per_category.setdefault(cat, {"acc": 0, "count": 0})
        bucket["acc"] += acc
        bucket["count"] += 1
    n = max(len(data), 1)
    category_accuracy = {
        cat: round(b["acc"] / b["count"] * 100, 2) for cat, b in sorted(per_category.items())
    }
    return {
        "avg_accuracy": round(total_acc / n * 100, 2),
        "avg_execution_rate": round(total_execution / n * 100, 2),
        "total_tokens": total_tokens,
        "num_questions": len(data),
        "category_accuracy": category_accuracy,
    }


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from Chen et al. 2021 (Codex).

    n: number of samples drawn, c: number of correct samples, k: the k in pass@k.
    Returns NaN when k > n (pass@k is undefined without enough samples)."""
    if k > n:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def compute_sample_statistics(data, sampling_eval):
    """Aggregate per-question sample results into avg@N ± CI, pass@k, execution
    rate, and mean per-token entropy. Each record must carry a `samples` list
    whose entries have a `result` dict (`acc`, `execution_rate`) and an optional
    `mean_entropy`."""
    ci_pct = int(round(sampling_eval.ci_confidence * 100))
    n_ref = sampling_eval.num_samples
    # Always report pass@n (n = num_samples) alongside the configured k's.
    ks = sorted(set(sampling_eval.pass_at_k) | {n_ref})
    per_q_acc = []       # per-question mean accuracy c_i / n_i
    per_q_exec = []      # per-question mean execution rate
    pass_scores = {k: [] for k in ks}
    entropies = []

    for record in data:
        samples = record.get("samples", [])
        n_i = len(samples)
        if n_i == 0:
            continue
        c = sum(s["result"]["acc"] for s in samples)
        per_q_acc.append(c / n_i)
        per_q_exec.append(sum(s["result"]["execution_rate"] for s in samples) / n_i)
        for k in ks:
            pass_scores[k].append(pass_at_k(n_i, c, k))
        for s in samples:
            if s.get("mean_entropy") is not None:
                entropies.append(s["mean_entropy"])

    acc = np.array(per_q_acc, dtype=float)
    n_q = len(acc)
    avg = float(acc.mean()) if n_q else 0.0
    if n_q > 1:
        half_width = float(stats.sem(acc) * stats.t.ppf(0.5 + sampling_eval.ci_confidence / 2, n_q - 1))
    else:
        half_width = 0.0

    def _mean_pct(values):
        arr = np.array(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        return round(float(arr.mean()) * 100, 2) if arr.size else None

    statistics = {
        "num_questions": n_q,
        "num_samples_per_question": n_ref,
        f"avg@{n_ref}": round(avg * 100, 2),
        f"avg@{n_ref}_ci{ci_pct}_halfwidth": round(half_width * 100, 2),
        f"avg@{n_ref}_report": f"{avg * 100:.2f} ± {half_width * 100:.2f} (CI{ci_pct})",
        "pass@k": {f"pass@{k}": _mean_pct(pass_scores[k]) for k in ks},
        "avg_execution_rate": _mean_pct(per_q_exec),
        "mean_token_entropy": round(float(np.mean(entropies)), 4) if entropies else None,
        "num_samples_with_entropy": len(entropies),
    }
    return statistics
