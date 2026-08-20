"""Pool two evaluated MMLU-Pro runs into one A/B table.

Both arms answer the SAME questions in the same order, so the comparison is
paired: the interesting quantity is not the difference of two independent
accuracies but the split of the questions the two arms disagree on. McNemar's
test uses exactly that split (b = control right / treatment wrong, c = the
reverse) and ignores the questions both got right or both got wrong, which is
why it needs far less data than an unpaired two-proportion test to see the same
effect. The naive unpaired CI is reported too, since that is the number people
expect to see next to a benchmark result.

Also reported per arm, because on a reasoning model they move the accuracy and a
difference in either one is a finding rather than noise:
  - extraction rate: answers where a letter was actually parsed. The rest are
    scored as a seeded random guess (utils/evaluation_utils.extract_answer_or_guess),
    so a low rate quietly converts to ~10% accuracy on 10-way choice.
  - truncation rate: completions that hit max_tokens. gpt-oss spends the whole
    budget on a runaway reasoning chain and returns EMPTY content, which lands in
    the guess bucket above.

Usage:
    python scripts/compare_runs.py \
        results/MMLU-Pro/all/cot/base-gpt-oss-20b \
        results/MMLU-Pro/all/cot/ckpt-0727-e1 \
        --max-tokens 32768
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.evaluation_utils import get_statistics


def load(run_dir):
    with open(os.path.join(run_dir, "evaluation.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def arm_summary(data, max_tokens):
    stats = get_statistics(data)
    n = max(len(data), 1)
    truncated = sum(1 for r in data if (r.get("completion_tokens") or 0) >= max_tokens)
    empty = sum(1 for r in data if not (r.get("output") or "").strip())
    stats["truncation_rate"] = round(truncated / n * 100, 2)
    stats["empty_output_rate"] = round(empty / n * 100, 2)
    stats["mean_completion_tokens"] = round(stats["total_tokens"] / n, 1)
    return stats


def mcnemar(control, treatment):
    """Paired disagreement counts plus a two-sided exact-binomial p-value.

    Exact rather than the chi-square approximation: the discordant pairs on a
    single category can be few, and that is where the approximation is worst.
    """
    b = c = 0  # b: control right, treatment wrong.  c: treatment right, control wrong.
    for x, y in zip(control, treatment):
        xa, ya = x["result"]["acc"], y["result"]["acc"]
        b += xa == 1 and ya == 0
        c += ya == 1 and xa == 0
    n = b + c
    if n == 0:
        return b, c, 1.0
    # Two-sided exact binomial test against p=0.5 on the discordant pairs.
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return b, c, min(1.0, 2 * tail)


def wilson(k, n, z=1.96):
    """Wilson score interval — behaves at the extremes where the normal
    approximation puts the bound outside [0, 1]."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half) * 100, (centre + half) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("control_dir")
    ap.add_argument("treatment_dir")
    ap.add_argument("--max-tokens", type=int, default=32768)
    args = ap.parse_args()

    ctl, trt = load(args.control_dir), load(args.treatment_dir)
    if len(ctl) != len(trt):
        raise SystemExit(f"arms differ in size: {len(ctl)} vs {len(trt)}")
    for x, y in zip(ctl, trt):
        if x["question_id"] != y["question_id"]:
            raise SystemExit("arms are not aligned question-for-question")

    ctl_name = os.path.basename(args.control_dir.rstrip("/"))
    trt_name = os.path.basename(args.treatment_dir.rstrip("/"))
    cs, ts = arm_summary(ctl, args.max_tokens), arm_summary(trt, args.max_tokens)
    n = len(ctl)

    print(f"\nMMLU-Pro — {n} questions, paired\n")
    row = "{:<22} {:>14} {:>14} {:>10}"
    print(row.format("", ctl_name, trt_name, "delta"))
    print("-" * 64)
    for label, key, unit in [
        ("accuracy %", "avg_accuracy", ""),
        ("extraction rate %", "avg_execution_rate", ""),
        ("truncation rate %", "truncation_rate", ""),
        ("empty output %", "empty_output_rate", ""),
        ("mean gen tokens", "mean_completion_tokens", ""),
    ]:
        d = ts[key] - cs[key]
        print(row.format(label + unit, f"{cs[key]:.2f}", f"{ts[key]:.2f}", f"{d:+.2f}"))

    lo_c, hi_c = wilson(round(cs["avg_accuracy"] / 100 * n), n)
    lo_t, hi_t = wilson(round(ts["avg_accuracy"] / 100 * n), n)
    print(f"\n95% CI (Wilson):  {ctl_name} [{lo_c:.2f}, {hi_c:.2f}]   "
          f"{trt_name} [{lo_t:.2f}, {hi_t:.2f}]")

    b, c, p = mcnemar(ctl, trt)
    print(f"\nPaired disagreements: {ctl_name}-only correct b={b}, "
          f"{trt_name}-only correct c={c}, both-agree {n - b - c}")
    print(f"McNemar exact two-sided p = {p:.3g}"
          f"{'  (significant at 0.05)' if p < 0.05 else '  (not significant at 0.05)'}")

    print("\nPer category:")
    crow = "{:<20} {:>6} {:>10} {:>10} {:>9}"
    print(crow.format("category", "n", ctl_name[:10], trt_name[:10], "delta"))
    print("-" * 60)
    counts = {}
    for r in ctl:
        counts[r.get("category", "unknown")] = counts.get(r.get("category", "unknown"), 0) + 1
    for cat in sorted(cs["category_accuracy"]):
        a, t = cs["category_accuracy"][cat], ts["category_accuracy"][cat]
        print(crow.format(cat, counts[cat], f"{a:.2f}", f"{t:.2f}", f"{t - a:+.2f}"))
    print()


if __name__ == "__main__":
    main()
