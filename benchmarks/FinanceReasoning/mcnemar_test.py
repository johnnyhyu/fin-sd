#!/usr/bin/env python3
"""
Run a McNemar test comparing two evaluation.json-style files.

Each file is a list of entries with a "question_id" and a per-question
correctness flag at result.acc (falling back to a top-level "acc").
Entries are paired by question_id; only IDs present in both files are used.

Usage:
    python mcnemar_test.py [flash.json] [self.json]

Defaults to flash.json and self.json in the current directory.
"""

import argparse
import json
import math
import sys


def load_acc(path: str) -> dict[str, int]:
    """Map question_id -> correctness (0/1) from an evaluation.json file."""
    with open(path) as f:
        entries = json.load(f)
    acc_by_id: dict[str, int] = {}
    for e in entries:
        qid = e.get("question_id")
        if qid is None:
            continue
        result = e.get("result")
        acc = result.get("acc") if isinstance(result, dict) else None
        if acc is None:
            acc = e.get("acc")
        if acc is None:
            continue
        acc_by_id[qid] = int(acc)
    return acc_by_id


def binom_two_sided_p(b: int, c: int) -> float:
    """Exact two-sided binomial p-value for McNemar (p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X <= k) under Binomial(n, 0.5)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def chi2_corrected_p(b: int, c: int) -> tuple[float, float]:
    """Continuity-corrected McNemar chi-square statistic and p-value (df=1)."""
    n = b + c
    if n == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / n
    # Survival function of chi-square with df=1: erfc(sqrt(stat/2))
    p = math.erfc(math.sqrt(stat / 2.0))
    return stat, p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("flash", nargs="?", default="flash.json",
                        help="First evaluation.json file (default: flash.json)")
    parser.add_argument("self_", nargs="?", default="self.json", metavar="self",
                        help="Second evaluation.json file (default: self.json)")
    args = parser.parse_args()

    flash = load_acc(args.flash)
    selfd = load_acc(args.self_)

    common = sorted(set(flash) & set(selfd))
    if not common:
        print("Error: no shared question_ids between the two files", file=sys.stderr)
        sys.exit(1)

    # Contingency table over discordant/concordant pairs
    a = b = c = d = 0  # a: both correct, b: flash correct/self wrong,
                       # c: flash wrong/self correct, d: both wrong
    for qid in common:
        f, s = flash[qid], selfd[qid]
        if f and s:
            a += 1
        elif f and not s:
            b += 1
        elif not f and s:
            c += 1
        else:
            d += 1

    flash_acc = (a + b) / len(common)
    self_acc = (a + c) / len(common)

    exact_p = binom_two_sided_p(b, c)
    stat, chi_p = chi2_corrected_p(b, c)

    print(f"flash: {args.flash} ({len(flash)} entries)")
    print(f"self:  {args.self_} ({len(selfd)} entries)")
    print(f"Paired question_ids: {len(common)}\n")

    print("Contingency table (paired by question_id):")
    print(f"{'':>20}{'self correct':>15}{'self wrong':>15}")
    print(f"{'flash correct':>20}{a:>15}{b:>15}")
    print(f"{'flash wrong':>20}{c:>15}{d:>15}\n")

    print(f"flash accuracy: {flash_acc:.4f} ({a + b}/{len(common)})")
    print(f"self  accuracy: {self_acc:.4f} ({a + c}/{len(common)})")
    print(f"Discordant pairs: b (flash-only correct)={b}, c (self-only correct)={c}\n")

    print("McNemar test (H0: the two models have equal error rates):")
    print(f"  Exact binomial (two-sided):        p = {exact_p:.6g}")
    print(f"  Chi-square w/ continuity corr.:    stat = {stat:.4f}, p = {chi_p:.6g}")
    print(f"\nRecommended: use the exact binomial p-value when b+c ({b + c}) is small (< ~25).")


if __name__ == "__main__":
    main()
