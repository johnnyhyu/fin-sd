"""Download TIGER-Lab/MMLU-Pro and lay it out the way the config-driven pipeline
expects, mirroring FinanceReasoning's `data/<dataset>/<subset>.json` convention.

Writes, under <data_dir>/MMLU-Pro/:
  - all.json                  : the full test split (subset "all")
  - <category>.json           : one file per category (subset = category name)
  - sampled.json              : a balanced sample of min(N, category size) questions
                                from every category (subset "sampled"), for a quick
                                run over the whole benchmark
  - validation.json           : few-shot CoT exemplars grouped by category

Run once before inference:  python prepare_data.py
"""

import argparse
import json
import os
import random

from datasets import load_dataset


def clean(record):
    """Drop the 'N/A' padding options MMLU-Pro uses to pad every question to 10
    choices, keeping only the real options."""
    record = dict(record)
    record["options"] = [opt for opt in record["options"] if opt != "N/A"]
    return record


def group_by_category(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["category"], []).append(row)
    return grouped


def balanced_sample(grouped, sample_per_category, seed):
    """Draw min(sample_per_category, len(rows)) questions from each category,
    concatenated into one list. Sampling is seeded so the subset is reproducible
    across runs."""
    rng = random.Random(seed)
    sampled = []
    for category in sorted(grouped):
        rows = grouped[category]
        k = min(sample_per_category, len(rows))
        sampled.extend(rng.sample(rows, k))
    return sampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="MMLU-Pro")
    parser.add_argument("--hf_repo", type=str, default="TIGER-Lab/MMLU-Pro")
    parser.add_argument(
        "--sample_per_category",
        type=int,
        default=100,
        help="Questions to draw per category for the balanced 'sampled' subset "
        "(min(this, category size)).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for the balanced 'sampled' subset."
    )
    parser.add_argument(
        "--exclude_categories",
        type=str,
        default="business,economics",
        help="Comma-separated categories to drop entirely (no per-category file, and "
        "excluded from 'all' and 'sampled').",
    )
    args = parser.parse_args()

    excluded = {
        c.strip() for c in args.exclude_categories.split(",") if c.strip()
    }

    out_dir = os.path.join(args.data_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    dataset = load_dataset(args.hf_repo)
    test_rows = [clean(r) for r in dataset["test"] if r["category"] not in excluded]
    val_rows = [clean(r) for r in dataset["validation"] if r["category"] not in excluded]
    if excluded:
        print(f"Excluding categories: {', '.join(sorted(excluded))}")

    # Full test split -> subset "all".
    all_path = os.path.join(out_dir, "all.json")
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(test_rows, f, indent=4, ensure_ascii=False)
    print(f"Wrote {len(test_rows)} questions -> {all_path}")

    # One file per category -> subset = category name.
    grouped = group_by_category(test_rows)
    for category, rows in grouped.items():
        cat_path = os.path.join(out_dir, f"{category}.json")
        with open(cat_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=4, ensure_ascii=False)
        print(f"Wrote {len(rows):5d} questions -> {cat_path}")

    # Balanced sample of min(N, category size) per category -> subset "sampled".
    sampled_rows = balanced_sample(grouped, args.sample_per_category, args.seed)
    sampled_path = os.path.join(out_dir, "sampled.json")
    with open(sampled_path, "w", encoding="utf-8") as f:
        json.dump(sampled_rows, f, indent=4, ensure_ascii=False)
    print(
        f"Wrote {len(sampled_rows)} questions "
        f"(<= {args.sample_per_category}/category) -> {sampled_path}"
    )

    # Few-shot CoT exemplars, grouped by category.
    shots = group_by_category(val_rows)
    shots_path = os.path.join(out_dir, "validation.json")
    with open(shots_path, "w", encoding="utf-8") as f:
        json.dump(shots, f, indent=4, ensure_ascii=False)
    print(f"Wrote few-shot exemplars for {len(shots)} categories -> {shots_path}")


if __name__ == "__main__":
    main()
