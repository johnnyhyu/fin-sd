#!/usr/bin/env python3
"""
Compare two models and extract problems where model1 got wrong (acc=0)
and model2 got right (acc=1).

Usage:
    python compare_models.py <model1> <model2> [--level hard] [--mode cot]
"""

import argparse
import json
import os
import sys


def load_eval(model: str, level: str, mode: str) -> dict[str, dict]:
    path = os.path.join(
        "results", "FinanceReasoning", level, mode, model, "evaluation.json"
    )
    if not os.path.exists(path):
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        entries = json.load(f)
    return {e["question_id"]: e for e in entries}


def extract_reasoning(entry: dict) -> str | None:
    if entry.get("reasoning_content"):
        return entry["reasoning_content"]
    try:
        return entry["raw_response"]["choices"][0]["message"]["reasoning"]
    except (KeyError, IndexError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model1", help="Model that got it wrong")
    parser.add_argument("model2", help="Model that got it right")
    parser.add_argument("--level", default="hard", help="Difficulty level (default: hard)")
    parser.add_argument("--mode", default="cot", help="Eval mode (default: cot)")
    parser.add_argument("--output", help="Output file path (default: results/<model1>_vs_<model2>_diff.json)")
    args = parser.parse_args()

    m1 = load_eval(args.model1, args.level, args.mode)
    m2 = load_eval(args.model2, args.level, args.mode)

    common_ids = set(m1.keys()) & set(m2.keys())
    print(f"{args.model1}: {len(m1)} entries")
    print(f"{args.model2}: {len(m2)} entries")
    print(f"Common question IDs: {len(common_ids)}")

    results = []
    for qid in sorted(common_ids):
        e1, e2 = m1[qid], m2[qid]
        acc1 = e1.get("result", {}).get("acc", e1.get("acc"))
        acc2 = e2.get("result", {}).get("acc", e2.get("acc"))
        if acc1 == 0 and acc2 == 1:
            results.append({
                "question_id": qid,
                "question": e1["question"],
                "context": e1["context"],
                "difficulty": e1.get("difficulty"),
                "ground_truth": e1.get("ground_truth"),
                args.model1: {
                    "output": e1.get("output"),
                    "reasoning": extract_reasoning(e1),
                    "extracted_answer": e1.get("result", {}).get("extracted_answer"),
                },
                args.model2: {
                    "output": e2.get("output"),
                    "reasoning": extract_reasoning(e2),
                    "extracted_answer": e2.get("result", {}).get("extracted_answer"),
                },
            })

    output_path = args.output or os.path.join(
        "results", f"{args.model1}_vs_{args.model2}_diff.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFound {len(results)} problems where {args.model1} failed and {args.model2} succeeded")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
