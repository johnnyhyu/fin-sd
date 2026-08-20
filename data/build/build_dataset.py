#!/usr/bin/env python3
"""
Master runner for the dataset build chain.

Runs the three phases in order, each of which resumes from whatever it already
wrote under data/, so a partial run can be re-invoked safely:

    Phase 1  phase1_curate_functions   functions-article-all.json → function-library.json
    Phase 2  phase2_generate_problems  function-library.json      → problemset.json
    Phase 3  phase3_build_trainingset  problemset.json            → trainingset.json

Usage:
    OPENROUTER_API_KEY=... python scripts/build_dataset.py            # all phases
    python scripts/build_dataset.py --from-phase 2                    # skip phase 1
    python scripts/build_dataset.py --only 3                          # just phase 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import phase1_curate_functions as phase1  # noqa: E402
import phase2_generate_problems as phase2  # noqa: E402
import phase3_build_trainingset as phase3  # noqa: E402

PHASES = {
    1: ("curate the function library", phase1.main),
    2: ("generate & extract problems", phase2.main),
    3: ("solve, grade & assemble training set", phase3.main),
}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Build data/trainingset.json end to end.")
    parser.add_argument("--from-phase", type=int, choices=[1, 2, 3], default=1,
                        help="Start at this phase and run through phase 3 (default: 1).")
    parser.add_argument("--only", type=int, choices=[1, 2, 3],
                        help="Run just this one phase.")
    args = parser.parse_args(argv)

    selected = [args.only] if args.only else [n for n in (1, 2, 3) if n >= args.from_phase]

    for n in selected:
        label, run = PHASES[n]
        print(f"\n########## PHASE {n} — {label} ##########", flush=True)
        run([])

    print("\nAll requested phases complete.")


if __name__ == "__main__":
    main()
