# Corpus construction

The chain that produced `../trainingset.json`, described in §4.1 of the paper.

**You do not need to run this** to train or evaluate — the finished corpus ships
in `../`. Run it to inspect how the corpus was made, or to build a new one from
a different function library.

It calls GPT-OSS-120B, DeepSeek-V4-Flash and GPT-5.5 through OpenRouter, so it
needs `OPENROUTER_API_KEY`, it costs money, and it will not reproduce the
existing corpus byte for byte.

```bash
python data/build/build_dataset.py                 # all phases
python data/build/build_dataset.py --from-phase 2  # skip phase 1
python data/build/build_dataset.py --only 3        # just phase 3
```

Each phase is one file whose steps run in order, and every step resumes from
whatever it already wrote under `../`, so a partial run can be re-invoked safely.
A phase can also be run directly, with `--only STEP` for a single step.

## Phases

**1. `phase1_curate_functions.py`** — the 3,133-function FinanceReasoning
library → 295 usable functions.

| Step | Output |
|---|---|
| `grade_functions` — LLM difficulty grade 1–8 | `functions-graded.json` |
| `filter_grade3plus` — keep grade ≥ 3 | `functions-hard.json` |
| `filter_grade5` — LLM viability judge | `functions-hard-filtered.json` |
| `build_function_library` — join back to source | `function-library.json` |

Reads the seed library from `benchmarks/FinanceReasoning/`, not from a copy, so
there is one library in the tree and its provenance is explicit.

**2. `phase2_generate_problems.py`** — functions → problems.

| Step | Output |
|---|---|
| `generate_problemset` — LLM writes problems | `extra_problemset.json` |
| `extract_problems` — parse | `problemset.json` |

**3. `phase3_build_trainingset.py`** — problems → the training corpus.

| Step | Output |
|---|---|
| `run_inference` — model answers each problem | `answers.json` |
| `grade_answers` — LLM equivalence judge | `grades.json` |
| `create_trainingset` | `trainingset.json` |

Optional QA, not part of the master run: `validation.py`, which writes
`feedback.json`.

## What the paper adds on top

The critique pass with Claude Fable 5 and the manual review of every item for
logical validity and answer uniqueness were performed once, by hand, and are not
part of this chain. The shipped corpus reflects them; a fresh run of the chain
will not. See [../DATASHEET.md](../DATASHEET.md).
