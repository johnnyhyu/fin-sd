# Paper to code

Each claim in the paper, and the code that implements it. This is orientation —
the fastest way to find the thing you came for — not a verification harness.

Section numbers refer to *How Much Does the Hint Matter? Scope Conditions for
Hindsight Self-Distillation in Financial Reasoning* (ICAIF '26).

## Method (§3)

| Paper | Code |
|---|---|
| §3.1 — the three policies (student, frozen teacher, hint model) all being the student | `pipeline/config.py` §7. The teacher is the live model conditioned on the hint; `FROZEN_TEACHER` switches it to the base weights instead. |
| §3.1 — re-freezing the teacher each epoch | The epoch boundary in `run_pipeline.main`: the adapter is merged and vLLM restarted on it, so the teacher tracks the student. |
| §3.2 — error localization, Eq. (1) | `pipeline/hint_script.run_hint`. Returns the quoted clause, the critique, and the arithmetic/conceptual classification in one call. |
| §3.2 — hint content by mode | `pipeline/hint_script._STEP3_INSTRUCTIONS`. `HINT_MODE_FULL` is Fin-SD's; its style examples are the Figure 2 critique verbatim. |
| §3.2 — dropping arithmetic slips without a gradient | `SKIP_ARITHMETIC_ERRORS` in `pipeline/config.py`, applied in `run_pipeline.process_item`. Such items still count as incorrect in accuracy. |
| §3.2 footnote 1 — quote-to-offset via fuzzy match | `pipeline/state_finder_script._truncate_before_sentence`, using rapidfuzz `partial_ratio_alignment`. Leftmost match on ties, so a clause recurring after the error still cuts at the first occurrence. |
| §3.3 — truncation and the two contexts, Eq. (2) | `pipeline/state_finder_script.run_state_finder` produces τ<k; `pipeline/probing_script.run_probing` builds `I_stud` and `I_teach` from it. |
| §3.4 — the rollout, Eq. (3) | `pipeline/probing_script.sample_student_rollouts`. `L` is `MAX_PROBE_TOKENS`, default 100. |
| §3.4 — reverse KL, Eq. (4) | `pipeline/loss_script.run_reverse_loss`, full vocabulary at each position, teacher treated as a constant target, prefix positions masked out. |
| §6.2 — the forward-KL comparison | `pipeline/loss_script.run_loss`, selected by `STUDENT_ROLLOUT_FORWARD=1`. |
| Algorithm 1 | `run_pipeline.main` is the outer loop; `process_item` is lines 11–19; `flush_training_batch` is the accumulate-and-step block, lines 20–29. |
| Algorithm 1, lines 27–29 — flushing a partial batch | `optimization_script.flush_gradients`, rescaling by `A/a`. |

## Corpus (§4.1)

| Paper | Code |
|---|---|
| Function curation: 3,133 → 295 | `data/build/phase1_curate_functions.py`. Grades on the 8-point scale, keeps ≥ 3, then applies the viability judge. |
| Problem generation, three phases | `data/build/phase2_generate_problems.py`. |
| Failure mining | `data/build/phase3_build_trainingset.py` for the initial build; `run_pipeline.main` re-mines every epoch thereafter. |
| Answer equivalence judge | `pipeline/eval_script.run_eval`. Raises rather than returning False when the grader gives no verdict — an ungradeable reply is not evidence of a wrong answer. |
| The critique pass and manual review | Not code. Performed once during corpus construction; see [../data/DATASHEET.md](../data/DATASHEET.md). |
| The 220-problem corpus itself | `data/trainingset.json`. |

## Training configuration (§4.2, Table 1)

Every value in Table 1 is pinned in `configs/finsd.env` and is also the default
in `pipeline/config.py`. See [CONFIG.md](CONFIG.md) for the full table.

Two rows deserve a note:

- **Reasoning effort.** Runs were at harmony's default, **medium**.
  `pipeline/prompts.py` builds the system message with `SystemContent.new()`, and
  neither the vLLM serve command nor the evaluation harnesses send
  `reasoning_effort`, so there is no path by which effort is raised. Table 1 of
  the paper as first published says "High"; that is a known erratum being
  corrected. Nothing about the runs changes.
- **"Warmup steps: 15."** This is the default here. A shorter ramp measured
  better after the paper; see "post-paper findings" in [CONFIG.md](CONFIG.md).

## Evaluation (§4.3, §5)

| Paper | Code |
|---|---|
| FinanceReasoning Hard, 238 problems | `benchmarks/FinanceReasoning/`, `subset: hard`, `prompt_type: cot`. |
| Accuracy, completion rate, total tokens | `benchmarks/FinanceReasoning/evaluation.py`. |
| MMLU-Pro, business and economics excluded (§5.4) | `benchmarks/MMLU-Pro/`. Excluded at the data layer: the vendored `all.json` holds 10,399 questions across 12 categories, and the two category files are absent. `subset: all` is the paper's configuration. |
| FinTrust, fourteen metrics across seven dimensions (§5.4) | `benchmarks/FinTrust/`, summarized by its `summarize_results.py`. |
| The five seeds | `SEED`; the paper sweeps 7, 42, 72, 2, 27. |
| Means, standard deviations, t-based intervals (Tables 2, 5; Figure 3) | **Not in this repository.** Run outputs overwrite in place, so no per-seed archive survives; the reported statistics were assembled outside the repo. |

## Ablations (§6)

Each is a preset. See [RUNNING.md](RUNNING.md).

| Paper | Preset |
|---|---|
| §5.3 — placebo hint | `configs/placebo.env` |
| §6.1 — 120B hint writer | `configs/hint-120b.env` |
| §6.2 — forward KL | `configs/forward-kl.env` |
| §6.3 — no rollout truncation | `configs/rollout-8192.env` |

## Baselines (§5.2)

| Paper | Code |
|---|---|
| OPSD — teacher conditioned on a full reference solution | `opsd/`. Stage 1 writes references; stage 2 trains under forward KL. |
| SFT on reference solutions | `sft/`. Rejection-sampled teacher traces, plain next-token loss. |
| Base GPT-OSS-20B | No training. Serve the base model and evaluate. |

## Not in the paper

`contrib/big-finance-benchmark/` produces no result in the paper. It is a
modified fork of a separate benchmark, retained as future work — see its
`CHANGES.md`.

The `LENGTH_PENALTY` and `tuned` presets are post-paper. The knobs listed under
"experimental surface" in [CONFIG.md](CONFIG.md) are unreported paths.
