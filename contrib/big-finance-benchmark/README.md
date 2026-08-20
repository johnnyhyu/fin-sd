# Big Finance Harness

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Dataset: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-lightgrey.svg)](data/LICENSE-DATA)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue.svg)](pyproject.toml)
[![Tests](https://github.com/Rogo-Technologies/big-finance-benchmark/actions/workflows/test.yml/badge.svg)](https://github.com/Rogo-Technologies/big-finance-benchmark/actions/workflows/test.yml)

Reference scaffold for evaluating LLM agents on the **Big Finance** benchmark — 928
workflow-grounded financial-research questions, each paired with an expert-authored
rubric and a reference answer.

This harness reproduces the headline numbers from the companion paper,
[BigFinanceBench: A Workflow-Grounded Benchmark for Financial-Research Agents](https://arxiv.org/abs/2606.03829).
It is deliberately minimal: a ReAct loop, four publicly-replicable tools, and a
unified message format that runs the same scaffold across any model accessible
through [LiteLLM](https://github.com/BerriAI/litellm).

Maintained by [Rogo Technologies](https://rogo.ai). Contact: open a
[GitHub issue](https://github.com/Rogo-Technologies/big-finance-benchmark/issues)
or email `alexwang@rogo.ai`.

## What's here

| | |
|---|---|
| `big_finance_harness/` | Python package: ReAct agent, tools, judge, types |
| `scripts/` | Orchestrator (eval + grade), analysis, plotting |
| `tests/` | Test suite (127 tests, no network deps) |
| `data/` | Public 50-item subset (`big_finance_subset.jsonl`) + datasheet |
| `AUDIT.md` | Integration audit: findings, fixes, and the decisions behind them |

## Tools

The four tools given to the agent (plus a terminal `final_answer`):

| Tool | Backed by |
|---|---|
| `web_search` | SerpAPI (preferred) or Tavily (fallback) |
| `edgar_search` | SEC EDGAR public REST API |
| `fetch_url` | httpx + BeautifulSoup + BM25 (optional in-document retrieval) + PyMuPDF (PDFs) |
| `python_exec` | sandboxed subprocess (5s timeout) |
| `final_answer` | terminator |

We deliberately exclude: vector-store retrieval, premium financial data sources
(FactSet, CapIQ, Bloomberg, etc.), broker research, and provider-specific affordances
(native web search, tool-search, deferred-loading, model grounding). Every model gets
the same surface so the evaluation measures the model, not the scaffold.

**Contamination guard.** The public subset and the paper are indexed by search engines,
and agents find them: in testing, a search on the question text returned the Hugging Face
dataset as the top organic result and the agent read the rubric out of the page.
`web_search` therefore drops results pointing at this benchmark's own dataset, repo, or
paper (reporting the count in its output), and `fetch_url` refuses those URLs — including
after a redirect. Ordinary sources are untouched. Set `BFH_ALLOW_BENCHMARK_SOURCES=1` to
disable; never do that for a scored run. See `big_finance_harness/contamination.py`.

## Install

Requires Python ≥ 3.11.

```bash
python3 -m venv .venv
.venv/bin/pip install -e .              # core eval + grade
.venv/bin/pip install -e ".[analysis]"  # add pandas + matplotlib for build_plots.py
.venv/bin/pip install -e ".[dev]"       # add pytest + ruff for development
```

Set environment variables. Every model — closed and open frontier, and both judges —
routes through OpenRouter, so a single key covers the whole lineup:

```bash
# All model + judge calls route through OpenRouter:
export OPENROUTER_API_KEY=...

# Web search (one of the two):
export SERP_API_KEY=...      # SerpAPI (preferred)
export TAVILY_API_KEY=...    # Tavily (fallback)

# SEC EDGAR requires a User-Agent on every request:
export SEC_EDGAR_USER_AGENT="Your Name your@email.com"
```

`inference.py`, `evaluation.py`, and `run_eval_set.py` verify these before the first API
call and refuse to start otherwise. The search keys matter most: without one the run still
*completes*, it just produces traces where every retrieval failed. For a self-hosted
route, preflight also checks the server is reachable and serving the configured
`model_id`. Bypass with `BFH_SKIP_PREFLIGHT=1`.

## Dataset

Each row is one item conforming to `DatasetItem` in `big_finance_harness/types.py`:

```json
{
  "id": "bf-4eb39b2c53",
  "query": "If I take Dayforce's management adjusted reported EBIT...",
  "reference_answer": "Overstated by $90.1m...",
  "rubric": [
    {"text": "Identifies DAY as ticker", "points": 1},
    {"text": "Identifies Fiscal Year Ended December 31 2024", "points": 2}
  ]
}
```

The publicly-released $50$-item subset is bundled in `data/big_finance_subset.jsonl`,
licensed CC BY 4.0, and mirrored on Hugging Face at
[`RogoAI/big-finance-benchmark`](https://huggingface.co/datasets/RogoAI/big-finance-benchmark).
See [`data/README.md`](data/README.md) for schema and provenance, and
[`data/DATASHEET.md`](data/DATASHEET.md) for the full datasheet. The held-back
remainder of the benchmark is available on request through the maintainer; place
it at `data/big_finance_full.jsonl` to swap into the commands below.

## Quickstart

A small end-to-end run on five questions, one model, one judge:

```bash
.venv/bin/python scripts/run_eval_set.py \
  --dataset data/big_finance_subset.jsonl \
  --run-id quickstart \
  --kind dry_run \
  --sample-n 5 \
  --judge openrouter:google/gemini-3.1-pro-preview
```

Output goes to `runs/quickstart/`:
- `manifest.json` — config, dataset hash, model list
- `<model_label>.traces.jsonl` — full ReAct trajectories
- `<model_label>.grades.jsonl` — judge verdicts per (question, rubric line)

For the headline run, see `scripts/run_eval_set.py --help` for all flags;
relevant ones: `--n-trials`, `--judge` (multiple), `--concurrency`,
`--grade-concurrency`, `--skip-model`, `--judge-alias`.

## Local (self-hosted vLLM) models

The `local-model1` entry in `config/config.yaml` points the harness at an
OpenAI-compatible vLLM server. Serve the checkpoint with tool calling enabled — the
harness is a tool-use loop, so a server without a tool-call parser is useless to it:

```bash
vllm serve <checkpoint-or-hf-id> \
  --served-model-name epoch_1 \
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --reasoning-parser openai_gptoss
```

Serving a checkpoint from this repo's pipeline goes through `tools/serve_checkpoint.py`,
which forwards raw flags via `VLLM_SERVE_EXTRA_ARGS`:

```bash
VLLM_SERVE_EXTRA_ARGS="--enable-auto-tool-choice --tool-call-parser openai --reasoning-parser openai_gptoss" \
  python tools/serve_checkpoint.py --source sft --epoch 2
```

Without `--tool-call-parser openai` the server never reports tool calls at all — the
harmony `commentary to=functions.*` message is parsed as ordinary content — and every
question ends in `no_tool_call` with no answer.

Then point `llms.local-model1.model_id` at the `--served-model-name` and run
`inference.py --config config/config.yaml` as usual.

One gpt-oss-on-vLLM quirk the harness works around, configured per-route in
`config/config.yaml`:

- **`extra_body.stop_token_ids`** — vLLM's chat-completions path drops the harmony stop
  tokens it registers for gpt-oss, so generation runs past `<|call|>` and the server
  500s parsing its own output (`HarmonyError: Unexpected token ... while expecting start
  token 200006`) on *every* tool-enabled request. Local routes therefore send
  `stop_token_ids: [200012, 200002]` by default.

Serve the checkpoint with a `--max-model-len` that comfortably fits
`inference.max_output_tokens` (65536 by default) plus the transcript — vLLM clamps an
over-large `max_tokens` to whatever context remains rather than rejecting it, so a
generation cut mid-harmony-message trips the same parser.

## Reproduce the paper's headline numbers

The paper's Table 1 was produced by:

```bash
# 1. Eval + grade across all default models with two judges
.venv/bin/python scripts/run_eval_set.py \
  --dataset data/big_finance_full.jsonl \
  --run-id headline \
  --kind headline \
  --n-trials 3 \
  --judge openrouter:google/gemini-3.1-pro-preview \
  --judge openrouter:anthropic/claude-opus-4.7

# 2. Backfill missing costs (models + judge snapshots that LiteLLM has no rate
#    table for)
.venv/bin/python scripts/recompute_costs.py --run-dir runs/headline

# 3. Build the long-form analysis CSVs and per-question metadata
.venv/bin/python scripts/build_analysis_csv.py \
  --run-dir runs/headline \
  --dataset data/big_finance_full.jsonl \
  --out-dir runs/headline/analysis

# 4. Headline accuracy table with bootstrap CIs and inter-judge kappa
.venv/bin/python scripts/headline_table.py \
  --per-grade-csv runs/headline/analysis/per_grade.csv \
  --out-dir runs/headline/analysis

# 5. Plots
.venv/bin/python scripts/build_plots.py \
  --analysis-dir runs/headline/analysis \
  --out-dir runs/headline/analysis/plots
```

## Methodology

- **Sampling**: no temperature is sent by default, so each provider applies its own —
  and a local vLLM route uses **1.0**. Pass `--temperature 0` (or set `temperature: 0`
  under `inference:`) for a reproducible comparison; the value is recorded in the
  manifest under `config.temperature`. This section previously claimed `temperature=0`
  while nothing in the harness ever sent one, which means past single-trial A/Bs were
  measuring the sampler as much as the weights. The default stays "unset" rather than 0
  so already recorded runs remain comparable — set it explicitly, on **both** arms.
- **Terminal tool**: `final_answer` by default. gpt-oss on vLLM emits
  `<|channel|>final_answer<|message|>…`, writing the tool name where harmony expects a
  channel; vLLM's parser then drops the message whole and the turn arrives empty, which
  cost gpt-oss-20b roughly a third of its runs on the 50-question subset. Set
  `BFH_TERMINAL_TOOL_NAME=submit_answer` for any gpt-oss arm — and for whatever it is
  compared against, so the tool surface matches. The name is interpolated into the
  system prompt and both empty-turn nudges automatically.
- **Replaying reasoning**: off by default. `BFH_REPLAY_REASONING=1` sends
  `reasoning_content` back with each assistant turn so vLLM's harmony path can rebuild
  the analysis message instead of replaying a history in which the model answered
  without thinking. It changes the conditioning, so it is a knob, not a fix — set it for
  both arms or neither.
- **Step budget**: 50 turns by default (`--max-steps`).
- **Trials**: each (question, model) pair runs 3 times.
- **Judges**: default two-judge panel (Gemini 3.1 Pro Preview + Claude Opus 4.7);
  per-rubric and final-answer scoring returned in one structured response. We report
  the two-judge mean and inter-judge Cohen's κ alongside accuracy. Both judges also
  appear in the evaluated lineup; averaging across two different model families is
  intended to limit any single-family self-preference, and the high inter-judge κ is
  the check on it. When both judges return the same constant verdict on every item, κ is
  undefined; `inter_judge_kappa.csv` flags those rows with `kappa_undefined`.
- **Reasoning in the graded trace**: reasoning models return an empty `content` on
  intermediate turns and put their analysis in a separate `reasoning_content` field. The
  harness captures it on each `StepRecord` and renders it in the trace the judge sees —
  without it, rubric lines of the form "Identifies X" read as unsatisfied for exactly the
  models that reason most. Set `BFH_JUDGE_INCLUDE_REASONING=0` to grade on assistant text
  alone.
- **Run health**: each model's phase prints its stop-reason histogram, answer rate, and
  tool-error rate, and stores them in the manifest under
  `results.eval[].trace_summary`. A run where retrieval was broken produces traces and
  reports zero errors, so the counts alone cannot tell you it failed — check the answer
  rate before grading.
- **Resumption**: keyed on `(question_id, trial_idx, judge)`; errored traces
  re-run, terminal states (`final_answer`, `max_steps`, `no_tool_call`,
  `context_exceeded`, `token_budget`) are treated as complete.
- **Snapshots**: model IDs without a date suffix emit a warning; the trace still
  captures the resolved snapshot returned by the provider via
  `RunRecord.resolved_model`. Dependencies are pinned in `pyproject.toml`.
- **Costs**: LiteLLM-reported `cost_usd` is authoritative when present.
  `recompute_costs.py` fills missing values from a pinned rate table (open models
  on the eval side; preview snapshots on the judge side that LiteLLM prices
  incompletely). Verify the table against current OpenRouter rates before
  publishing.
- **`python_exec` is not a sandbox.** It's a subprocess with a 5-second timeout
  and no filesystem, network, or syscall isolation. Users running untrusted
  prompts should run the harness inside a container with `--network=none
  --read-only` and a tightened seccomp profile.

## Contamination policy

Only the 50-item public subset under `data/big_finance_subset.jsonl` is
released publicly; the remaining 878 items are held back to support periodic
contamination re-evaluation. The public subset is a stratified sample of the
full benchmark — see [`data/README.md`](data/README.md) and
[`data/DATASHEET.md`](data/DATASHEET.md) for stratification details. Held-back
access for academic evaluation is mediated through the maintenance contact in
the intro; the held-back items should not be posted publicly or used as
training data.

## Citation

If you use this benchmark or harness, please cite the paper
([arXiv:2606.03829](https://arxiv.org/abs/2606.03829)):

```bibtex
@misc{bigfinancebench2026,
  title         = {BigFinanceBench: A Workflow-Grounded Benchmark for Financial-Research Agents},
  author        = {Wang, Alex and Meinhardt, Georg and Katz, Jacob and Kim, Joseph H. and Chaudhary, Pratyush K. and Blagden, Chase and Xu, Eric},
  year          = {2026},
  eprint        = {2606.03829},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}
```

## License

Apache 2.0. See [`LICENSE`](LICENSE). The bundled 50-item dataset subset under
`data/` is licensed separately under CC BY 4.0; see [`data/LICENSE-DATA`](data/LICENSE-DATA).
