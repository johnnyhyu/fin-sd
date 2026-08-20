# FinTrust: A Comprehensive Benchmark of Trustworthiness Evaluation in Finance Domain

## Introduction

FinTrust is the first comprehensive benchmark designed specifically to evaluate the trustworthiness of Large Language Models (LLMs) in financial applications. As finance is a high-stakes domain with strict trustworthy standards, our benchmark provides a systematic framework to assess LLMs across seven critical dimensions: trustfulness, robustness, safety, fairness, privacy, transparency, and knowledge discovery.

Our benchmark comprises 15,680 answer pairs spanning textual, tabular, and time-series data. Unlike existing benchmarks that primarily focus on task completion, FinTrust evaluates alignment issues in practical contexts with fine-grained tasks for each dimension of trustworthiness.

## Dataset Access

📊 **HuggingFace Full Dataset**: [https://huggingface.co/datasets/HughieHu/FinTrust]

## Repository Structure

This repository contains the following directories:

- `fairness/`: Evaluates models' ability to provide unbiased responses 
- `knowledge_discovery/`: Tests models' capability to uncover non-trivial investment insights
- `privacy/`: Assesses resistance to information leakage
- `robustness/`: Examines models' resilience and ability to abstain when confidence is low
- `safety/`: Tests handling of various LLM attack strategies with financial crime scenarios
- `transparency/`: Evaluates disclosure of limitations and potential conflicts of interest
- `trustfulness/`: Measures models' accuracy and factuality in financial contexts

Each directory contains:
- `api_call.py`: Script to call the model API and generate responses
- A sample dataset of 100 test cases
- `postprocess_response.py`: Script to process and evaluate model responses

## Usage Instructions

FinTrust is run exactly like FinanceReasoning: a single `config/config.yaml` is the
source of truth, and each stage takes `--config config/config.yaml`.

### Configuration

`config/config.yaml` controls everything:

- `run:` — inference settings (model, global concurrency cap, dataset selection, privacy condition).
- `postprocess:` — evaluation settings (model to score, privacy condition, concurrency, dataset selection).
- `llms:` — the model catalog. Each key maps a friendly `model` name to an OpenRouter/vLLM route.

Secrets are referenced by name (`${OPENROUTER_API_KEY}`) and resolved from the
repo-root `.env`, so no keys live in the config. All models route through
OpenRouter (or a local vLLM endpoint), so the only required key is:

```
OPENROUTER_API_KEY=""   # in the repo-root .env
```

### Running Inference

Generate model responses for every dataset in one process under a single global
concurrency cap:

```bash
python inference.py --config config/config.yaml
# or: bash scripts/inference.sh
```

### Automated Evaluation

Post-process (judge/score) the responses written above. Evaluation finishes by
collapsing every judged/scored file into one headline metric table, so there is
no separate summarize step:

```bash
python evaluation.py --config config/config.yaml
# or: bash scripts/evaluation.sh
```

(`summarize_results.py` can still be run on its own to re-print the table without
re-evaluating, but `evaluation.py` already does this at the end.)

To change the model, dataset subset, or parallelism, edit `config/config.yaml`
(`run.model`, `run.only`, `run.global_parallel`, …) — no CLI flags or env vars
needed. `python inference.py --list` prints the valid dataset names for `only`.
