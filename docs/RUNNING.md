# Running

How to train an arm, serve the resulting checkpoint, and evaluate it. This
describes how to *operate* the code; it does not attempt to reproduce the
paper's tables. See [PAPER-MAP.md](PAPER-MAP.md) for where each claim lives.

## 1. Train

```bash
python launch.py --config configs/finsd.env
```

The preset selects the arm. Presets layer over `.env` and lose to the shell
environment, so a seed sweep is:

```bash
for s in 7 42 72 2 27; do
  SEED=$s python launch.py --config configs/finsd.env
done
```

Before training starts, the run logs the configuration it actually resolved to:

```
Resolved arm | hint=full writer=self (frozen teacher) | reverse KL over student
rollouts | L=100 | warmup=15 accum=16(eff batch 16) lr=2e-05 decay_floor=1 |
epochs=10 seed=7 | length_penalty=off
```

That line is the record of what a run was. Config arrives from three layers, so
the resolved values are the only reliable account of an arm.

### The arms

| Preset | Entrypoint | |
|---|---|---|
| `configs/finsd.env` | `launch.py` | The method. |
| `configs/placebo.env` | `launch.py` | Fixed information-free hint string (§5.3). |
| `configs/hint-120b.env` | `launch.py` | Hint written by GPT-OSS-120B (§6.1). Needs `OPENROUTER_API_KEY`. |
| `configs/forward-kl.env` | `launch.py` | Forward instead of reverse KL (§6.2). |
| `configs/rollout-8192.env` | `launch.py` | No rollout truncation (§6.3). Higher training memory. |
| `configs/opsd.env` | `python -m opsd.run` | OPSD baseline. Builds reference solutions first. |
| `configs/sft.env` | `python -m sft.run` | SFT baseline. Builds teacher traces first. |

Both baselines take `--config` the same way:

```bash
python -m opsd.run --config configs/opsd.env
python -m sft.run  --config configs/sft.env --limit 20   # smoke test
```

### What a run produces

Adapters land under `checkpoints/<run-id>/epoch_<N>/`, with `epoch_<N>_merged/`
holding the same weights merged into a standalone checkpoint. Only the newest
merged copy is kept — see [ARCHITECTURE.md](ARCHITECTURE.md).

Roughly 60 of the 220 problems fail per epoch, of which about 35 are arithmetic
slips that are dropped without a gradient, leaving ~25 training items per epoch.
That is expected: only wrong answers train, and the set is re-mined each epoch.

## 2. Serve a checkpoint

Evaluation runs against a served checkpoint, not against the training process.

```bash
python tools/serve_checkpoint.py --epoch 4                       # latest run
python tools/serve_checkpoint.py --run-id 20260717-1330 --epoch 4
python tools/serve_checkpoint.py --source opsd --epoch 2         # a baseline
```

This re-merges the epoch's adapter when its merged copy has been pruned. The
served name is what the benchmark configs must point at: set `model_id` in the
harness config to match, or the request 404s against a name the server does not
have.

Stop the training pipeline first — it owns port 8000 and will collide.

## 3. Evaluate

All three harnesses take the same shape: edit `config/config.yaml`, run
inference, then evaluation.

### FinanceReasoning — the target domain

The Hard subset, 238 problems. Point `llms.local-model1.model_id` at the served
checkpoint name, then:

```bash
cd benchmarks/FinanceReasoning
python inference.py  --config config/config.yaml
python evaluation.py --config config/config.yaml
```

Results land in `results/FinanceReasoning/hard/cot/<model_name>/` as
`inference.json` and `evaluation.json`. Answers are extracted by
`deepseek-v4-flash` through OpenRouter, tolerant of formatting and rounding.

**Results overwrite in place.** A second run against the same `model_name`
replaces the first. Give each arm and seed a distinct `model_name`, or copy the
directory out, if you want to keep more than one.

### MMLU-Pro — general reasoning

```bash
cd benchmarks/MMLU-Pro
python inference.py  --config config/config.yaml
python evaluation.py --config config/config.yaml
```

`subset` accepts `all`, `sampled`, or a single category.

The business and economics categories are **already excluded from the vendored
dataset**, which is how the paper keeps this evaluation out of domain. `all.json`
holds 10,399 questions across 12 categories rather than upstream's ~12,000 across
14, and the two category files are absent. Nothing needs excluding at run time;
`subset: all` is the paper's configuration.

The few-shot exemplar pool in `validation.json` still carries all 14 categories.
That is harmless — `inference.py` looks exemplars up by the question's own
category, so the two unused entries are never drawn.

### FinTrust — compliance

```bash
cd benchmarks/FinTrust
python inference.py  --config config/config.yaml
python evaluation.py --config config/config.yaml
python summarize_results.py
```

`python inference.py --list` prints the valid dataset names for the `only`
setting, if you want to run a single dimension. FinTrust writes its output into
the same directories as its input datasets; generated files carry an `output`
segment in the name and are gitignored.

## 4. Rebuild the corpus (optional)

The 220-problem corpus ships in `data/`. To regenerate it from the
FinanceReasoning function library, see [../data/build/](../data/build/). It
needs `OPENROUTER_API_KEY` and calls external models, so it will not reproduce
byte-for-byte.

## Troubleshooting

**A vLLM port collision at startup.** Something else is already serving. The
pipeline owns that process; do not start one by hand, and stop
`serve_checkpoint.py` before training.

**A 404 from the served model.** `model_id` in the harness config does not match
the server's `--served-model-name`. A checkpoint arm is served under its epoch
directory's name, so a base-model id silently fails.

**Expert LoRA refuses to load.** Expert targeting needs BF16 weights for both
the trainer and the served model. Use `unsloth/gpt-oss-20b-BF16`.

**A run wants a Weights & Biases login.** `WANDB_MODE` is not `disabled`. It
defaults to disabled; something in your `.env` or shell overrode it.
