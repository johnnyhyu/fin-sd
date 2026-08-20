# Fin-SD

Reference implementation and corpus for **"How Much Does the Hint Matter? Scope
Conditions for Hindsight Self-Distillation in Financial Reasoning"** (ICAIF '26).

Fin-SD is a hindsight self-distillation recipe for multi-step reasoning. On a
problem the student gets wrong, the student itself locates the first faulty step
of its own trajectory and writes a short critique of it. The trajectory is
truncated immediately *before* that step, so what remains is a correct, on-policy
prefix. A frozen copy of the student is then conditioned on that same prefix plus
the critique, and distilled into the student under reverse KL over a 100-token
rollout. No external model contributes a gradient.

```
python launch.py --config configs/finsd.env
```

## Requirements

Two H100-class GPUs for the 20B student the paper uses: one serves the model
through vLLM, one trains it. The pipeline places them itself and keeps them
disjoint. An `OPENROUTER_API_KEY` is needed for the answer-equivalence judge.

The repository's bare default is the 120B, which needs eight cards.
`configs/finsd.env` pins the 20B, and every preset extends it — so always pass
`--config`.

```bash
./setup.sh                       # venv + dependencies
cp .env.example .env             # then fill in OPENROUTER_API_KEY
python launch.py --config configs/finsd.env
```

Weights & Biases is off by default; a run needs no account. See `.env.example`.

## Layout

| Path | |
|---|---|
| `run_pipeline.py` | The training loop — Algorithm 1 in the paper. |
| `launch.py` | Entrypoint. Places GPUs, then runs locally or dispatches to Modal. |
| `pipeline/` | Core package: hint synthesis, error localization, probing, losses, optimization, and the managed vLLM server. |
| `configs/` | One preset per arm in the paper. Start here. |
| `data/` | The 220-problem corpus, and `data/build/` — the chain that produced it. |
| `opsd/`, `sft/` | The two training baselines the paper compares against. |
| `benchmarks/` | The three vendored evaluation harnesses. |
| `docs/` | Reproduction, configuration, architecture, and a paper-to-code map. |
| `tools/` | Checkpoint serving for evaluation, plus the expert profiler. |
| `tests/` | `python -m pytest tests/` — 110 tests, no GPU or network needed. |
| `visualizers/` | Static HTML viewers for run output. |
| `contrib/` | A modified fork used by no result in the paper. See its `CHANGES.md`. |

## Choosing an arm

Every configuration in the paper is a preset. `configs/finsd.env` pins every
value in Table 1 and is the base the others extend.

| Preset | |
|---|---|
| `finsd.env` | **The method.** Self-written critique, reverse KL, L = 100. |
| `placebo.env` | §5.3 — the critique replaced by a fixed, information-free string. |
| `hint-120b.env` | §6.1 — the hint written by GPT-OSS-120B instead of by the student. |
| `forward-kl.env` | §6.2 — forward KL instead of reverse. |
| `rollout-8192.env` | §6.3 — no rollout truncation. |
| `opsd.env`, `sft.env` | The two baselines. Run through their own entrypoints (opsd.run, sft.run; NOT launch.py). |

Presets layer over `.env` and lose to the shell environment, so a seed sweep is
just `SEED=42 python launch.py --config configs/finsd.env`. Every run prints the
arm it resolved to before training starts.

Two presets are **not** from the paper and are labelled as such:
`length-penalty.env` and `tuned.env`. See [docs/CONFIG.md](docs/CONFIG.md).

## Documentation

- **[docs/RUNNING.md](docs/RUNNING.md)** — train an arm, serve a checkpoint, run
  each of the three evaluations.
- **[docs/CONFIG.md](docs/CONFIG.md)** — every knob, split into what the paper
  used, what is infrastructure, and what is unreported experimental surface.
- **[docs/PAPER-MAP.md](docs/PAPER-MAP.md)** — each claim in the paper to the
  code that implements it.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — how serving, training and
  the epoch boundary fit together. Read before changing the GPU or vLLM setup.
- **[data/DATASHEET.md](data/DATASHEET.md)** — corpus provenance and limitations.

## Citing

See [CITATION.cff](CITATION.cff).

## License

Apache 2.0 for the code ([LICENSE](LICENSE)); CC BY 4.0 for the corpus
([data/LICENSE-DATA](data/LICENSE-DATA)). This distribution vendors modified
copies of four third-party projects, each under its own upstream license with
its modifications recorded — see [NOTICE](NOTICE).
