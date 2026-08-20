# `pipeline/`

The core package. `run_pipeline.py` at the repo root is the training loop;
everything it needs lives here.

| Module | |
|---|---|
| `config.py` | Every knob, loaded from the environment. Resolves `configs/` presets before `.env`. See [../docs/CONFIG.md](../docs/CONFIG.md). |
| `prompts.py` | Harmony-format prompt rendering. Encodes through the same library vLLM serves with, so training and serving contexts are token-identical. |
| `inference_script.py` | Sample the student on a problem; parse reasoning and answer back out. |
| `eval_script.py` | The answer-equivalence judge. Raises rather than returning False on an ungradeable reply. |
| `hint_script.py` | §3.2 — locate the first faulty step, write the critique, classify the error as conceptual or arithmetic. |
| `state_finder_script.py` | Map the hint model's quoted clause to a character offset and truncate before it. |
| `probing_script.py` | Build the two contexts, sample the rollout, and extract logits. Also owns the HF model singleton and expert profiling. |
| `loss_script.py` | The divergences. `run_reverse_loss` is Eq. (4); `run_loss` is the forward-KL comparison. |
| `optimization_script.py` | AdamW, the LR schedule, gradient clipping, and adapter/merged-checkpoint saving. |
| `attention.py` | Picks the attention kernel. gpt-oss declares `_supports_sdpa = False`, so transformers falls back to `eager`, whose `[batch, heads, seq, seq]` score matrix costs ~30 GiB of transient VRAM per layer at 8192 tokens on the 120B. This is the one place that choice is made. |
| `expert_lora.py` | Memory-efficient LoRA over gpt-oss's fused MoE parameters. PEFT's `ParamWrapper` materialises the full delta for a `[128, 2880, 5760]` tensor — 18.1 GiB transient per layer on the 120B, against 0.69 GiB here. Falls back to stock PEFT if its layout stops matching. |
| `vllm_server.py` | The managed vLLM process: startup, health, adapter loading, and the epoch-boundary restart. |
| `endpoint.py`, `utils.py` | HTTP client with retries; GPU reservation, logging, model-family helpers. |

Read [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) before changing the
serving or GPU-placement paths — several constraints there are not obvious from
the code.
