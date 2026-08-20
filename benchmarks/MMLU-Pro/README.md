# MMLU-Pro

|[**🤗 Dataset**](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | [**🏆Leaderboard**](https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro) | [**📖 Paper**](https://arxiv.org/abs/2406.01574) |

This repo contains the evaluation code for the NeurIPS-24 paper "[MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark](https://arxiv.org/abs/2406.01574.pdf)"

## Introduction
We introduce MMLU-Pro, an enhanced benchmark designed to evaluate language understanding models across broader and more challenging tasks. Building on the Massive Multitask Language Understanding (MMLU) dataset, MMLU-Pro integrates more challenging, reasoning-focused questions and increases the answer choices per question from four to ten, significantly raising the difficulty and reducing the chance of success through random guessing. MMLU-Pro comprises over 12,000 rigorously curated questions from academic exams and textbooks, spanning 14 diverse domains including Biology, Business, Chemistry, Computer Science, Economics, Engineering, Health, History, Law, Math, Philosophy, Physics, Psychology, and Others.  

Our experimental results show that MMLU-Pro not only raises the challenge, causing a significant drop in accuracy by 16% to 33% compared to MMLU but also demonstrates greater stability under varying prompts. With 24 different prompt styles tested, the sensitivity of model scores to prompt variations decreased from 4-5% in MMLU to just 2% in MMLU-Pro. Additionally, we found that models utilizing Chain of Thought (CoT) reasoning achieved better performance on MMLU-Pro compared to direct answering, which starkly contrasts the findings on the original MMLU, indicating that MMLU-Pro includes more complex reasoning questions. 

<img width="1432" alt="abs" src="https://github.com/TIGER-AI-Lab/MMLU-Pro/assets/20929360/8e369fc2-5b6b-4bab-8a44-9e222e742027">

## Updates
**October 10, 2024**: Added the 24 tested prompt styles from the paper to the repository.

## Dataset Creation
MMLU-Pro was created to provide language models with a more challenging and robust benchmark, pushing the boundaries of what these models can achieve in terms of expert-level knowledge and reasoning. Please refer to our huggingface [**🤗 Dataset**](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) for more details.

## Evaluation

This benchmark runs through the same config-driven pipeline as `FinanceReasoning`:
a single `config/config.yaml` describes the model, subset, and prompt, and every
model call is routed through OpenRouter (hosted) or a local vLLM server
(self-hosted). Set `OPENROUTER_API_KEY` (and, for local models, `VLLM_API_KEY`)
in the repo-root `.env`.

**1. Prepare the data** (downloads `TIGER-Lab/MMLU-Pro` into `data/MMLU-Pro/`):

```bash
python prepare_data.py            # or: sh scripts/prepare_data.sh
```

This writes `all.json` (full test split), one `<category>.json` per category,
`sampled.json` (a balanced sample of `min(100, category size)` questions from every
category — a quick pass over the whole benchmark), and `validation.json` (few-shot
CoT exemplars grouped by category). Tune the sample size with
`--sample_per_category` (and `--seed` for reproducibility), e.g.
`python prepare_data.py --sample_per_category 50`.

**2. Edit `config/config.yaml`** — pick the model under test (`inference.model_name`,
matching a key under `llms`), the `subset` (`all`, `sampled`, or a single category
such as `law`, `math`, `physics`, ...), and `num_shots`. Add or adjust `llms`
entries as needed; each must point at OpenRouter or a local vLLM `base_url`.

**3. Run inference, then evaluation:**

```bash
python inference.py  --config config/config.yaml   # or: sh scripts/inference.sh
python evaluation.py --config config/config.yaml   # or: sh scripts/evaluation.sh
```

Results land under `results/MMLU-Pro/<subset>/<prompt_type>/<model_name>/` as
`inference.json` (raw completions) and `evaluation.json` (per-question scores).
`evaluation.py` logs overall and per-category accuracy.

**Multi-sample evaluation** (avg@N ± CI, pass@k, mean per-token entropy): set
`sampling_eval.enabled: true` in the config and rerun inference then evaluation.

### Legacy scripts

The original standalone scripts (`evaluate_from_local.py`, `evaluate_from_api.py`,
`evaluate_from_apiX.py`, `compute_accuracy.py`, `evalshowpro.py`, and
`scripts/examples/`) remain in the repo for reference and are unchanged.
## 🏆 Mini-Leaderboard
| Model                          | Overall Accuracy | 
|--------------------------------|:----------------:|
| Claude-3.5-Sonnet              | 76.12            |
| GPT-4o                         | 72.55            | 
| Gemini-1.5-Pro                 | 69.03            |
| Claude-3-Opus                  | 68.45            |
| GPT-4-Turbo                    | 63.71            | 
| Gemini-1.5-Flash               | 59.12            |
| Yi-large                       | 57.53            |
| Claude-3-Sonnet                | 56.80            |
| Llama-3-70B-Instruct           | 56.20            |
| Phi3-medium-4k                 | 55.70            |
| Deepseek-V2-Chat               | 54.81            |
| Phi-3-medium-4k-instruct       | 53.48            |
| Llama-3-70B                    | 52.78            |
| Qwen1.5-72B-Chat               | 52.64            |
| Yi-1.5-34B-Chat                | 52.29            |
| Phi3-medium-128k               | 51.91            |
| MAmmoTH2-8x7B-Plus             | 50.40            |

For more details on various models and their accuracy across different subjects, please visit our [**Leaderboard**](https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro).

## Benchmarking Answer Extraction
We provide different alternatives to do answer extraction. We found that different answer extraction mechanisms have minor impact on the results.
```
python compute_accuracy.py results/llama-3-8b-quantized/CoT/all/
```

Thanks to @chibop1 for evaluating the robustness of MMLU-Pro across all the different answer extraction strategies and temperature. A detailed discussion is posted at [Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1e4eyoi/mmlu_pro_how_different_parameters_and_regex/).

## Contact
- Yubo Wang: y726wang@uwaterloo.ca
- Xueguang Ma: x93ma@uwaterloo.ca
- Wenhu Chen: wenhuchen@uwaterloo.ca

## Citation

**BibTeX:**
```bibtex
@article{wang2024mmlu,
  title={Mmlu-pro: A more robust and challenging multi-task language understanding benchmark},
  author={Wang, Yubo and Ma, Xueguang and Zhang, Ge and Ni, Yuansheng and Chandra, Abhranil and Guo, Shiguang and Ren, Weiming and Arulraj, Aaran and He, Xuan and Jiang, Ziyan and others},
  journal={arXiv preprint arXiv:2406.01574},
  year={2024}
}
```
