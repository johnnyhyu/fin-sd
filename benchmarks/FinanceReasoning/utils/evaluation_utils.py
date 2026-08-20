"""
This code is adapted from:
https://github.com/yale-nlp/FinanceMath/blob/main/utils/evaluation_utils.py

Original repository: https://github.com/yale-nlp/FinanceMath
"""

import re
import numpy as np
from scipy import stats
from utils.llm import LLM
from loguru import logger

def within_eps(pred: float, gt: float):
    eps = abs(gt) * 0.002
    if pred >= gt - eps and pred <= gt + eps:
        return True
    else:
        return False

def is_number(string):
    pattern = r'^[-+]?(\d{1,3}(,\d{3})*|(\d+))(\.\d+)?$'
    match = re.match(pattern, string)
    return bool(match)

def is_scientific_number(string):
    pattern = r'^[-+]?\d+(\.\d+)?e[-]?\d+$'
    match = re.match(pattern, string)
    return bool(match)

def contain_num_and_str(string):
    pattern_str = r'[a-zA-Z]'
    pattern_num = r'[0-9]'
    return bool(re.search(pattern_str, string) and re.search(pattern_num, string))

def normalize(prediction: str):
    try:
        prediction = eval(prediction)
    except Exception:
    # Preprocessing the string [Stage 1]
        prediction = prediction.strip().lower()
        prediction = prediction.rstrip('.')

        for money in ["£", "€", "¥", "million", "billion", "thousand", "us", "usd", "rmb"]:
            prediction = prediction.replace(money, '')
            
        # Replace special tokens
        if '=' in prediction:
            prediction = prediction.split('=')[-1].strip()
        if '≈' in prediction:
            prediction = prediction.split('≈')[-1].strip()
        if '`' in prediction:
            prediction = prediction.replace('`', '')
        if '%' in prediction:
            prediction = prediction.replace('%', '')
        if '$' in prediction:
            prediction = prediction.replace('$', '')
        if '°' in prediction:
            prediction = prediction.replace('°', '')

        # Detect the boolean keyword in the generation
        if prediction in ['true', 'yes', 'false', 'no']:
            if prediction == 'true' or prediction == 'yes':
                prediction = 'True'
            else:
                prediction = 'False'
        if 'true' in prediction or 'false' in prediction:
            prediction = 'True' if 'true' in prediction else 'False'

        # Detect the approximation keyword
        if 'approximately' in prediction:
            prediction = prediction.replace('approximately', '').strip()
        if ' or ' in prediction:
            prediction = prediction.split(' or ')[0]

        # Drop the units before and after the number
        if re.match(r'[-+]?(?:[\d,]*\.*\d+) [^0-9 ]+$', prediction):
            prediction = re.search(r'([-+]?(?:[\d,]*\.*\d+)) [^0-9 ]+$', prediction).group(1)
        if re.match(r'[^0-9 ]+ [-+]?(?:[\d,]*\.*\d+)$', prediction):
            prediction = re.search(r'[^0-9 ]+ ([-+]?(?:[\d,]*\.*\d+))$', prediction).group(1)
        if re.match(r'[-+]?(?:[\d,]*\.*\d+)[^\d]{1,2}$', prediction):
            prediction = re.search(r'([-+]?(?:[\d,]*\.*\d+))[^\d]{1,2}$', prediction).group(1)
        if re.match(r'[^-+\d]{1,2}(?:[\d,]*\.*\d+)$', prediction):
            prediction = re.search(r'[^-+\d]{1,2}((?:[\d,]*\.*\d+))$', prediction).group(1)

        # Preprocessing the number [Stage 1]
        if '10^' in prediction:
            prediction = re.sub(r'10\^(-?\d+)', r'math.pow(10, \1)', prediction)
        if ' x ' in prediction:
            prediction = prediction.replace(' x ', '*')
        if ' × ' in prediction:
            prediction = prediction.replace(' × ', '*')
        if is_number(prediction):
            prediction = prediction.replace(',', '')

        # If the prediction is empty, use dummy '0'
        if not prediction:
            prediction = "None" 

        try:
            prediction = eval(prediction)
        except Exception:
            prediction = None

        # Check the type of the prediction

    return prediction

def get_acc(prediction, gt):
    try:
        assert isinstance(gt, (int, float, bool)), type(gt)
        if isinstance(prediction, str):
            prediction = normalize(prediction)
        if isinstance(prediction, (tuple, list)):
            prediction = prediction[0]
        if prediction is None:
            return 0
        elif isinstance(gt, bool) or isinstance(prediction, bool):
            return int(prediction == gt)
        else:
            return int(within_eps(prediction, gt))
    except Exception as e:
        logger.warning(f"Error while comparing prediction: {prediction} and ground truth: {gt}, {e}")
        return 0
    
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from Chen et al. 2021 (Codex).

    n: number of samples drawn, c: number of correct samples, k: the k in pass@k.
    Returns NaN when k > n (pass@k is undefined without enough samples)."""
    if k > n:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def compute_sample_statistics(data, sampling_eval):
    """Aggregate per-question sample results into avg@N ± CI, pass@k, execution
    rate, and mean per-token entropy. Each record must carry a `samples` list
    whose entries have a `result` dict (`acc`, `execution_rate`) and an optional
    `mean_entropy`."""
    ci_pct = int(round(sampling_eval.ci_confidence * 100))
    n_ref = sampling_eval.num_samples
    # Always report pass@n (n = num_samples) alongside the configured k's.
    ks = sorted(set(sampling_eval.pass_at_k) | {n_ref})
    per_q_acc = []       # per-question mean accuracy c_i / n_i
    per_q_exec = []      # per-question mean execution rate
    pass_scores = {k: [] for k in ks}
    entropies = []

    for record in data:
        samples = record.get("samples", [])
        n_i = len(samples)
        if n_i == 0:
            continue
        c = sum(s["result"]["acc"] for s in samples)
        per_q_acc.append(c / n_i)
        per_q_exec.append(sum(s["result"]["execution_rate"] for s in samples) / n_i)
        for k in ks:
            pass_scores[k].append(pass_at_k(n_i, c, k))
        for s in samples:
            if s.get("mean_entropy") is not None:
                entropies.append(s["mean_entropy"])

    acc = np.array(per_q_acc, dtype=float)
    n_q = len(acc)
    avg = float(acc.mean()) if n_q else 0.0
    if n_q > 1:
        half_width = float(stats.sem(acc) * stats.t.ppf(0.5 + sampling_eval.ci_confidence / 2, n_q - 1))
    else:
        half_width = 0.0

    def _mean_pct(values):
        arr = np.array(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        return round(float(arr.mean()) * 100, 2) if arr.size else None

    statistics = {
        "num_questions": n_q,
        "num_samples_per_question": n_ref,
        f"avg@{n_ref}": round(avg * 100, 2),
        f"avg@{n_ref}_ci{ci_pct}_halfwidth": round(half_width * 100, 2),
        f"avg@{n_ref}_report": f"{avg * 100:.2f} ± {half_width * 100:.2f} (CI{ci_pct})",
        "pass@k": {f"pass@{k}": _mean_pct(pass_scores[k]) for k in ks},
        "avg_execution_rate": _mean_pct(per_q_exec),
        "mean_token_entropy": round(float(np.mean(entropies)), 4) if entropies else None,
        "num_samples_with_entropy": len(entropies),
    }
    return statistics


def extract_cot_answers(data, ans_extract_model: LLM):
    system_prompt = """Extract the final answer of the question as a numeric value from the given solution. If you cannot extract an answer, return "None".

You should either return "None" or a numeric value without any additional words."""
    user_inputs = []
    for record in data:
        user_inputs.append(f"Question: {record['question']}\nSolution: {record['output']}")
    prompts = ans_extract_model.apply_chat_template(
        [system_prompt] * len(data),
        user_inputs
    )
    results = ans_extract_model.batch_generate(prompts, desc="Extracting COT answers")
    return results

def extract_pot_answers(output):
    # this heuristic is not perfect, if you have a better heuristic, please submit a PR, thanks!
    if not output or 'argparse' in output:
        return ''
    tmp = re.findall(r"```python(.*?)```", output, re.DOTALL)
    if len(tmp) > 0:
        processed_output = tmp[0].strip("\n")
    else:
        tmp = re.findall(r"```(.*?)```", output, re.DOTALL)
        if len(tmp) > 0:
            processed_output = tmp[0].strip("\n")
        else:
            tmp = re.findall(r"```", output, re.DOTALL)
            if len(tmp) == 1 and 'def solution():' not in output:
                if len(output) > 4 and output[:4] == '    ':
                    processed_output = "def solution():\n" + output.split("```")[0]
                else:
                    processed_output = "def solution():\n    " + output.split("```")[0]
            else:
                if 'def solution():' not in output and len(output) > 4 and output[:4] == '    ':
                    processed_output = "def solution():\n" + output
                elif 'def solution():' not in output:
                    processed_output = "def solution():\n    " + output
                else:
                    processed_output = output.strip()
    processed_output = processed_output.strip("```")
    processed_output = processed_output.strip()
    return processed_output