import json
import argparse

from tqdm import tqdm
from loguru import logger

from utils.config import EvaluationConfig
from utils.evaluation_utils import (
    extract_answer_or_guess,
    get_acc,
    get_statistics,
    compute_sample_statistics,
)


def eval_cot(data):
    """Extract the predicted choice letter from each completion and score it
    against the ground-truth letter (`answer`). `execution_rate` records whether
    a letter was parsed without falling back to a random guess."""
    for record in tqdm(data, desc="Evaluating COT"):
        pred, extracted = extract_answer_or_guess(record.get("output"))
        record["result"] = {
            "execution_rate": int(extracted),
            "acc": get_acc(pred, record["answer"]),
            "extracted_answer": pred,
        }
    return data, get_statistics(data)


def eval_cot_samples(data, sampling_eval):
    """Score every sampled completion, then aggregate into avg@N ± CI, pass@k,
    and mean per-token entropy."""
    for record in tqdm(data, desc="Evaluating COT samples"):
        for sample in record["samples"]:
            pred, extracted = extract_answer_or_guess(sample.get("output"))
            sample["result"] = {
                "execution_rate": int(extracted),
                "acc": get_acc(pred, record["answer"]),
                "extracted_answer": pred,
            }
    return data, compute_sample_statistics(data, sampling_eval)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main():
    args = make_args()
    config = EvaluationConfig.from_yaml(args.config)
    with open(config.inference_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    sampling_eval = config.sampling_eval
    if sampling_eval is not None and sampling_eval.enabled:
        data, statistics = eval_cot_samples(data, sampling_eval)
    else:
        data, statistics = eval_cot(data)

    logger.info(f"Statistics: {statistics}")

    with open(config.evaluation_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
