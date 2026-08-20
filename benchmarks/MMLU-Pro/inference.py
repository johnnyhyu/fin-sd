import json
import os
import argparse

from utils.llm import LLM
from utils.config import InferenceConfig

CHOICE_MAP = "ABCDEFGHIJ"


def format_example(question, options, cot_content=""):
    """Render one MMLU-Pro question as `Question / Options / Answer` text. With
    `cot_content` this is a solved few-shot exemplar; without it, it is the target
    question ending in the answer prefix supplied by the caller."""
    example = f"Question: {question}\nOptions: "
    for i, opt in enumerate(options):
        example += f"{CHOICE_MAP[i]}. {opt}\n"
    if cot_content:
        cot_content = cot_content[3:] if cot_content.startswith("A: ") else cot_content
        example += f"Answer: {cot_content}\n\n"
    return example


def prepare_inputs(data, config: InferenceConfig):
    """Build (system, user) text for every question. The system message states
    the per-question subject/format; the user message stacks few-shot CoT
    exemplars (from the record's own category) then the target question."""
    prompt_template = config.prompt.template
    answer_prefix = prompt_template["answer_prefix"]

    shots_by_category = {}
    if config.num_shots > 0 and os.path.exists(config.shots_file):
        with open(config.shots_file, "r", encoding="utf-8") as f:
            shots_by_category = json.load(f)

    system_inputs, user_inputs = [], []
    for record in data:
        subject = record.get("category", "knowledge")
        system_inputs.append(prompt_template["system"].replace("{subject}", subject))

        user_input = ""
        for ex in shots_by_category.get(subject, [])[: config.num_shots]:
            user_input += format_example(ex["question"], ex["options"], ex["cot_content"])
        user_input += format_example(record["question"], record["options"])
        user_input += answer_prefix
        user_inputs.append(user_input)

    return system_inputs, user_inputs


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main():
    args = make_args()
    config = InferenceConfig.from_yaml(args.config)
    llms = {model: LLM(config.llms[model]) for model in config.llms}
    test_llm = llms[config.model_name]

    with open(config.data_file, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    system_inputs, user_inputs = prepare_inputs(qa_data, config)
    prompts = test_llm.apply_chat_template(system_inputs, user_inputs)

    sampling_eval = config.sampling_eval
    if sampling_eval is not None and sampling_eval.enabled:
        # Multi-sample mode: draw N completions per question at (temp, top_p),
        # optionally requesting logprobs so we can report mean per-token entropy.
        test_llm.config.sampling_args["temperature"] = sampling_eval.temperature
        test_llm.config.sampling_args["top_p"] = sampling_eval.top_p
        if sampling_eval.compute_entropy:
            test_llm.config.sampling_args["logprobs"] = True
            test_llm.config.sampling_args["top_logprobs"] = sampling_eval.top_logprobs

        sample_results = test_llm.batch_generate_samples(prompts, sampling_eval.num_samples)
        for idx, samples in enumerate(sample_results):
            # Drop raw_response to keep the samples file compact.
            qa_data[idx]["samples"] = [
                {
                    "output": s["output"],
                    "reasoning_content": s["reasoning_content"],
                    "completion_tokens": s["completion_tokens"],
                    "mean_entropy": s["mean_entropy"],
                }
                for s in samples
            ]
    else:
        results = test_llm.batch_generate(prompts)
        [qa_data[idx].update(result) for idx, result in enumerate(results)]

    os.makedirs(config.save_path, exist_ok=True)
    with open(os.path.join(config.save_path, "inference.json"), "w", encoding="utf-8") as f:
        json.dump(qa_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
