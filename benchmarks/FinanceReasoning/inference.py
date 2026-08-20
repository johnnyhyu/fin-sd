import json
import os
from utils.llm import LLM 
from utils.retriever import Retriever
from utils.config import InferenceConfig 
import argparse

def prepare_user_inputs(
  data, 
  config: InferenceConfig, 
  llms: dict[str, LLM] = None,
):
    prompt_template = config.prompt.template
    user_inputs = []
    question_inputs = []
    useful_functions = []
    for record in data:
        if record.get("tables", None):
            table_input = "Table:\n"
            table_input += "\n\n".join(record["tables"]) + "\n"
            question_input = f"{table_input}\nQuestion: {record['question']}\n"
        elif record.get("context", None):
            context_input = "The following question context is provided for your reference.\n"
            context_input += record["context"] + "\n"
            question_input = f"{context_input}\nQuestion: {record['question']}\n"
        else:
            question_input = f"Question: {record['question']}\n"
        question_inputs.append(question_input)

    if config.use_retrieve:
        retriever = Retriever(config.retrieve, llms)
        querys = question_inputs if config.retrieve.use_llm_optimize else [record["question"] for record in data]
        useful_functions = retriever.retrieve(querys)
    else:
        useful_functions = [[] for _ in question_inputs]

    for idx, question_input in enumerate(question_inputs):
        useful_function = useful_functions[idx]
        user_input = question_input
        if len(useful_function) > 0:
            knowledge_input = "The following are the financial functions for your reference.\n"
            knowledge_input += "\n".join(useful_function) + "\n"
            user_input += f"\n{knowledge_input}"
        user_input += f"\n{prompt_template['program_prefix']}"
        user_inputs.append(user_input)

    return user_inputs

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()

def main():
    args = make_args()
    config = InferenceConfig.from_yaml(args.config)
    llms = {model: LLM(config.llms[model]) for model in config.llms}
    test_llm = llms[config.model_name]

    with open(config.data_file, 'r') as f:
        qa_data = json.load(f)

    user_inputs  = prepare_user_inputs(qa_data, config, llms)
    system_inputs = [config.prompt.template["system"]] * len(user_inputs)
    prompts = test_llm.apply_chat_template(system_inputs, user_inputs)

    """
    ** You can save the prompts to a file for openai batch inference. **
    dataset = config.dataset
    subset = config.subset
    prompt_type = config.prompt.prompt_type
    model_id = llms[config.model_name].model_id
    prefix = f"{dataset}_{subset}_{prompt_type}_{model_id}"
    with open(f"{prefix}_prompts.json", "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=4, ensure_ascii=False)
    exit()
    """

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
        results =  test_llm.batch_generate(prompts)
        [qa_data[idx].update(result) for idx, result in enumerate(results)]

    os.makedirs(config.save_path, exist_ok=True)
    with open(os.path.join(config.save_path, f"inference.json"), "w", encoding="utf-8") as f:
        json.dump(qa_data, f, indent=4, ensure_ascii=False)



if __name__ == "__main__":
    main()