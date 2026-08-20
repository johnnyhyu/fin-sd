import os
from openai import OpenAI
import json
import re
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
from datasets import load_dataset
import argparse
import requests

# Every hosted model routes through OpenRouter; only OPENROUTER_API_KEY is needed
# (read from the unified repo-root .env). The special model_name "local" targets a
# self-hosted vLLM server instead. No direct OpenAI/DeepSeek/Gemini/Anthropic calls.
API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
random.seed(12345)

# Back-compat aliases: map the legacy --model_name values to OpenRouter slugs.
# Any value not listed is passed through unchanged, so you can pass a raw slug
# like "anthropic/claude-3.5-sonnet" or "google/gemini-2.0-flash-001" directly.
OPENROUTER_SLUGS = {
    "gpt-4": "openai/gpt-4",
    "gpt-4o": "openai/gpt-4o",
    "deepseek-chat": "deepseek/deepseek-chat",
    "deepseek-coder": "deepseek/deepseek-chat",
    "gemini-1.5-flash-latest": "google/gemini-flash-1.5",
    "gemini-1.5-pro-latest": "google/gemini-pro-1.5",
    "gemini-1.5-flash-8b": "google/gemini-flash-1.5-8b",
    "gemini-002-pro": "google/gemini-2.0-pro-exp",
    "gemini-002-flash": "google/gemini-2.0-flash-001",
    "claude-3-opus-20240229": "anthropic/claude-3-opus",
    "claude-3-sonnet-20240229": "anthropic/claude-3-sonnet",
}

# Reasoner model families spend tokens on a reasoning trace before the answer, so
# the default 4000-token cap truncates them before "The answer is (X)". Detected
# by slug substring (extend as new reasoners are added) and given REASONER_MAX_TOKENS
# instead. Non-reasoners are unaffected. o1/o3 are intentionally absent — those
# reject temperature=0 and are not routed here.
REASONER_HINTS = ("deepseek-r1", "deepseek-v4", "gpt-oss", "kimi-k", "nemotron",
                  "glm-", "qwq", "reasoner")
REASONER_MAX_TOKENS = 8192


def to_openrouter_slug(name):
    return OPENROUTER_SLUGS.get(name, name)


def is_reasoner(slug):
    s = slug.lower()
    return any(hint in s for hint in REASONER_HINTS)


def get_client():
    if args.model_name == "local":
        # Local OpenAI-compatible endpoint (e.g. a vLLM server started with
        # `vllm serve ... --api-key token-abc123`). Point OpenAI at the base_url.
        return OpenAI(api_key=args.api_key or "token-abc123", base_url=args.url)
    # Everything else goes through OpenRouter.
    if not API_KEY:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is unset. Set it in the repo-root .env "
            "(all hosted models route through OpenRouter)."
        )
    return OpenAI(api_key=API_KEY, base_url=OPENROUTER_BASE_URL)


def call_api(client, instruction, inputs):
    start = time.time()
    message_text = [{"role": "user", "content": instruction + inputs}]
    if args.model_name == "local":
        completion = client.chat.completions.create(
            model=args.served_model_name,
            messages=message_text,
            temperature=0,
            max_tokens=args.max_tokens,
            top_p=1,
        )
    else:
        slug = to_openrouter_slug(args.model_name)
        max_tokens = REASONER_MAX_TOKENS if is_reasoner(slug) else args.max_tokens
        completion = client.chat.completions.create(
            model=slug,
            messages=message_text,
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
        )
    result = completion.choices[0].message.content
    print("cost time", time.time() - start)
    return result


def load_mmlu_pro():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df


def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    res = {}
    for each in res_df:
        if each["category"] not in res:
            res[each["category"]] = []
        res[each["category"]].append(each)
    return res


def format_example(question, options, cot_content=""):
    if cot_content == "":
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = "Question: {}\nOptions: ".format(question)
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        example += "{}. {}\n".format(choice_map[i], opt)
    if cot_content == "":
        example += "Answer: "
    else:
        example += "Answer: " + cot_content + "\n\n"
    return example


def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def single_request(client, single_question, cot_examples_dict, exist_result):
    exist = True
    q_id = single_question["question_id"]
    for each in exist_result:
        if q_id == each["question_id"] and single_question["question"] == each["question"]:
            pred = extract_answer(each["model_outputs"])
            return pred, each["model_outputs"], exist
    exist = False
    category = single_question["category"]
    cot_examples = cot_examples_dict[category]
    question = single_question["question"]
    options = single_question["options"]
    prompt = "The following are multiple choice questions (with answers) about {}. Think step by" \
             " step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n" \
        .format(category)
    for each in cot_examples:
        prompt += format_example(each["question"], each["options"], each["cot_content"])
    input_text = format_example(question, options)
    try:
        response = call_api(client, prompt, input_text)
        response = response.replace('**', '')
    except Exception as e:
        print("error", e)
        return None, None, exist
    pred = extract_answer(response)
    return pred, response, exist


def update_result(output_res_path):
    category_record = {}
    res = []
    success = False
    while not success:
        try:
            if os.path.exists(output_res_path):
                with open(output_res_path, "r") as fi:
                    res = json.load(fi)
                    for each in res:
                        category = each["category"]
                        if category not in category_record:
                            category_record[category] = {"corr": 0.0, "wrong": 0.0}
                        if not each["pred"]:
                            x = random.randint(0, len(each["options"]) - 1)
                            if x == each["answer_index"]:
                                category_record[category]["corr"] += 1
                            else:
                                category_record[category]["wrong"] += 1
                        elif each["pred"] == each["answer"]:
                            category_record[category]["corr"] += 1
                        else:
                            category_record[category]["wrong"] += 1
            success = True
        except Exception as e:
            print("Error", e, "sleep 2 seconds")
            time.sleep(2)
    return res, category_record


def merge_result(res, curr):
    merged = False
    for i, single in enumerate(res):
        if single["question_id"] == curr["question_id"] and single["question"] == curr["question"]:
            res[i] = curr
            merged = True
    if not merged:
        res.append(curr)
    return res


def _pipeline_base_model():
    """Best-effort read of the pipeline's base vLLM model id (config.VLLM_MODEL),
    used to exclude the base when auto-detecting the served adapter. Returns None
    if the pipeline package isn't importable (MMLU-Pro run standalone)."""
    try:
        import sys
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from pipeline import config
        return config.VLLM_MODEL
    except Exception:
        return None


def resolve_served_model_name(client):
    """For a local endpoint, figure out which model to target. If the user gave
    --served_model_name, honor it. Otherwise ask the server what it's serving and
    auto-pick the sole adapter/model (excluding the known base VLLM_MODEL), so
    pointing at a serve_epoch server "just works" without naming the epoch."""
    if args.served_model_name:
        return args.served_model_name
    models = [m.id for m in client.models.list().data]
    if not models:
        raise ValueError(f"Local endpoint {args.url} reports no served models.")
    base = os.getenv("VLLM_MODEL") or _pipeline_base_model()
    adapters = [m for m in models if m != base]
    if len(adapters) == 1:
        return adapters[0]
    if len(models) == 1:
        return models[0]
    raise ValueError(
        f"Multiple models served at {args.url}: {models}. "
        "Pass --served_model_name to pick one (e.g. the epoch adapter name)."
    )


def evaluate(subjects):
    client = get_client()
    if args.model_name == "local":
        args.served_model_name = resolve_served_model_name(client)
        print(f"local endpoint {args.url} -> served model '{args.served_model_name}'")
    test_df, dev_df = load_mmlu_pro()
    if not subjects:
        subjects = list(test_df.keys())
    print("assigned subjects", subjects)
    for subject in subjects:
        test_data = test_df[subject]
        output_res_path = os.path.join(args.output_dir, subject + "_result.json")
        output_summary_path = os.path.join(args.output_dir, subject + "_summary.json")
        res, category_record = update_result(output_res_path)

        # Resume: skip questions already answered in a prior run.
        done_ids = {each["question_id"] for each in res if each.get("model_outputs")}
        todo = [each for each in test_data if each["question_id"] not in done_ids]

        # Fan out requests across `--num_workers` threads. The OpenAI/provider
        # clients are thread-safe for concurrent calls; a lock guards the shared
        # result list and the incremental save so progress survives a crash.
        save_lock = threading.Lock()

        def process(question):
            pred, response, _ = single_request(client, question, dev_df, [])
            return question, pred, response

        with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
            futures = [pool.submit(process, each) for each in todo]
            for future in tqdm(as_completed(futures), total=len(futures)):
                question, pred, response = future.result()
                if response is None:
                    # Request failed after retries; leave it unanswered so a
                    # later run picks it up again.
                    continue
                question["pred"] = pred
                question["model_outputs"] = response
                with save_lock:
                    merge_result(res, question)
                    save_res(res, output_res_path)

        # Recompute the summary from the saved file so counts (including the
        # random-guess fallback for unanswered questions) match the prior logic.
        res, category_record = update_result(output_res_path)
        save_res(res, output_res_path)
        save_summary(category_record, output_summary_path)


def save_res(res, output_res_path):
    temp = []
    exist_q_id = []
    for each in res:
        if each["question_id"] not in exist_q_id:
            exist_q_id.append(each["question_id"])
            temp.append(each)
        else:
            continue
    res = temp
    with open(output_res_path, "w") as fo:
        fo.write(json.dumps(res))


def save_summary(category_record, output_summary_path):
    total_corr = 0.0
    total_wrong = 0.0
    for k, v in category_record.items():
        if k == "total":
            continue
        cat_acc = v["corr"] / (v["corr"] + v["wrong"])
        category_record[k]["acc"] = cat_acc
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    acc = total_corr / (total_corr + total_wrong)
    category_record["total"] = {"corr": total_corr, "wrong": total_wrong, "acc": acc}
    with open(output_summary_path, "w") as fo:
        fo.write(json.dumps(category_record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default="eval_results/")
    parser.add_argument("--model_name", "-m", type=str, default="openai/gpt-4o",
                        help="'local' for a self-hosted vLLM server (see --url), or "
                             "an OpenRouter model slug (e.g. 'openai/gpt-4o', "
                             "'anthropic/claude-3.5-sonnet', "
                             "'deepseek/deepseek-chat'). Legacy short names like "
                             "'gpt-4o'/'deepseek-chat' are mapped automatically.")
    parser.add_argument("--assigned_subjects", "-a", type=str, default="all")
    # Local OpenAI-compatible endpoint options (used when --model_name local).
    # Defaults mirror pipeline/config.py so this points at the serve_epoch server
    # out of the box (same VLLM_BASE_URL / VLLM_API_KEY env vars).
    parser.add_argument("--url", "-u", type=str,
                        default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
                        help="base_url of a local OpenAI-compatible server (e.g. vLLM). "
                             "Defaults to $VLLM_BASE_URL.")
    parser.add_argument("--api_key", type=str,
                        default=os.getenv("VLLM_API_KEY", "EMPTY"),
                        help="API key for the local endpoint. Defaults to $VLLM_API_KEY.")
    parser.add_argument("--served_model_name", type=str, default=None,
                        help="model id the local server expects (vLLM --served-model-name, "
                             "e.g. an epoch adapter name). Auto-detected from the server "
                             "when omitted.")
    parser.add_argument("--max_tokens", type=int, default=4000,
                        help="max completion tokens for the local endpoint.")
    parser.add_argument("--num_workers", "-n", type=int, default=4,
                        help="number of concurrent requests.")
    assigned_subjects = []
    args = parser.parse_args()

    if args.assigned_subjects == "all":
        assigned_subjects = []
    else:
        assigned_subjects = args.assigned_subjects.split(",")
    os.makedirs(args.output_dir, exist_ok=True)
    evaluate(assigned_subjects)
