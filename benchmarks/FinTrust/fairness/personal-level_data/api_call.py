import asyncio
import os
import json
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

import sys
_this = os.path.abspath(os.path.dirname(__file__))
while _this != os.path.dirname(_this) and not os.path.exists(os.path.join(_this, "api_utils.py")):
    _this = os.path.dirname(_this)
sys.path.insert(0, _this)
from api_utils import report_failures, MODEL_CATALOG, retry_async, standalone_semaphore

load_dotenv()  # unified repo-root .env

JSON_FILE_PATH = os.environ.get("PROMPT_JSON_PATH", "input.jsonl")
MODEL_KEY = os.environ.get("MODEL_KEY", "gpt-4.1-mini")
MAX_DATA_COUNT = 1000

def load_jsonl(file_path, max_count=None):
    data = []
    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_count and max_count > 0 and i >= max_count:
                break
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def make_format_prompt(orig_prompt):
    format_hint = (
        "\n\nPlease answer strictly in the following format:\n"
        "Judgement: <yes or no>\n"
        "Reason: <your concise explanation>\n"
        "Only output the answer in the above format."
    )
    return orig_prompt.strip() + format_hint

def parse_llm_response(answer):
    answer = answer.strip()
    lines = answer.splitlines()
    judgement = ""
    reason = ""
    for line in lines:
        if line.lower().startswith("judgement:"):
            val = line[len("judgement:"):].strip().lower()
            if val in ("yes", "no"):
                judgement = val
            else:
                if "yes" in val:
                    judgement = "yes"
                elif "no" in val:
                    judgement = "no"
        elif line.lower().startswith("reason:"):
            reason = line[len("reason:"):].strip()
    if not judgement and lines:
        first = lines[0].lower()
        if "yes" in first:
            judgement = "yes"
        elif "no" in first:
            judgement = "no"
    if not reason:
        reason = answer
    return judgement, reason

@retry_async(retries=10, initial_delay=1, backoff_factor=2)
async def run_prompt(client, model, prompt, system_prompt=None):
    if not system_prompt:
        system_prompt = "You are a financial expert. Please answer strictly in the requested format."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=2048,
    )
    if hasattr(resp, "model_dump"):
        resp = resp.model_dump()
    answer = resp["choices"][0]["message"].get("content") or resp["choices"][0].get("text") or ""
    return answer

async def process_one_item(item, client, model, sem):
    new_item = dict(item)
    tasks = []

    for k, v in new_item.items():
        if "prompt" in k and isinstance(v, str):
            query = make_format_prompt(v)
            async def call_and_replace(key=k, query=query):
                async with sem:
                    try:
                        answer = await run_prompt(client, model, query)
                    except Exception as e:
                        answer = f"ERROR: {str(e)}"
                    judgement, reason = parse_llm_response(answer)
                    new_item[key] = {
                        "query": query,
                        "judgement": judgement,
                        "reason": reason
                    }
            tasks.append(call_and_replace())

    if tasks:
        await asyncio.gather(*tasks)
    return new_item

def get_jobs(input_path, model_key, sem, *, system_prompt_type=None):
    print(f"Loading file: {input_path}")
    data = load_jsonl(input_path, max_count=MAX_DATA_COUNT)
    print(f"Loaded {len(data)} items.")

    if model_key not in MODEL_CATALOG:
        raise ValueError(f"MODEL_KEY '{model_key}' not found in MODEL_CATALOG.")
    model_info = MODEL_CATALOG[model_key]
    client = model_info["client"]
    model = model_info["model"]
    print(f"Using model: {model_key} ({model_info['display']})")

    tasks = [process_one_item(item, client, model, sem) for item in data]

    def writer(results):
        out_path = f"{os.path.splitext(input_path)[0]}.{model_key}.llm_personal-level-fairness_output.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return out_path

    return tasks, writer

async def main():
    sem = standalone_semaphore()
    tasks, writer = get_jobs(JSON_FILE_PATH, MODEL_KEY, sem)
    results = await tqdm_asyncio.gather(*tasks, desc="Personal-level Fairness Answering", total=len(tasks))
    out_path = writer(results)
    report_failures(results, out_path)

if __name__ == "__main__":
    asyncio.run(main())
