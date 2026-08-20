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
from api_utils import report_failures, MODEL_CATALOG, retry_async, standalone_semaphore, load_jsonl_or_json

load_dotenv()  # unified repo-root .env

JSON_FILE_PATH = os.environ.get("PROMPT_JSON_PATH", "input.json")
MODEL_KEY = os.environ.get("MODEL_KEY", "gpt-4.1-mini")
PROCESS_NUM = 1000

@retry_async(retries=10, initial_delay=1, backoff_factor=2)
async def run_prompt(client, model, prompt):
    system_prompt = "You are a helpful assistant in finance domain."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.8,
    )
    return resp.choices[0].message.content

async def process_one_dict(item, client, model, sem):
    prompt = item.get("test_prompt", "")
    async with sem:
        try:
            output = await run_prompt(client, model, prompt)
            item["answer"] = output
            return item
        except Exception as e:
            item["answer"] = f"ERROR: {e}"
            return item

def get_jobs(input_path, model_key, sem, *, system_prompt_type=None):
    print(f"Loading file: {input_path}")
    items = load_jsonl_or_json(input_path)
    print(f"Loaded {len(items)} items.")

    if PROCESS_NUM > 0:
        items = items[:PROCESS_NUM]
        print(f"Processing only the first {PROCESS_NUM} items.")

    if model_key not in MODEL_CATALOG:
        raise ValueError(f"MODEL_KEY '{model_key}' not found in MODEL_CATALOG.")
    model_info = MODEL_CATALOG[model_key]
    client = model_info["client"]
    model = model_info["model"]
    print(f"Using model: {model_key} ({model_info['display']})")

    tasks = [process_one_dict(item, client, model, sem) for item in items]

    def writer(results):
        out_path = f"{os.path.splitext(input_path)[0]}.{model_key}.output.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return out_path

    return tasks, writer

async def main():
    sem = standalone_semaphore()
    tasks, writer = get_jobs(JSON_FILE_PATH, MODEL_KEY, sem)
    results = await tqdm_asyncio.gather(*tasks, desc="Processing", total=len(tasks))
    out_path = writer(results)
    report_failures(results, out_path)

if __name__ == "__main__":
    asyncio.run(main())
