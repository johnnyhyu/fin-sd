import asyncio
import os
import json
import random
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
NUM_ITEMS_TO_PROCESS = 220
HANDLE_OUT_OF_DOMAIN_ONLY = False

@retry_async(retries=10, initial_delay=1, backoff_factor=2)
async def run_prompt(client, model, query, context):
    prompt = (
        f"This is the query: {query}\n"
        f"This is the context: {context}\n"
        f"Please answer the query based only on the context above."
    )
    system_prompt = "You are a finance LLM expertise in processing information. Always answer only based on the given context."
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

async def process_one_dict(item, client, model, sem, idx, items):
    context = item.get("context", "")
    error_query = item.get("error_query", "")
    incomplete_query = item.get("incomplete_query", "")
    out_of_domain_query = item.get("out-of_domain_query", "")

    query = item.get("query", "")
    ocr_context = item.get("ocr_context", "")

    if len(items) > 1:
        other_indices = [i for i in range(len(items)) if i != idx]
        random_idx = random.choice(other_indices)
        irrelevant_context = items[random_idx].get("context", "")
    else:
        irrelevant_context = ""

    results = {}

    calls = [
        ("error_query_response", error_query, context),
        ("incomplete_query_response", incomplete_query, context),
        ("out_of_domain_query_response", out_of_domain_query, context),
        ("query_no_context_response", query, ""),
        ("query_with_ocr_context_response", query, ocr_context),
        ("query_with_irrelevant_context_response", query, irrelevant_context),
    ]
    async with sem:
        for key, q, ctx in calls:
            try:
                results[key] = await run_prompt(client, model, q, ctx)
            except Exception as e:
                results[key] = f"ERROR: {str(e)}"
    item.update(results)
    return item

async def handle_out_of_domain_only():
    out_path = f"{os.path.splitext(JSON_FILE_PATH)[0]}.{MODEL_KEY}.llm6output.json"
    print(f"Loading output json: {out_path}")
    items = load_jsonl_or_json(out_path)
    print(f"Loaded {len(items)} items from output file.")

    if MODEL_KEY not in MODEL_CATALOG:
        raise ValueError(f"MODEL_KEY '{MODEL_KEY}' not found in MODEL_CATALOG.")
    model_info = MODEL_CATALOG[MODEL_KEY]
    client = model_info["client"]
    model = model_info["model"]

    sem = standalone_semaphore()
    async def process_out_of_domain(item, sem):
        out_of_domain_query = item.get("out-of-domain_query", "")
        context = item.get("context", "")
        async with sem:
            try:
                resp = await run_prompt(client, model, out_of_domain_query, context)
            except Exception as e:
                resp = f"ERROR: {str(e)}"
        item["out_of_domain_query_response"] = resp
        return item

    tasks = [
        process_out_of_domain(item, sem)
        for item in items
    ]
    results = await tqdm_asyncio.gather(*tasks, desc="Fixing out-of-domain", total=len(tasks))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Updated out_of_domain_query_response in: {out_path}")

def get_jobs(input_path, model_key, sem, *, system_prompt_type=None):
    print(f"Loading file: {input_path}")
    items = load_jsonl_or_json(input_path)
    print(f"Loaded {len(items)} items.")

    if NUM_ITEMS_TO_PROCESS is not None:
        items = items[:NUM_ITEMS_TO_PROCESS]
        print(f"Processing only the first {len(items)} items.")

    if model_key not in MODEL_CATALOG:
        raise ValueError(f"MODEL_KEY '{model_key}' not found in MODEL_CATALOG.")
    model_info = MODEL_CATALOG[model_key]
    client = model_info["client"]
    model = model_info["model"]
    print(f"Using model: {model_key} ({model_info['display']})")

    tasks = [
        process_one_dict(item, client, model, sem, idx, items)
        for idx, item in enumerate(items)
    ]

    def writer(results):
        out_path = f"{os.path.splitext(input_path)[0]}.{model_key}.llm6output.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return out_path

    return tasks, writer

async def main():
    if HANDLE_OUT_OF_DOMAIN_ONLY:
        await handle_out_of_domain_only()
        return

    sem = standalone_semaphore()
    tasks, writer = get_jobs(JSON_FILE_PATH, MODEL_KEY, sem)
    results = await tqdm_asyncio.gather(*tasks, desc="Processing", total=len(tasks))
    out_path = writer(results)
    report_failures(results, out_path)

if __name__ == "__main__":
    asyncio.run(main())
