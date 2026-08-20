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

JSON_FILE_PATH = os.environ.get("PROMPT_JSON_PATH", "input.json")
MODEL_KEY = os.environ.get("MODEL_KEY", "gpt-4.1-mini")

def load_json_list(file_path):
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)

def extract_first_token_probs(resp):
    try:
        choice = resp["choices"][0]
        logprobs = choice.get("logprobs", {})
        if "content" in logprobs and isinstance(logprobs["content"], list) and len(logprobs["content"]) > 0:
            first_token = logprobs["content"][0]
            if first_token.get("top_logprobs"):
                return first_token["top_logprobs"]
            else:
                return {first_token["token"]: first_token["logprob"]}
        elif "top_logprobs" in logprobs and isinstance(logprobs["top_logprobs"], list) and len(logprobs["top_logprobs"]) > 0:
            return logprobs["top_logprobs"][0]
        elif "tokens" in logprobs and "token_logprobs" in logprobs:
            token = logprobs["tokens"][0]
            logprob = logprobs["token_logprobs"][0]
            return {token: logprob}
        if hasattr(choice, "logprobs") and hasattr(choice.logprobs, "content"):
            content = choice.logprobs.content
            if content and len(content) > 0:
                tok = content[0]
                if hasattr(tok, "top_logprobs") and tok.top_logprobs:
                    return tok.top_logprobs
                else:
                    return {tok.token: tok.logprob}
    except Exception as e:
        pass
    return None

@retry_async(retries=10, initial_delay=1, backoff_factor=2)
async def run_prompt(client, model, prompt, system_prompt=None):
    if not system_prompt:
        system_prompt = "You are a finance LLM expertise in predicting finance trends."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    # Reasoning models (e.g. gpt-oss) spend most of the token budget on
    # internal reasoning before emitting any content. With a tiny max_tokens
    # the answer never gets produced (content comes back empty), so give it
    # plenty of room to finish reasoning and still output the buy/sell answer.
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.8,
        max_tokens=2048,
        logprobs=True,
    )
    if hasattr(resp, "model_dump"):
        resp = resp.model_dump()
    answer = resp["choices"][0]["message"].get("content") or resp["choices"][0].get("text") or ""
    first_token_probs = extract_first_token_probs(resp)
    return answer, first_token_probs

async def process_one_company(item, client, model, sem):
    prompt = item.get("prompt", "")
    async with sem:
        try:
            answer, first_token_probs = await run_prompt(client, model, prompt)
        except Exception as e:
            answer = f"ERROR: {str(e)}"
            first_token_probs = None
        item["answer"] = answer
        item["first_token_probs"] = first_token_probs
        return item

def get_jobs(input_path, model_key, sem, *, system_prompt_type=None):
    print(f"Loading file: {input_path}")
    data = load_json_list(input_path)
    print(f"Loaded {len(data)} items.")
    print(f"Processing {len(data)} items (no sampling by gsector)")

    if model_key not in MODEL_CATALOG:
        raise ValueError(f"MODEL_KEY '{model_key}' not found in MODEL_CATALOG.")
    model_info = MODEL_CATALOG[model_key]
    client = model_info["client"]
    model = model_info["model"]
    print(f"Using model: {model_key} ({model_info['display']})")

    tasks = [process_one_company(item, client, model, sem) for item in data]

    def writer(results):
        out_path = f"{os.path.splitext(input_path)[0]}.{model_key}.llm_company-level-fairness_output.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return out_path

    return tasks, writer

async def main():
    sem = standalone_semaphore()
    tasks, writer = get_jobs(JSON_FILE_PATH, MODEL_KEY, sem)
    results = await tqdm_asyncio.gather(*tasks, desc="Company-level Fairness Answering", total=len(tasks))
    out_path = writer(results)
    report_failures(results, out_path)

if __name__ == "__main__":
    asyncio.run(main())
