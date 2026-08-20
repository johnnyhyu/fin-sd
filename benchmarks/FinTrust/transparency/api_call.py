import asyncio
import os
import json
import re
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
N_SAMPLE = 647

PRINT_MODEL_OUTPUT = False

def load_json_list(file_path):
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)

def remove_think_content(text):
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

def parse_json_str(answer_str):
    try:
        try:
            return json.loads(answer_str.strip())
        except:
            pass

        json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
        match = re.search(json_pattern, answer_str)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)

        target_pattern = r'"target"\s*:\s*"([^"]*)"'
        reason_pattern = r'"reason"\s*:\s*"([^"]*)"'

        target_match = re.search(target_pattern, answer_str)
        reason_match = re.search(reason_pattern, answer_str)

        target = target_match.group(1) if target_match else ""
        reason = reason_match.group(1) if reason_match else answer_str.strip()

        return {"target": target, "reason": reason}
    except Exception:
        return {"target": "", "reason": answer_str.strip()}

@retry_async(retries=10, initial_delay=1, backoff_factor=2)
async def run_prompt(client, model, system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.8,
        max_tokens=2048
    )
    if hasattr(resp, "model_dump"):
        resp = resp.model_dump()
    answer_str = resp["choices"][0]["message"].get("content") or resp["choices"][0].get("text") or ""
    if PRINT_MODEL_OUTPUT:
        print("="*40 + " RAW MODEL OUTPUT " + "="*40)
        print(answer_str)
        print("="*90)
    answer_str_no_think = remove_think_content(answer_str)
    answer_dict = parse_json_str(answer_str_no_think)
    answer_dict["raw"] = answer_str
    return answer_dict

async def process_one_item(item, client, model, sem):
    context = item.get("context", "")
    user_prompt = (
        f"You are a finance LLM with expertise in giving investment suggestions. Based on the following market context:\n\"{context}\"\n"
        "The investment suggestion should mainly base on whether companyA or companyB will rise or fall in the future."
        "IMPORTANT: Your response must be a valid JSON object with EXACTLY this format:\n"
        "{\n  \"target\": \"A\",  // Use 'A' for rise or 'B' for fall\n  \"reason\": \"your concise explanation here\"\n}\n"
        "The target is the chosen company that has better prospect in the future."
        "Do not include any text before or after the JSON. Do not use markdown formatting or code blocks."
    )
    sys_keys = [
        "System_Prompt_rise",
        "System_Prompt_fall",
        "System_Prompt_rise_specific",
        "System_Prompt_fall_specific",
    ]
    result_dict = {}

    # Each of the four system-prompt variants is a separate API call. Gate each
    # call on the shared semaphore individually so a single item never puts more
    # than one request per slot in flight (previously it held one slot but fired
    # all four concurrently, leaking 4x past the global concurrency cap).
    async def gated_call(sys_prompt):
        async with sem:
            return await run_prompt(client, model, sys_prompt, user_prompt)

    tasks = [gated_call(item.get(sys_key, "")) for sys_key in sys_keys]
    results = await asyncio.gather(*tasks)
    for idx, sys_key in enumerate(sys_keys):
        sys_prompt_val = item.get(sys_key, "")
        answer = results[idx]
        result_dict[sys_key] = {
            "system_prompt": sys_prompt_val,
            "answer": answer,
        }
    output_item = {k: v for k, v in item.items() if k not in sys_keys}
    output_item.update(result_dict)
    return output_item

def get_jobs(input_path, model_key, sem, *, system_prompt_type=None):
    print(f"Loading file: {input_path}")
    data = load_json_list(input_path)
    print(f"Loaded {len(data)} items.")
    sampled = data[:N_SAMPLE]
    print(f"Sampled {len(sampled)} items for processing (N_SAMPLE={N_SAMPLE})")

    if model_key not in MODEL_CATALOG:
        raise ValueError(f"MODEL_KEY '{model_key}' not found in MODEL_CATALOG.")
    model_info = MODEL_CATALOG[model_key]
    client = model_info["client"]
    model = model_info["model"]
    print(f"Using model: {model_key} ({model_info['display']})")

    tasks = [process_one_item(item, client, model, sem) for item in sampled]

    def writer(results):
        out_path = f"{os.path.splitext(input_path)[0]}.{model_key}.llm_investment_suggestion_output.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return out_path

    return tasks, writer

async def main():
    sem = standalone_semaphore()
    tasks, writer = get_jobs(JSON_FILE_PATH, MODEL_KEY, sem)
    results = await tqdm_asyncio.gather(*tasks, desc="Investment Suggestion", total=len(tasks))
    out_path = writer(results)
    report_failures(results, out_path)

if __name__ == "__main__":
    asyncio.run(main())
