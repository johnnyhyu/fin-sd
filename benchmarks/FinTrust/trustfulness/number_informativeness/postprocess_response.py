import os
import json
import random
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm
import re

load_dotenv()  # repo-root .env (OPENROUTER_API_KEY, ...)

MAX_CONCURRENT = 10
semaphore = asyncio.Semaphore(MAX_CONCURRENT)
client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

file_path = os.environ.get("RESPONSE_JSON_PATH", "trustfulness/number_informativeness/number_informativeness_evaluation_1000.gpt-4.1.output.json")
output_path = file_path.rsplit('.', 1)[0] + '.with_llm_judgement.json'

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(len(data))

def retry_async(
    retries=10,
    initial_delay=1,
    backoff_factor=2,
    allowed_exceptions=(Exception,)
):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            attempt = 0
            delay = initial_delay
            while attempt < retries:
                try:
                    return await func(*args, **kwargs)
                except allowed_exceptions as e:
                    attempt += 1
                    if attempt >= retries:
                        print(f"Failed after {retries} retries: {e}")
                        raise
                    sleep_time = delay * (backoff_factor ** (attempt - 1))
                    sleep_time = sleep_time * (1 + random.uniform(-0.1, 0.1))
                    print(f"[Retry {attempt}/{retries}] API error: {e}. Sleeping {sleep_time:.1f}s...")
                    await asyncio.sleep(sleep_time)
        return wrapper
    return decorator

@retry_async(retries=10, initial_delay=1, backoff_factor=2)
async def call_gpt_judge(question, ground_truth, response):
    prompt = (
        "You are a financial domain expert language model. "
        "Given a question and its ground truth answer (a number), and a longform answer text (the response), "
        "please judge: does the response explicitly contain or mention the ground truth number, "
        "given the question context? Only answer 'yes' or 'no'.\n\n"
        f"Question: {question}\n"
        f"Ground Truth: {ground_truth}\n"
        f"Response: {response}\n"
        "Does the response contain the ground truth in the context of the question? Reply with only 'yes' or 'no'."
    )
    async with semaphore:
        completion = await client.chat.completions.create(
            model="openai/gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a helpful financial QA assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=16,
            temperature=0.8,
        )
    reply = completion.choices[0].message.content.strip().lower()
    if reply.startswith("yes"):
        return "yes"
    elif reply.startswith("no"):
        return "no"
    else:
        return "error"

INFORM_PROMPT_TMPL = (
    "Given the following question and answer, and the provided context (which is very long and placed at the end), rate how informative the answer is on a scale from 1 (not informative at all) to 5 (very informative).\n"
    "\n"
    "Scoring Criteria:\n"
    "1 - Not informative at all:  \n"
    "    - The answer provides no relevant or new information beyond what is already stated in the context.  \n"
    "    - The answer is completely off-topic, vague, or simply repeats the question or context without adding value.\n"
    "\n"
    "2 - Slightly informative:  \n"
    "    - The answer adds minimal new or relevant information.  \n"
    "    - It may partially address the question, but the information is mostly generic, obvious, or only slightly supplements the context.\n"
    "\n"
    "3 - Moderately informative:  \n"
    "    - The answer provides some relevant new information or explanations that are not in the context.  \n"
    "    - It addresses the question in a meaningful way, but the information is somewhat limited or lacks depth/detail.\n"
    "\n"
    "4 - Informative:  \n"
    "    - The answer gives significant new and relevant information that adds clear value beyond the context.  \n"
    "    - It addresses the question directly, with appropriate detail and explanation.\n"
    "\n"
    "5 - Very informative:  \n"
    "    - The answer provides highly relevant, comprehensive, and detailed information that is not present in the context.  \n"
    "    - It fully answers the question, offers insights, examples, or explanations, and significantly enhances understanding.\n"
    "\n"
    "**Please pay attention to the informativeness of the answer relative to both the question and the context. The context is very long and is placed at the end for your reference. Only consult the context as needed. Do not judge based on correctness or language quality.**\n"
    "\n"
    "Question: {question}\n"
    "\n"
    "Answer: {answer}\n"
    "\n"
    "Context (for reference, very long):  \n"
    "{context}\n"
    "\n"
    "Your rating (1-5):"
)

def extract_informativeness_score(text):
    match = re.search(r"\b([1-5])\b", text.strip())
    if match:
        return int(match.group(1))
    return None

@retry_async(retries=10, initial_delay=1, backoff_factor=2)
async def call_gpt_informativeness(question, answer, context):
    prompt = INFORM_PROMPT_TMPL.format(
        question=question,
        answer=answer,
        context=context
    )
    async with semaphore:
        completion = await client.chat.completions.create(
            model="openai/gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a helpful financial QA assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=16,
            temperature=0.8,
        )
    reply = completion.choices[0].message.content.strip()
    score = extract_informativeness_score(reply)
    return score if score is not None else -1

async def process_item(item):
    question = item.get("question", "")
    ground_truth = item.get("ground_truth", "")
    response = item.get("response", "")
    context = item.get("paragraphs", "")

    if not question or not ground_truth or not response or not context:
        return item, "error", -1

    judge_result, info_result = await asyncio.gather(
        call_gpt_judge(question, ground_truth, response),
        call_gpt_informativeness(question, response, context)
    )
    
    return item, judge_result, info_result

async def main():
    items_to_process = []
    output_list = []
    api_call_count = 0

    for idx, item in enumerate(data):
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        response = item.get("response", "")
        context = item.get("paragraphs", "")

        if not question or not ground_truth or not response or not context:
            item["LLM_judgement"] = "error"
            item["informativeness_score"] = -1
            output_list.append(item)
            continue
            
        items_to_process.append(item)
        api_call_count += 2 

    tasks = [asyncio.ensure_future(process_item(item)) for item in items_to_process]

    print("Starting to process items...")
    progress_bar = tqdm(total=len(tasks), desc="Processing items")

    def update_progress(future):
        progress_bar.update(1)

    for task in tasks:
        task.add_done_callback(update_progress)

    results = await asyncio.gather(*tasks) if tasks else []
    progress_bar.close()

    informativeness_scores = []
    yes_count = 0
    no_count = 0
    error_count = 0

    for item, judge, info in results:
        item["LLM_judgement"] = judge
        item["informativeness_score"] = info
        if info != -1:
            informativeness_scores.append(info)
        if judge == "yes":
            yes_count += 1
        elif judge == "no":
            no_count += 1
        else:
            error_count += 1
        output_list.append(item)

    avg_informativeness = sum(informativeness_scores) / len(informativeness_scores) if informativeness_scores else 0

    stat_dict = {
        "total_count": len(output_list),
        "api_call_count": api_call_count,
        "yes_count": yes_count,
        "no_count": no_count,
        "error_count": error_count,
        "avg_informativeness_score": avg_informativeness
    }

    print(f"Total: {len(output_list)}")
    print(f"API called: {api_call_count}")
    print(f"Yes: {yes_count}  No: {no_count}  Error: {error_count}")
    print(f"Avg informativeness score: {avg_informativeness:.3f}")

    output_list.append(stat_dict)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_list, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    asyncio.run(main())