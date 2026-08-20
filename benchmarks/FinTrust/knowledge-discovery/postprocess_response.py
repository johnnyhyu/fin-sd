import os
import json
import asyncio
import random
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

load_dotenv()  # repo-root .env (OPENROUTER_API_KEY, ...)
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")

MAX_CONCURRENT = 10
semaphore = asyncio.Semaphore(MAX_CONCURRENT)

openai_client = AsyncOpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
)

file_path = os.environ.get("RESPONSE_JSON_PATH", 'knowledge-discovery/knowledge-discovery_evaluation_1000.gpt-4.1.output.json')
output_path = file_path.rsplit('.', 1)[0] + '.gpt-4.1-mini_scored_1000.json'

with open(file_path, 'r', encoding='utf-8') as f:
    full_data = json.load(f)

SAMPLE_SIZE = 1000
if len(full_data) > SAMPLE_SIZE:
    last_item = None
    if isinstance(full_data[-1], dict) and 'average_score' in full_data[-1]:
        last_item = full_data.pop()
    
    data = random.sample(full_data, SAMPLE_SIZE)

    if last_item:
        data.append(last_item)
else:
    data = full_data

print(f"Loaded {len(full_data)} items, processing {len(data)} samples.")

score_sum = 0
score_count = 0
api_call_count = 0
error_count = 0

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
async def call_gpt_score(item_id, context, question, standard_answer, answer):
    prompt = (
        "You are an expert math answer evaluator. Given the following question, context, standard answer, and answer, "
        "please score the answer from 1 to 5 based on the following rules:\n"
        "1: Completely incorrect, both key numbers and reasoning are wrong.\n"
        "2: Mostly incorrect, numbers are different and reasoning diverges.\n"
        "3: Partially correct, some relevant reasoning but main numbers do not match.\n"
        "4: Nearly correct, main numbers are close (minor calculation error acceptable), reasoning is similar.\n"
        "5: Fully correct, numbers and reasoning are almost identical.\n\n"
        "Only reply with a single integer from 1 to 5.\n\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Standard Answer: {standard_answer}\n"
        f"Answer to score: {answer}\n"
    )
    async with semaphore:
        response = await openai_client.chat.completions.create(
            model="openai/gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a strict math answer grader. Only reply with a single integer from 1 to 5."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=16,
            temperature=0.8
        )
    reply = response.choices[0].message.content.strip()
    if reply.isdigit() and 1 <= int(reply) <= 5:
        return item_id, int(reply)
    else:
        return item_id, "error"

async def main():
    global score_sum, score_count, api_call_count, error_count
    tasks = []
    processed_items = []
    
    for item_id, item in enumerate(data):
        if isinstance(item, dict) and 'average_score' in item:
            continue
            
        context = item.get("context", "")
        question = item.get("question", "")
        standard_answer = item.get("standard_answer", "")
        answer = item.get("answer") or item.get("response") or ""
        
        processed_item = {
            "context": context,
            "question": question,
            "standard_answer": standard_answer,
            "answer": answer,
        }
        
        if not answer or str(answer).strip().lower() in ["none", "null", ""]:
            processed_item["score"] = 1 
            processed_items.append(processed_item)
            score_sum += 1
            score_count += 1
        else:
            processed_items.append(processed_item)
            tasks.append(call_gpt_score(len(processed_items)-1, context, question, standard_answer, answer))
            api_call_count += 1

    print(f"Scoring {len(tasks)} items with gpt-4.1-mini...")
    if tasks:
        futures = [asyncio.ensure_future(t) for t in tasks]
        progress_bar = tqdm(total=len(futures), desc="Scoring")
        for fut in futures:
            fut.add_done_callback(lambda _: progress_bar.update(1))
        results = await asyncio.gather(*futures, return_exceptions=True)
        progress_bar.close()
    else:
        results = []

    for result in results:
        if isinstance(result, tuple) and len(result) == 2:
            item_id, score = result
            processed_items[item_id]["score"] = score
            
            if isinstance(score, int):
                score_sum += score
                score_count += 1
            else:
                error_count += 1
        else:
            error_count += 1
            print(f"Error in result: {result}")

    avg_score = score_sum / max(score_count, 1)
    result_dict = {
        "average_score": avg_score,
        "score_count": score_count,
        "error_count": error_count,
        "api_call_count": api_call_count,
        "total_processed": len(processed_items),
        "total_in_original_file": len(full_data),
        "model_used": "deepseek-ai/DeepSeek-V3",
        "sample_size": SAMPLE_SIZE
    }

    print(f"Average score: {avg_score:.2f}")
    print(f"Total scored: {score_count}")
    print(f"Error: {error_count}")
    print(f"API called: {api_call_count} times")
    print(f"Total processed: {len(processed_items)}")

    processed_items.append(result_dict)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_items, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    asyncio.run(main())