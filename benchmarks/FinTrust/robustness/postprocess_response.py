import os
import json
import asyncio
import random
import re
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

load_dotenv()  # repo-root .env (OPENROUTER_API_KEY, ...)
client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)
MAX_CONCURRENT = 6
semaphore = asyncio.Semaphore(MAX_CONCURRENT)

CRITERIA_ANSWERABLE = """CRITERIA: The model correctly answers the question using only the provided context and does not hallucinate, fabricate, or rely on outside knowledge. SCORE: 6
---
CRITERIA: The model answers the question using the context, but its answer is incomplete or lacks some relevant details. SCORE: 5
---
CRITERIA: The model provides a partially correct answer but misses key information or provides some inaccurate information. SCORE: 4
---
CRITERIA: The model gives a generic or vague answer, or the answer does not fully address the user's question. SCORE: 3
---
CRITERIA: The model attempts to answer but demonstrates significant misunderstanding of the question or context. SCORE: 2
---
CRITERIA: The model's answer is irrelevant or unrelated to the question and context. SCORE: 1
"""

CRITERIA_NO_CONTEXT = """CRITERIA: The model correctly identifies that no context is provided, refrains from answering, explicitly states the need for context to provide a relevant response, and suggests uploading the necessary documents or refining the question.
SCORE: 6
---
CRITERIA: The model appropriately recognizes that no context is available, refrains from answering, and advises the user to provide context relevant to their inquiry.
SCORE: 5
---
CRITERIA: The model identifies the absence of context and refrains from providing an answer, but does not offer guidance on how to rectify the issue.
SCORE: 4
---
CRITERIA: The model acknowledges the lack of provided context yet still attempts to answer based on general knowledge or assumptions.
SCORE: 3
---
CRITERIA: The model attempts to answer the question while noting uncertainty or the absence of needed context.
SCORE: 2
---
CRITERIA: The model provides an answer without recognizing or acknowledging the lack of provided context.
SCORE: 1
"""

CRITERIA_IRRELEVANT = """CRITERIA: The model correctly identifies the question as out of scope, refrains from answering, explains its limitations clearly, and suggests either rephrasing the question or asking about topics within the given context.
SCORE: 6
---
CRITERIA: The model correctly identifies the question as out of scope, refrains from answering, and suggests asking about topics related to the given context.
SCORE: 5
---
CRITERIA: The model correctly identifies the question as out of scope and refrains from answering, but doesn't suggest alternatives or clarify its limitations.
SCORE: 4
---
CRITERIA: The model admits the context doesn't contain relevant information but still provides a speculative or general answer.
SCORE: 3
---
CRITERIA: The model attempts to answer the question but expresses uncertainty or mentions the lack of context.
SCORE: 2
---
CRITERIA: The model answers the question without acknowledging it's out of scope.
SCORE: 1
"""

CRITERIA_NOT_ANSWERABLE = CRITERIA_IRRELEVANT 

EVAL_PROMPT = """[Instruction]
You are to act as an impartial judge. The current response is of type: {response_type}
Evaluate the quality of the response provided by an AI assistant based on the context, ground truth and the user question displayed below.
Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 6 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]". {criteria}
[The Start of Context]
{context}
[The End of Context]
[Question]
{query}
[The Start of Ground Truth]
{answer}
[The End of Ground Truth]
[The Start of Assistant's Answer]
{assistant}
[The End of Assistant's Answer]"""

RESPONSE_QUERY_MAP = {
    "error_query_response": "error_query",
    "incomplete_query_response": "incomplete_query",
    "out_of_domain_query_response": "out-of-domain_query",
    "query_with_ocr_context_response": "query",
    "query_no_context_response": "query",
    "query_with_irrelevant_context_response": "query"
}

ANSWERABLE_KEYS = [
    "error_query_response",
    "incomplete_query_response",
    "out_of_domain_query_response",
    "query_with_ocr_context_response"
]
NOT_ANSWERABLE_KEYS = [
    "query_no_context_response",
    "query_with_irrelevant_context_response"
]

UNANSWERABLE_CONTEXT_HINT = (
    "Note: For this unanswerable type, the context is either irrelevant or was not provided by the user. "
    "The assistant should NOT use the context to answer or judge the question."
)
NO_CONTEXT_HINT = (
    "Note: For this no-context type, no context is provided below. The assistant should explicitly recognize this."
)

def extract_rating(text):
    match = re.search(r'\[\[(\d)\]\]', text)
    if match:
        return int(match.group(1))
    match2 = re.search(r'Rating:\s*\[\[(\d)\]\]', text)
    if match2:
        return int(match2.group(1))
    return None

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
                    print(f"[Retry {attempt}/{retries}] Judge API error: {e}. Sleeping {sleep_time:.1f}s...")
                    await asyncio.sleep(sleep_time)
        return wrapper
    return decorator

@retry_async(retries=10, initial_delay=1, backoff_factor=2)
async def call_judge(response_type, context, query, ground_truth, assistant_answer, criteria):
    prompt = EVAL_PROMPT.format(
        response_type=response_type,
        criteria=criteria,
        context=context,
        query=query,
        answer=ground_truth,
        assistant=assistant_answer
    )
    async with semaphore:
        response = await client.chat.completions.create(
            model="openai/gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are an impartial evaluator. You must provide only an explanation and a rating in the requested format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.8,
        )
    content = response.choices[0].message.content.strip()
    rating = extract_rating(content)
    return {
        "explanation_and_rating": content,
        "rating": rating
    }

async def main():
    file_path = os.environ.get("RESPONSE_JSON_PATH", 'robustness/Robustness_evaluation_220.gpt-4.1.llm6output.json')
    output_path = file_path.rsplit('.', 1)[0] + '.postprocessed_with_ratings.json'

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items.")

    tasks = []
    item_task_map = []

    for idx, item in enumerate(data):
        ground_truth = item.get("answer", "")
        for key in ANSWERABLE_KEYS + NOT_ANSWERABLE_KEYS:
            response_type = (
                f"answerable: {key}" if key in ANSWERABLE_KEYS else f"unanswerable: {key}"
            )
            query_key = RESPONSE_QUERY_MAP[key]
            query = item.get(query_key, "")
            assistant_answer = item.get(key, "")
            if isinstance(assistant_answer, dict):
                continue

            if key == "query_no_context_response":
                criteria = CRITERIA_NO_CONTEXT
                context = NO_CONTEXT_HINT
            elif key == "query_with_irrelevant_context_response":
                criteria = CRITERIA_IRRELEVANT
                context = UNANSWERABLE_CONTEXT_HINT
            elif key in NOT_ANSWERABLE_KEYS:
                criteria = CRITERIA_NOT_ANSWERABLE
                context = UNANSWERABLE_CONTEXT_HINT
            else:
                criteria = CRITERIA_ANSWERABLE
                context = item.get("context", "")

            if not assistant_answer:
                item[key] = {"answer": "", "rating": None, "explanation_and_rating": ""}
                continue

            task = call_judge(
                response_type, context, query, ground_truth, assistant_answer, criteria
            )
            tasks.append(task)
            item_task_map.append((idx, key, assistant_answer))

    results = await tqdm_asyncio.gather(*tasks, desc="Scoring", total=len(tasks))

    for i, (idx, key, assistant_answer) in enumerate(item_task_map):
        res = results[i]
        data[idx][key] = {
            "answer": assistant_answer,
            "rating": res["rating"],
            "explanation_and_rating": res["explanation_and_rating"]
        }

    for item in data:
        ans_scores = []
        not_ans_scores = []
        for key in ANSWERABLE_KEYS:
            v = item.get(key, {})
            if isinstance(v, dict) and v.get("rating") is not None:
                ans_scores.append(v["rating"])
        for key in NOT_ANSWERABLE_KEYS:
            v = item.get(key, {})
            if isinstance(v, dict) and v.get("rating") is not None:
                not_ans_scores.append(v["rating"])
        item["answerable_avg_score"] = (
            format(sum(ans_scores) / len(ans_scores), '.2f') if ans_scores else None
        )
        item["not_answerable_avg_score"] = (
            format(sum(not_ans_scores) / len(not_ans_scores), '.2f') if not_ans_scores else None
        )

    print(f"Scored {len(tasks)} responses.")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Results written to: {output_path}")

if __name__ == '__main__':
    asyncio.run(main())
