"""
inference_script.py — calls the self-hosted vLLM server on a problem.

Returns both the full reasoning chain and the extracted final answer.
The model is prompted to wrap reasoning in <reasoning>...</reasoning> and
the final answer in <answer>...</answer>.
"""
import re
from typing import Optional

from . import config
from .utils import logger, parse_vllm_token_id, post_with_retry, parse_chat_message

# The base vLLM server is launched under the constant served-model name
# config.VLLM_MODEL (pipeline/vllm_server.py). In the expert-LoRA path trained
# weights are picked up by restarting the server on a merged checkpoint, so
# _active_model stays None. In the non-expert LoRA path run_pipeline hot-loads the
# adapter and calls set_active_model() with its served name, so requests target the
# adapter instead of the base weights.
_active_model: Optional[str] = None


def set_active_model(name: Optional[str]) -> None:
    """Override the vLLM model name used for subsequent calls (None ⇒ default).

    Normally unused: weights are refreshed by restarting vLLM on a merged
    checkpoint while the served-model name stays config.VLLM_MODEL.
    """
    global _active_model
    _active_model = name
    logger.info("Inference now targets vLLM model '%s'.", name or config.VLLM_MODEL)


def get_active_model() -> Optional[str]:
    """Return the active vLLM model-name override (None ⇒ config.VLLM_MODEL).

    Used by the vLLM-backed probing path so the teacher distribution is read from
    the same served weights that rollouts use.
    """
    return _active_model


def _completion_token_ids(data: dict) -> list[int]:
    """Recover the generated token ids from a chat response's logprobs, or [].

    Only populated when the request asked for logprobs (see run_inference: the
    token-budget penalty needs the rollout's ids to score its tail on HF). Any
    shape the server does not give us in `token_id:N` form is reported and
    downgraded to [] — a missing length signal is worth a warning, not a dead
    training item.
    """
    try:
        content = ((data["choices"][0].get("logprobs") or {}).get("content")) or []
    except (KeyError, IndexError, TypeError):
        return []
    ids: list[int] = []
    try:
        for entry in content:
            ids.append(parse_vllm_token_id(entry["token"]))
    except (RuntimeError, KeyError, TypeError) as exc:
        logger.warning(
            "vLLM did not return usable token ids for the rollout (%s); the "
            "token-budget penalty will be skipped for this item.", exc,
        )
        return []
    return ids


def run_inference(problem: str) -> dict:
    """Call the vLLM server and return parsed reasoning + answer.

    Args:
        problem: Raw problem text from the training set.

    Returns:
        {
            "reasoning":         str  — chain-of-thought steps,
            "generated_answer":  str  — extracted final answer,
            "raw_output":        str  — complete model output,
            "finish_reason":     str|None — vLLM's stop reason; "length" means the
                                 rollout was cut off at the MAX_NEW_TOKENS budget,
            "completion_tokens": list[int] — generated token ids (empty unless
                                 config.LENGTH_PENALTY asked for logprobs),
            "n_completion_tokens": int — length of the rollout in tokens,
            "budget_frac":       float — n_completion_tokens / MAX_NEW_TOKENS,
        }
        The last four exist so the caller can tell a rollout that ran out of
        budget from one that answered, and score its tail — see
        config.LENGTH_PENALTY. Before they were plumbed through, a truncated
        rollout was indistinguishable downstream from a short wrong answer.
    """
    url = f"{config.VLLM_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.VLLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": _active_model or config.VLLM_MODEL,
        "messages": [
            {"role": "system", "content": config.INFERENCE_SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ],
        "temperature": config.TEMPERATURE,
        "max_tokens": config.MAX_NEW_TOKENS,
    }
    if config.LENGTH_PENALTY:
        # The token-budget penalty teacher-forces the rollout's tail on the HF
        # model, so it needs the exact ids the server generated — recoverable only
        # via logprobs + return_tokens_as_token_ids. top_logprobs=0 keeps this to
        # one entry per position (the chosen token); no alternatives are wanted.
        payload["logprobs"] = True
        payload["top_logprobs"] = 0
        payload["return_tokens_as_token_ids"] = True
    # Reproducible sampling (only affects output when TEMPERATURE > 0).
    if config.SEED >= 0:
        payload["seed"] = config.SEED

    resp = post_with_retry(url, headers=headers, payload=payload, timeout=300, label="vLLM")
    data = resp.json()
    # Hitting the max_tokens budget is expected occasionally for long
    # reasoning chains — the answer-tag fallback handles it — so unlike the
    # OpenRouter calls we only warn instead of raising on finish_reason.
    finish_reason = None
    try:
        finish_reason = data["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError):
        pass
    completion_tokens = _completion_token_ids(data) if config.LENGTH_PENALTY else []
    n_completion = len(completion_tokens)
    if not n_completion:
        # No logprobs (penalty off, or the server withheld them): fall back to the
        # usage block so budget_frac is still reported for logging.
        try:
            n_completion = int((data.get("usage") or {}).get("completion_tokens") or 0)
        except (TypeError, ValueError):
            n_completion = 0
    if finish_reason == "length":
        logger.warning(
            "vLLM completion hit the %d-token limit; output may be cut off.",
            config.MAX_NEW_TOKENS,
        )
    # parse_chat_message, NOT parse_chat_completion: an empty/null `content` beside
    # a full reasoning channel is the normal shape of a gpt-oss completion that
    # never opened its final message — the chain-of-thought goes to the analysis
    # channel and `content` only appears once the model starts answering. vLLM's
    # harmony parser reports that as content=None (see its parse_chat_output:
    # "Empty ... final content is returned as None"), which parse_chat_completion
    # rejects outright.
    #
    # That happens two ways, and BOTH have to survive here. Truncation mid-thought
    # (finish_reason="length") is the common one — measured at 3 of 6 opsd items on
    # the base 20B, and opsd then took zero training steps. The model simply ending
    # its turn from the analysis channel (finish_reason="stop") is rarer but lands
    # in exactly the same place. Either way, raising discards the one thing the
    # model produced and turns the item most worth training on into a skipped
    # error, so we keep the reasoning and fall through with an empty answer — which
    # the grader scores wrong, as it is. Only a response with NOTHING in it (both
    # channels empty) is a real failure, and parse_chat_message still raises there.
    message = parse_chat_message(data, label="vLLM", allow_truncated=True)
    native_reasoning = message["reasoning"]
    raw_output: str = message["content"] or ""

    reasoning, generated_answer = _parse_output(raw_output)

    # Reasoning models (gpt-oss et al.) emit their chain-of-thought on a
    # separate channel that vLLM's reasoning parser returns as
    # message.reasoning_content (message.reasoning on some versions), leaving
    # message.content with little beyond the final answer. In that case
    # _parse_output's no-tags fallback would have set `reasoning` to the
    # answer text, so prefer the native reasoning channel.
    if native_reasoning:
        if "<reasoning>" in raw_output and reasoning:
            # Content also carried a tagged reasoning block; keep both in
            # generation order (analysis channel precedes the final channel).
            reasoning = f"{native_reasoning}\n\n{reasoning}"
        else:
            reasoning = native_reasoning
        raw_output = f"{native_reasoning}\n\n{raw_output}"

    return {
        "reasoning": reasoning,
        "generated_answer": generated_answer,
        "raw_output": raw_output,
        "finish_reason": finish_reason,
        "completion_tokens": completion_tokens,
        "n_completion_tokens": n_completion,
        "budget_frac": (
            n_completion / config.MAX_NEW_TOKENS if config.MAX_NEW_TOKENS > 0 else 0.0
        ),
    }


def _parse_output(raw: str) -> tuple[str, str]:
    """Extract <reasoning> and <answer> blocks; fall back gracefully.

    When one tag is missing, use the other tag's position to split the text
    rather than returning the entire raw output for both fields.
    """
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", raw, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)

    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    elif answer_match:
        reasoning = raw[: answer_match.start()].strip()
    else:
        reasoning = raw.strip()

    if answer_match:
        answer = answer_match.group(1).strip()
    elif reasoning_match:
        answer = raw[reasoning_match.end():].strip() or raw.strip()
    else:
        answer = raw.strip()

    return reasoning, answer
