"""opsd probing — student rollouts + aligned teacher/student logits.

The on-policy half of opsd. For one wrong problem:

  1. Sample a student rollout from the *bare problem* prompt on vLLM (the served
     student weights), capped at config.MAX_ROLLOUT_TOKENS.
  2. Teacher-force those exact tokens through two HF prefixes and read the
     per-token logits:
       • student  = [system, user: problem]                              (grad)
       • teacher  = [system, user: problem + reference solution + re-solve] (no-grad)
     Both open a fresh assistant turn (add_generation_prompt=True), so the
     rollout tokens align to both prefixes — the two distributions differ only by
     the reference solution in the teacher's context.

The forward KL(teacher ‖ student) over those tokens (loss_script.run_loss) pulls
the un-conditioned student toward the reference-conditioned teacher.

Reuses the pipeline's HF singleton, low-level logit extraction, and token-id
parsing (pipeline.probing_script); only the prefix construction and the
rollout-sampling prompt are opsd-specific.
"""
from __future__ import annotations

from typing import Optional

import torch

from pipeline import config as pc
from pipeline import inference_script, probing_script
from pipeline.prompts import serving_prompt_ids
from pipeline.utils import logger, post_with_retry

from . import config

# Reused low-level helpers from the main pipeline's probing module.
get_model_and_tokenizer = probing_script.get_model_and_tokenizer
is_loaded = probing_script.is_loaded
_get_completion_logits = probing_script._get_completion_logits
_parse_token_id = probing_script._parse_token_id


# ── Prefix builders (fresh assistant turn — no mistake-point continuation) ───

def _prefix_ids(system: str, user: str, tokenizer) -> torch.Tensor:
    """1-D LongTensor of token ids for [system, user], opening a fresh assistant turn.

    Unlike pipeline.probing_script._apply_chat_template (which continues a partial
    assistant message), this ends with the assistant-turn opening — matching how
    vLLM tokenizes a bare chat request, so a sampled completion's ids concatenate
    cleanly onto this prefix.

    Built by pipeline.prompts.serving_prompt_ids, which renders through the same
    harmony encoding the server uses. Going through the chat template instead adds
    a 14-token tools sentence vLLM never sends, so the rollouts would be sampled
    under one context and teacher-forced under another (measured: 0.0014
    nats/token, ~0.5% of this objective's signal — small, but it is pure bias).
    """
    return torch.tensor(
        serving_prompt_ids(system, user, tokenizer, model_path=pc.HF_MODEL_PATH),
        dtype=torch.long,
    )


def _student_messages(problem: str) -> list[dict]:
    """The chat payload for a rollout request — vLLM renders these itself.

    Kept as messages (not ids) because the server does the rendering; the HF-side
    counterpart is build_student_prefix_ids, and pipeline.prompts is what keeps the
    two token-identical.
    """
    return [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


def build_student_prefix_ids(problem: str, tokenizer) -> torch.Tensor:
    return _prefix_ids(config.SYSTEM_PROMPT, problem, tokenizer)


def build_teacher_prefix_ids(problem: str, reference_solution: str, tokenizer) -> torch.Tensor:
    user = config.TEACHER_USER_TEMPLATE.format(
        problem=problem, reference_solution=reference_solution
    )
    return _prefix_ids(config.SYSTEM_PROMPT, user, tokenizer)


# ── vLLM student rollout sampling ────────────────────────────────────────────

def _vllm_rollouts(problem: str, n: int, seed: Optional[int]) -> list[list[int]]:
    """Sample n completions from the bare-problem prompt via vLLM.

    Drawn from the *served* student weights (the current merged/hot-swapped
    checkpoint) with temperature/top-p sampling in a single n-way request. Each
    chosen token is recovered as an HF-aligned vocab id via
    return_tokens_as_token_ids so it can be teacher-forced on the HF model. Uses
    add_generation_prompt (the vLLM default for chat) so the prompt tokenization
    matches build_student_prefix_ids.
    """
    url = f"{pc.VLLM_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {pc.VLLM_API_KEY}",
        "Content-Type": "application/json",
    }
    student_model = inference_script.get_active_model() or pc.VLLM_MODEL
    payload = {
        "model": student_model,
        "messages": _student_messages(problem),
        "temperature": config.ROLLOUT_TEMPERATURE,
        "top_p": config.ROLLOUT_TOP_P,
        "max_tokens": config.MAX_ROLLOUT_TOKENS,
        "n": n,
        "logprobs": True,
        "top_logprobs": 0,
        "return_tokens_as_token_ids": True,
    }
    if seed is not None:
        payload["seed"] = seed

    resp = post_with_retry(url, headers=headers, payload=payload, timeout=600,
                           label="opsd-rollout")
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"opsd-rollout returned an error body: {str(data['error'])[:500]}")
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"opsd-rollout response had no choices: {str(data)[:500]}")

    completions: list[list[int]] = []
    n_length_truncated = 0
    for choice in choices:
        content = ((choice.get("logprobs") or {}).get("content")) or []
        ids = [_parse_token_id(item["token"]) for item in content]
        if choice.get("finish_reason") == "length":
            n_length_truncated += 1
        if ids:
            completions.append(ids)
    if n_length_truncated:
        logger.debug(
            "opsd-rollout: %d/%d rollouts hit the %d-token cap.",
            n_length_truncated, len(choices), config.MAX_ROLLOUT_TOKENS,
        )
    return completions


def sample_student_rollouts(problem: str, seed: Optional[int] = None) -> dict:
    """Sample the student rollouts (cheap, no-grad half) for one wrong problem.

    Returns a dict carrying everything rollout_logits() needs:
        {
            "completions":       list[Tensor] — non-empty completion id tensors,
            "student_prefix_ids": Tensor      — [system, user: problem] prefix,
            "device":            torch.device,
        }

    Raises:
        RuntimeError: if every sampled rollout is empty (zero completion tokens).
    """
    model, tokenizer = get_model_and_tokenizer()
    device = next(model.parameters()).device

    student_prefix_ids = build_student_prefix_ids(problem, tokenizer).to(device)

    completion_id_lists = _vllm_rollouts(problem, config.NUM_ROLLOUTS, seed)
    completions = [
        torch.tensor(ids, dtype=torch.long, device=device)
        for ids in completion_id_lists if ids
    ]
    if not completions:
        raise RuntimeError(
            "All opsd student rollouts were empty (zero completion tokens)."
        )
    logger.debug(
        "opsd rollouts: %d/%d non-empty; lengths %s.",
        len(completions), config.NUM_ROLLOUTS, [c.shape[0] for c in completions],
    )
    return {
        "completions": completions,
        "student_prefix_ids": student_prefix_ids,
        "device": device,
    }


def rollout_logits(
    problem: str,
    reference_solution: str,
    sampled: dict,
    completion_ids: torch.Tensor,
) -> dict:
    """Run one rollout's teacher (no-grad) + student (grad) forwards.

    The teacher uses the *live* student weights (adapter on) conditioned on the
    reference solution — no adapter disable — so the target co-evolves with the
    student. Only one grad-carrying student forward is resident at a time.

    Returns {logits_with_hint, logits_without_hint, completion_tokens} matching
    the argument order loss_script.run_loss expects (teacher = with-hint target,
    student = without-hint, grad).
    """
    model, tokenizer = get_model_and_tokenizer()
    device = sampled["device"]

    teacher_prefix_ids = build_teacher_prefix_ids(problem, reference_solution, tokenizer).to(device)

    # Teacher (reference-conditioned) target — no grad, live weights.
    teacher_logits = _get_completion_logits(
        model, teacher_prefix_ids, completion_ids, device, no_grad=True
    )
    # Student (problem-only) — gradients flow.
    student_logits = _get_completion_logits(
        model, sampled["student_prefix_ids"], completion_ids, device, no_grad=False
    )
    assert teacher_logits.shape == student_logits.shape, (
        f"Logit shape mismatch: {teacher_logits.shape} vs {student_logits.shape}"
    )
    return {
        "logits_with_hint": teacher_logits,      # (N, V) — no grad, teacher/target
        "logits_without_hint": student_logits,   # (N, V) — has grad, student
        "completion_tokens": completion_ids.tolist(),
    }
