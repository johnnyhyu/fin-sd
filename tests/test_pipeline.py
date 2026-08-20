"""Unit tests for the pure-logic parts of the pipeline (no GPU / network)."""
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import config, eval_script, inference_script, loss_script, probing_script, state_finder_script


# ── loss_script ─────────────────────────────────────────────────────────────

def test_loss_identical_logits_is_zero():
    logits = torch.randn(5, 32)
    loss = loss_script.run_loss(logits, logits.clone())
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_loss_matches_manual_kl():
    with_hint = torch.tensor([[1.0, 0.0, -1.0, 2.0], [0.5, 0.5, 0.5, 0.5]])
    without_hint = torch.tensor([[0.0, 1.0, 1.0, 0.0], [2.0, -1.0, 0.0, 1.0]])

    p = torch.softmax(with_hint, dim=-1)
    q = torch.softmax(without_hint, dim=-1)
    manual = (p * (p.log() - q.log())).sum(dim=-1).mean()

    loss = loss_script.run_loss(with_hint, without_hint)
    assert loss.item() == pytest.approx(manual.item(), rel=1e-5)


def test_loss_gradient_flows_to_student():
    with_hint = torch.randn(3, 8)
    without_hint = torch.randn(3, 8, requires_grad=True)
    loss = loss_script.run_loss(with_hint, without_hint)
    loss.backward()
    assert without_hint.grad is not None
    assert torch.isfinite(without_hint.grad).all()


def test_loss_shape_mismatch_raises():
    with pytest.raises(ValueError):
        loss_script.run_loss(torch.randn(3, 8), torch.randn(4, 8))


# ── loss_script.run_reverse_loss (on-policy reverse KL) ──────────────────────

def test_reverse_loss_identical_logits_is_zero():
    logits = torch.randn(5, 32)
    loss = loss_script.run_reverse_loss(logits, logits.clone())
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_reverse_loss_matches_manual_reverse_kl():
    with_hint = torch.tensor([[1.0, 0.0, -1.0, 2.0], [0.5, 0.5, 0.5, 0.5]])
    without_hint = torch.tensor([[0.0, 1.0, 1.0, 0.0], [2.0, -1.0, 0.0, 1.0]])

    p = torch.softmax(with_hint, dim=-1)        # teacher
    q = torch.softmax(without_hint, dim=-1)     # student
    # Reverse KL is weighted by the student q (vs run_loss's forward KL by p).
    manual = (q * (q.log() - p.log())).sum(dim=-1).mean()

    loss = loss_script.run_reverse_loss(with_hint, without_hint)
    assert loss.item() == pytest.approx(manual.item(), rel=1e-5)


def test_reverse_loss_gradient_flows_to_student_only():
    with_hint = torch.randn(3, 8, requires_grad=True)
    without_hint = torch.randn(3, 8, requires_grad=True)
    loss = loss_script.run_reverse_loss(with_hint, without_hint)
    loss.backward()
    # Student carries gradient; teacher is detached.
    assert without_hint.grad is not None
    assert torch.isfinite(without_hint.grad).all()
    assert with_hint.grad is None


def test_reverse_loss_shape_mismatch_raises():
    with pytest.raises(ValueError):
        loss_script.run_reverse_loss(torch.randn(3, 8), torch.randn(4, 8))


# ── loss_script.run_exopd_loss (Generalized Off-Policy Distillation) ─────────

def test_exopd_loss_matches_manual_objective():
    with_hint = torch.tensor([[1.0, 0.0, -1.0, 2.0], [0.5, 0.5, 0.6, 0.4]])
    without_hint = torch.tensor([[0.0, 1.0, 1.0, 0.0], [2.0, -1.0, 0.0, 1.0]])
    ref = torch.tensor([[0.2, -0.3, 1.0, 0.1], [1.0, 0.0, -0.5, 0.5]])
    lam = 0.7

    log_star = torch.log_softmax(with_hint, dim=-1)
    log_ref = torch.log_softmax(ref, dim=-1)
    log_theta = torch.log_softmax(without_hint, dim=-1)
    q = log_theta.exp()
    # loss = -mean_pos( Σ_v q·[ λ(log* - log_ref) - (log_θ - log_ref) ] )
    j_pos = (q * (lam * (log_star - log_ref) - (log_theta - log_ref))).sum(dim=-1)
    manual = -j_pos.mean()

    loss = loss_script.run_exopd_loss(with_hint, without_hint, ref, lam)
    assert loss.item() == pytest.approx(manual.item(), rel=1e-5)


def test_exopd_loss_lambda_zero_is_forward_kl_to_ref():
    # With λ=0 the reward term drops and the loss is +KL(π_θ ‖ π_ref) ≥ 0.
    without_hint = torch.randn(5, 16)
    with_hint = torch.randn(5, 16)   # π* is irrelevant when λ=0
    ref = torch.randn(5, 16)

    log_theta = torch.log_softmax(without_hint, dim=-1)
    log_ref = torch.log_softmax(ref, dim=-1)
    kl = (log_theta.exp() * (log_theta - log_ref)).sum(dim=-1).mean()

    loss = loss_script.run_exopd_loss(with_hint, without_hint, ref, lam=0.0)
    assert loss.item() >= -1e-6
    assert loss.item() == pytest.approx(kl.item(), rel=1e-5)


def test_exopd_loss_zero_when_lambda_zero_and_student_equals_ref():
    # λ=0 and π_θ == π_ref ⇒ KL(π_θ ‖ π_ref) = 0.
    logits = torch.randn(4, 12)
    loss = loss_script.run_exopd_loss(
        torch.randn(4, 12), logits, logits.clone(), lam=0.0
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_exopd_loss_gradient_flows_to_student_only():
    with_hint = torch.randn(3, 8, requires_grad=True)
    without_hint = torch.randn(3, 8, requires_grad=True)
    ref = torch.randn(3, 8, requires_grad=True)
    loss = loss_script.run_exopd_loss(with_hint, without_hint, ref, lam=1.0)
    loss.backward()
    # Student carries gradient; teacher and reference are detached.
    assert without_hint.grad is not None
    assert torch.isfinite(without_hint.grad).all()
    assert with_hint.grad is None
    assert ref.grad is None


def test_exopd_loss_shape_mismatch_raises():
    with pytest.raises(ValueError):
        loss_script.run_exopd_loss(
            torch.randn(3, 8), torch.randn(3, 8), torch.randn(4, 8), lam=1.0
        )


# ── loss_script.run_reverse_topk_loss (top-k restricted reverse KL) ──────────

def test_reverse_topk_loss_identical_logits_is_zero():
    logits = torch.randn(5, 32)
    loss = loss_script.run_reverse_topk_loss(logits, logits.clone(), k=4)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_reverse_topk_loss_matches_manual_partial_sum():
    with_hint = torch.tensor([[1.0, 0.0, -1.0, 2.0], [0.5, 0.5, 0.6, 0.4]])
    without_hint = torch.tensor([[0.0, 1.0, 1.0, 0.0], [2.0, -1.0, 0.0, 1.0]])
    k = 2

    log_p = torch.log_softmax(with_hint, dim=-1)
    log_q = torch.log_softmax(without_hint, dim=-1)
    topk_ids = log_p.topk(k, dim=-1).indices
    log_p_k = log_p.gather(-1, topk_ids)
    log_q_k = log_q.gather(-1, topk_ids)
    # The loss renormalises both distributions over the top-k support (see
    # run_reverse_topk_loss), so the manual reference must too.
    log_p_k = log_p_k - torch.logsumexp(log_p_k, dim=-1, keepdim=True)
    log_q_k = log_q_k - torch.logsumexp(log_q_k, dim=-1, keepdim=True)
    manual = (log_q_k.exp() * (log_q_k - log_p_k)).sum(dim=-1).mean()

    loss = loss_script.run_reverse_topk_loss(with_hint, without_hint, k=k)
    assert loss.item() == pytest.approx(manual.item(), rel=1e-5)


def test_reverse_topk_loss_full_k_matches_full_reverse_kl():
    with_hint = torch.randn(4, 16)
    without_hint = torch.randn(4, 16)
    # k == vocab covers the whole support, so it reduces to the full reverse KL.
    topk = loss_script.run_reverse_topk_loss(with_hint, without_hint, k=16)
    full = loss_script.run_reverse_loss(with_hint, without_hint)
    assert topk.item() == pytest.approx(full.item(), rel=1e-5)


def test_reverse_topk_loss_clamps_k_above_vocab():
    with_hint = torch.randn(3, 8)
    without_hint = torch.randn(3, 8)
    loss = loss_script.run_reverse_topk_loss(with_hint, without_hint, k=100)
    full = loss_script.run_reverse_loss(with_hint, without_hint)
    assert loss.item() == pytest.approx(full.item(), rel=1e-5)


def test_reverse_topk_loss_gradient_flows_to_student_only():
    with_hint = torch.randn(3, 8, requires_grad=True)
    without_hint = torch.randn(3, 8, requires_grad=True)
    loss = loss_script.run_reverse_topk_loss(with_hint, without_hint, k=3)
    loss.backward()
    assert without_hint.grad is not None
    assert torch.isfinite(without_hint.grad).all()
    assert with_hint.grad is None


def test_reverse_topk_loss_shape_mismatch_raises():
    with pytest.raises(ValueError):
        loss_script.run_reverse_topk_loss(torch.randn(3, 8), torch.randn(4, 8), k=2)


# ── loss_script.run_reverse_topk_loss_from_teacher (vLLM top-k teacher) ───────

def _teacher_topk_from_logits(with_hint: torch.Tensor, k: int):
    """Build (ids, logprobs) as the vLLM top-k path would from full teacher logits."""
    log_p = torch.log_softmax(with_hint, dim=-1)
    topk = log_p.topk(k, dim=-1)
    return topk.indices, topk.values  # ids (N, k), full-vocab log P at those ids


def test_reverse_topk_from_teacher_matches_full_logit_version():
    # Deriving the teacher top-k from full teacher logits must reproduce
    # run_reverse_topk_loss exactly (the from-teacher form only changes how the
    # teacher distribution is supplied, not the maths).
    with_hint = torch.randn(4, 16)
    without_hint = torch.randn(4, 16)
    k = 5
    ids, logprobs = _teacher_topk_from_logits(with_hint, k)

    from_teacher = loss_script.run_reverse_topk_loss_from_teacher(ids, logprobs, without_hint)
    from_logits = loss_script.run_reverse_topk_loss(with_hint, without_hint, k=k)
    assert from_teacher.item() == pytest.approx(from_logits.item(), rel=1e-5)


def test_reverse_topk_from_teacher_identical_distributions_is_zero():
    # Teacher top-k drawn from the student's own distribution → renormalised P̃ and
    # Q̃ coincide over the support, so the divergence is zero.
    student = torch.randn(5, 12)
    k = 4
    ids, logprobs = _teacher_topk_from_logits(student, k)
    loss = loss_script.run_reverse_topk_loss_from_teacher(ids, logprobs, student.clone())
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_reverse_topk_from_teacher_mask_drops_padded_entries():
    # A masked-out column must be excluded from the renormalised support, so a
    # full-mask call over the first m columns equals a (m+pad) call masking the pad.
    with_hint = torch.randn(3, 16)
    without_hint = torch.randn(3, 16)
    ids4, lp4 = _teacher_topk_from_logits(with_hint, 4)

    # Mask the 4th column off on every row; the loss should match using only the
    # first 3 columns with a full mask.
    mask = torch.ones(3, 4)
    mask[:, 3] = 0.0
    masked = loss_script.run_reverse_topk_loss_from_teacher(ids4, lp4, without_hint, mask)
    unmasked3 = loss_script.run_reverse_topk_loss_from_teacher(
        ids4[:, :3], lp4[:, :3], without_hint
    )
    assert masked.item() == pytest.approx(unmasked3.item(), rel=1e-5)
    assert torch.isfinite(masked)


def test_reverse_topk_from_teacher_gradient_flows_to_student_only():
    ids, logprobs = _teacher_topk_from_logits(torch.randn(3, 8), k=3)
    student = torch.randn(3, 8, requires_grad=True)
    loss = loss_script.run_reverse_topk_loss_from_teacher(ids, logprobs, student)
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_reverse_topk_from_teacher_shape_mismatch_raises():
    with pytest.raises(ValueError):
        loss_script.run_reverse_topk_loss_from_teacher(
            torch.zeros(3, 4, dtype=torch.long), torch.zeros(3, 4), torch.randn(5, 8)
        )


# ── inference_script._parse_output ──────────────────────────────────────────

def test_parse_output_with_both_tags():
    raw = "<reasoning>\nstep 1\nstep 2\n</reasoning>\n<answer>\n$42\n</answer>"
    reasoning, answer = inference_script._parse_output(raw)
    assert reasoning == "step 1\nstep 2"
    assert answer == "$42"


def test_parse_output_missing_tags_falls_back_to_raw():
    raw = "  just some text  "
    reasoning, answer = inference_script._parse_output(raw)
    assert reasoning == "just some text"
    assert answer == "just some text"


def test_parse_output_answer_only():
    raw = "preamble <answer>7%</answer>"
    reasoning, answer = inference_script._parse_output(raw)
    assert answer == "7%"
    assert reasoning == "preamble"  # no reasoning tag → text before <answer>


def test_parse_output_reasoning_only():
    raw = "<reasoning>\nstep 1\n</reasoning>\nThe answer is 42."
    reasoning, answer = inference_script._parse_output(raw)
    assert reasoning == "step 1"
    assert answer == "The answer is 42."  # no answer tag → text after </reasoning>


def test_parse_output_reasoning_only_empty_tail_falls_back_to_raw():
    raw = "<reasoning>\nstep 1\n</reasoning>"
    reasoning, answer = inference_script._parse_output(raw)
    assert reasoning == "step 1"
    assert answer == raw.strip()


# ── inference_script.run_inference (reasoning channel) ──────────────────────

def _mock_vllm_response(message: dict, finish_reason: str = "stop"):
    return SimpleNamespace(
        json=lambda: {"choices": [{"message": message, "finish_reason": finish_reason}]}
    )


def test_run_inference_prefers_native_reasoning_channel():
    # gpt-oss via vLLM: chain-of-thought arrives in message.reasoning_content
    # and content holds only the final answer (no tags). Regression: reasoning
    # must not collapse to the answer text.
    message = {"content": "$42", "reasoning_content": "step 1\nstep 2"}
    with mock.patch.object(
        inference_script, "post_with_retry", return_value=_mock_vllm_response(message)
    ):
        out = inference_script.run_inference("problem")
    assert out["reasoning"] == "step 1\nstep 2"
    assert out["generated_answer"] == "$42"


def test_run_inference_combines_native_and_tagged_reasoning():
    message = {
        "content": "<reasoning>\nfinal-channel step\n</reasoning>\n<answer>\n7%\n</answer>",
        "reasoning": "analysis-channel step",
    }
    with mock.patch.object(
        inference_script, "post_with_retry", return_value=_mock_vllm_response(message)
    ):
        out = inference_script.run_inference("problem")
    assert out["reasoning"] == "analysis-channel step\n\nfinal-channel step"
    assert out["generated_answer"] == "7%"


def test_run_inference_without_reasoning_channel_unchanged():
    message = {"content": "<reasoning>\nstep 1\n</reasoning>\n<answer>\n$1\n</answer>"}
    with mock.patch.object(
        inference_script, "post_with_retry", return_value=_mock_vllm_response(message)
    ):
        out = inference_script.run_inference("problem")
    assert out["reasoning"] == "step 1"
    assert out["generated_answer"] == "$1"


# ── probing_script prefix normalisation ─────────────────────────────────────

def test_run_probing_strips_reasoning_tags_from_truncated_reasoning(monkeypatch):
    """If truncated_reasoning already wraps content in <reasoning> tags (e.g.
    because reasoning_content echoed them), run_probing must not double-wrap."""
    import types

    captured_prefixes: list[str] = []

    class _FakeTokenizer:
        bos_token = None
        eos_token_id = 1
        pad_token_id = 1

        def apply_chat_template(self, messages, **kwargs):
            # Record the assistant prefix so we can assert on it
            captured_prefixes.append(messages[-1]["content"])
            # prompts.py asks for return_tensors=None and then does list(ids), so
            # a real tokenizer hands back a flat list of ints there. Returning a
            # 2-D tensor unconditionally (as this stub once did) yields a list of
            # 1-D tensors, which torch.tensor() cannot index.
            if kwargs.get("return_tensors", "pt") is None:
                return [0, 0, 0, 0, 0]
            return torch.zeros(1, 5, dtype=torch.long)

        def encode(self, text, add_special_tokens=False):
            # prompts.py encodes the harmony channel header and message body
            # through the tokenizer; one id per character is enough for a stub
            # whose ids are never decoded, only counted and concatenated.
            return [ord(c) % 256 for c in text]

    class _FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros(1))
            self.config = types.SimpleNamespace(use_cache=False)

        def eval(self): return self
        def train(self, mode=True): return self
        def gradient_checkpointing_disable(self): pass
        def gradient_checkpointing_enable(self): pass

        def generate(self, input_ids, **kwargs):
            # Return one extra token
            return torch.cat([input_ids, torch.ones(1, 1, dtype=torch.long)], dim=1)

        def forward(self, full_ids, logits_to_keep=0):
            seq = full_ids.shape[1]
            logits = torch.zeros(1, seq, 4)
            if logits_to_keep:
                logits = logits[:, -logits_to_keep:, :]
            return types.SimpleNamespace(logits=logits)

    monkeypatch.setattr(probing_script, "_model", _FakeModel())
    monkeypatch.setattr(probing_script, "_tokenizer", _FakeTokenizer())
    # This test is about tag stripping, not prompt encoding. Under a gpt-oss path
    # prompts.py renders through openai_harmony rather than the chat template,
    # which the stub tokenizer above does not implement; pin a non-harmony model
    # so the assertion below sees the chat-template prefix it inspects.
    monkeypatch.setattr(config, "HF_MODEL_PATH", "meta-llama/Meta-Llama-3-8B-Instruct")

    # truncated_reasoning has stray <reasoning> tag — should be stripped
    result = probing_script.run_probing(
        problem="p",
        truncated_reasoning="<reasoning>\nstep 1\nstep 2",
        hint="check step 2",
    )
    assert result["completion_tokens"] == [1]
    # Neither prefix should start with double <reasoning>
    for p in captured_prefixes:
        assert not p.startswith("<reasoning>\n<reasoning>"), (
            f"Double-wrapped prefix: {p!r}"
        )
    # The with-hint prefix should contain just the stripped text
    with_hint = next(p for p in captured_prefixes if "[Hint]" in p)
    assert with_hint.startswith("<reasoning>\nstep 1"), f"Unexpected prefix: {with_hint!r}"


def test_run_probing_raises_on_empty_truncated_reasoning(monkeypatch):
    import types

    class _FakeTokenizer:
        bos_token = None
        eos_token_id = 1
        pad_token_id = 1

        def apply_chat_template(self, messages, **kwargs):
            return torch.zeros(1, 5, dtype=torch.long)

    class _FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros(1))
            self.config = types.SimpleNamespace(use_cache=False)

        def eval(self): return self
        def train(self, mode=True): return self
        def gradient_checkpointing_disable(self): pass
        def gradient_checkpointing_enable(self): pass

        def generate(self, input_ids, **kwargs):
            return torch.cat([input_ids, torch.ones(1, 1, dtype=torch.long)], dim=1)

        def forward(self, full_ids, logits_to_keep=0):
            seq = full_ids.shape[1]
            logits = torch.zeros(1, seq, 4)
            if logits_to_keep:
                logits = logits[:, -logits_to_keep:, :]
            return types.SimpleNamespace(logits=logits)

    monkeypatch.setattr(probing_script, "_model", _FakeModel())
    monkeypatch.setattr(probing_script, "_tokenizer", _FakeTokenizer())

    import pytest
    with pytest.raises(RuntimeError, match="empty"):
        probing_script.run_probing(problem="p", truncated_reasoning="   ", hint="h")


# ── state_finder_script truncation direction ─────────────────────────────────

# Tails long enough to clear state_finder's _MIN_TAIL_CHARS guard, which rejects
# a cut leaving under ~150 chars on the grounds that the quote named the
# final-answer line rather than the mistake.
_TAIL = ("\nAnnualizing the monthly figures across all twelve units gives the gross "
         "scheduled rent, from which the vacancy allowance is deducted before "
         "operating expenses and debt service are subtracted to reach the "
         "pre-tax cash flow used in the final yield calculation.")


def test_state_finder_find_not_rfind():
    """partial_ratio_alignment returns the leftmost (first) match on ties, so
    when the mistake sentence recurs after the error site, we still truncate at
    the first occurrence and exclude erroneous content."""
    reasoning = "Step 1: ok\nStep 2: ok\nStep 3: wrong\nStep 3: wrong again" + _TAIL
    # "Step 3: wrong" is the mistake sentence; it appears first before "Step 3: wrong again".
    # Truncation must cut before the first "Step 3: wrong".
    result = state_finder_script.run_state_finder(reasoning, "a hint", "Step 3: wrong")
    assert "Step 3: wrong" not in result
    assert result.strip() == "Step 1: ok\nStep 2: ok"


# ── eval_script ──────────────────────────────────────────────────────────────

def test_eval_system_prompt_has_no_escape_corruption():
    # Regression: \t in \text and \f in \frac must stay literal backslashes.
    assert r"\text" in eval_script._SYSTEM
    assert r"\frac" in eval_script._SYSTEM
    assert "\t" not in eval_script._SYSTEM
    assert "\f" not in eval_script._SYSTEM


@pytest.mark.parametrize("response,expected", [
    ("1", True),
    ("0", False),
    (" 1\n", True),
    ("1.", True),
    ("Verdict: 1", True),
    ("I would say 0", False),
])
def test_eval_parses_grader_response(response, expected):
    with mock.patch.object(eval_script, "openrouter_call", return_value=response):
        assert eval_script.run_eval("$21,600", "21600") is expected


def test_eval_raises_on_ungradeable_response():
    """A reply with no 0/1 anywhere is ungradeable, which is NOT evidence the
    answer was wrong — returning False there would send a correct item into
    hinting and training. run_eval raises so the caller can skip the item."""
    with mock.patch.object(eval_script, "openrouter_call", return_value="The answer is wrong"):
        with pytest.raises(RuntimeError, match="no 0/1 verdict"):
            eval_script.run_eval("$21,600", "21600")


# ── utils.parse_chat_completion ──────────────────────────────────────────────

from pipeline import utils


def test_parse_chat_completion_ok():
    data = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}
    assert utils.parse_chat_completion(data, label="t") == "hi"


@pytest.mark.parametrize("data", [
    {"error": {"message": "provider down"}},          # 200-with-error body
    {"choices": []},                                   # empty choices
    {},                                                # missing choices
    {"choices": [{"message": {"content": None}}]},     # null content
    {"choices": [{"message": {"content": "cut"}, "finish_reason": "length"}]},
])
def test_parse_chat_completion_rejects_bad_bodies(data):
    with pytest.raises(RuntimeError):
        utils.parse_chat_completion(data, label="t")


@pytest.mark.parametrize("message", [
    {"content": "$42", "reasoning_content": "cot"},   # vLLM-style channel
    {"content": "$42", "reasoning": "cot"},           # OpenRouter-style channel
])
def test_parse_chat_message_keeps_reasoning_channel(message):
    data = {"choices": [{"message": message, "finish_reason": "stop"}]}
    assert utils.parse_chat_message(data, label="t") == {"content": "$42", "reasoning": "cot"}


def test_parse_chat_message_allows_reasoning_only_message():
    data = {"choices": [{"message": {"content": None, "reasoning": "cot"}}]}
    assert utils.parse_chat_message(data, label="t")["reasoning"] == "cot"


# ── sft.generate_data trace parsing ─────────────────────────────────────────

from sft import generate_data as sft_gen


@pytest.mark.parametrize("content,native,expected", [
    # Teacher inlined the tags (no separate reasoning channel).
    ("<reasoning>\nstep 1\n</reasoning>\n<answer>\n$42\n</answer>", "", ("step 1", "$42")),
    # Reasoning model: chain-of-thought on its own channel, answer in content.
    ("<answer>$42</answer>", "cot", ("cot", "$42")),
    ("$42", "cot", ("cot", "$42")),
    # Both channels populated — native comes first (generation order).
    ("<reasoning>visible</reasoning><answer>$42</answer>", "cot", ("cot\n\nvisible", "$42")),
    # Off-format responses still yield reasoning, not just the answer.
    ("preamble steps\n<answer>$42</answer>", "", ("preamble steps", "$42")),
    ("<reasoning>steps</reasoning>\n$42", "", ("steps", "$42")),
    ("thinking a\nthinking b\n$42", "", ("thinking a\nthinking b", "$42")),
    # Nothing but a bare answer: no reasoning to keep.
    ("$42", "", ("", "$42")),
])
def test_split_trace(content, native, expected):
    assert sft_gen.split_trace(content, native) == expected


def test_record_completion_carries_reasoning_and_answer():
    rec = sft_gen._record(
        {"number": 1, "problem": "p", "answer": "$42"}, "step 1", "$42", correct=True
    )
    assert rec["reasoning"] == "step 1"
    assert rec["completion"] == "<reasoning>\nstep 1\n</reasoning>\n<answer>\n$42\n</answer>"
    assert sft_gen.extract_answer(rec["completion"]) == "$42"


# ── state_finder_script ─────────────────────────────────────────────────────

def test_state_finder_returns_truncation():
    reasoning = "Step 1: ok\nStep 2: ok\nStep 3: wrong" + _TAIL
    result = state_finder_script.run_state_finder(reasoning, "a hint", "Step 3: wrong")
    assert result.strip() == "Step 1: ok\nStep 2: ok"
    assert "Step 3: wrong" not in result


def test_state_finder_raises_when_tail_too_short():
    """A cut leaving almost nothing after it means the quote named the final
    answer, not the mistake; distilling that would treat the whole flawed chain
    as a correct prefix, so run_state_finder refuses rather than truncating."""
    reasoning = "Step 1: ok\nStep 2: ok\nStep 3: wrong"
    with pytest.raises(RuntimeError, match="chars follow the quoted sentence"):
        state_finder_script.run_state_finder(reasoning, "a hint", "Step 3: wrong")


def test_state_finder_raises_when_mistake_at_start():
    reasoning = "Step 1: wrong\nStep 2: ok"
    with pytest.raises(RuntimeError):
        state_finder_script.run_state_finder(reasoning, "a hint", "Step 1: wrong")


# ── probing_script._get_completion_logits slice math ─────────────────────────

class _StubModel:
    """Returns logits where logits[0, t, 0] == t, so positions are identifiable.

    Honors logits_to_keep the way a real CausalLM does: when set, only the last
    `logits_to_keep` positions are returned (their values still encode the
    absolute sequence position so the slice math stays checkable).
    """

    def __call__(self, full_ids, logits_to_keep=0):
        seq_len = full_ids.shape[1]
        vocab = 4
        logits = torch.zeros(1, seq_len, vocab)
        logits[0, :, 0] = torch.arange(seq_len, dtype=torch.float32)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


# ── probing_script LoRA target resolution (gpt-oss MoE coverage) ─────────────

class _FakeModel:
    """Stub exposing just what the resolver helpers touch: named_parameters,
    config and an active peft_config."""

    def __init__(self, param_names, num_experts=128, target_parameters=None):
        self._names = list(param_names)
        self.config = SimpleNamespace(num_local_experts=num_experts)
        if target_parameters is not None:
            self.peft_config = {"default": SimpleNamespace(target_parameters=target_parameters)}

    def named_parameters(self):
        return [(n, torch.zeros(1)) for n in self._names]


def test_resolve_target_parameters_finds_present_experts():
    m = _FakeModel([
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
        "model.layers.0.self_attn.q_proj.weight",
    ])
    out = probing_script._resolve_target_parameters(
        m, ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
    )
    assert out == ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]


def test_resolve_target_parameters_dense_model_returns_empty():
    m = _FakeModel(["model.layers.0.mlp.gate_proj.weight"], num_experts=0)
    assert probing_script._resolve_target_parameters(
        m, ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
    ) == []


def test_resolve_target_parameters_partial_match():
    m = _FakeModel(["model.layers.0.mlp.experts.gate_up_proj"])
    assert probing_script._resolve_target_parameters(
        m, ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
    ) == ["mlp.experts.gate_up_proj"]


def test_num_experts_reads_config():
    assert probing_script._num_experts(_FakeModel([], num_experts=128)) == 128
    assert probing_script._num_experts(_FakeModel([], num_experts=0)) is None


def test_peft_targets_parameters_detects_expert_targeting():
    targeting = _FakeModel([], target_parameters=["mlp.experts.gate_up_proj"])
    attn_only = _FakeModel([], target_parameters=[])
    assert probing_script._peft_targets_parameters(targeting) is True
    assert probing_script._peft_targets_parameters(attn_only) is False
    assert probing_script._peft_targets_parameters(_FakeModel([])) is False


@pytest.mark.parametrize("p,n", [(5, 3), (1, 1), (10, 4)])
def test_completion_logit_slice_offsets(p, n):
    prefix_ids = torch.arange(p)
    completion_ids = torch.arange(n)
    out = probing_script._get_completion_logits(
        _StubModel(), prefix_ids, completion_ids, torch.device("cpu"), no_grad=True
    )
    assert out.shape == (n, 4)
    # The logit predicting completion token i sits at sequence position p-1+i.
    expected_positions = torch.arange(p - 1, p - 1 + n, dtype=torch.float32)
    assert torch.equal(out[:, 0], expected_positions)


# ── probing_script._batched_completion_logits (real batching) ────────────────

class _BatchedStubModel:
    """Encodes each token's position id into logits[b, t, 0].

    Honors the left-padding contract the batched helper relies on: it reads the
    explicit position_ids (not the raw column index), so a correct left-pad +
    position-id construction yields position-equivariant logits identical to the
    per-item path. Also honors logits_to_keep like a real CausalLM.
    """

    def __call__(self, input_ids, attention_mask=None, position_ids=None, logits_to_keep=0):
        B, L = input_ids.shape
        vocab = 4
        logits = torch.zeros(B, L, vocab)
        if position_ids is None:
            position_ids = torch.arange(L).unsqueeze(0).expand(B, L)
        logits[:, :, 0] = position_ids.float()
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


def test_batched_completion_logits_matches_per_item():
    # Variable (prefix, completion) lengths so left-padding + tail-slice math is
    # exercised across items of different sizes in one batch.
    shapes = [(5, 3), (1, 1), (10, 4)]
    full_seqs = [torch.arange(p + n) for p, n in shapes]
    completion_lens = [n for _, n in shapes]

    out = probing_script._batched_completion_logits(
        _BatchedStubModel(), full_seqs, completion_lens,
        torch.device("cpu"), no_grad=True, pad_id=0,
    )

    assert len(out) == len(shapes)
    for (p, n), sliced in zip(shapes, out):
        assert sliced.shape == (n, 4)
        # Each item's completion logits must reproduce the per-item semantics:
        # the logit predicting completion token i sits at position p-1+i.
        expected_positions = torch.arange(p - 1, p - 1 + n, dtype=torch.float32)
        assert torch.equal(sliced[:, 0], expected_positions)


# ── utils backend concurrency gate (item-level batching) ─────────────────────

def test_concurrency_gate_routes_by_label():
    from pipeline import utils

    # vLLM rollout + teacher decode share the vLLM cap; eval + hint share OpenRouter.
    assert utils._concurrency_gate("vLLM") is utils._VLLM_SEM
    assert utils._concurrency_gate("vLLM-probe") is utils._VLLM_SEM
    assert utils._concurrency_gate("OpenRouter") is utils._OPENROUTER_SEM
    # Unknown labels are ungated (a plain context manager, not a semaphore).
    other = utils._concurrency_gate("LoRA-sync")
    assert other is not utils._VLLM_SEM and other is not utils._OPENROUTER_SEM


def test_post_with_retry_respects_backend_cap(monkeypatch):
    """Concurrent calls for one backend never exceed its semaphore cap."""
    import threading
    import time
    from pipeline import utils

    cap = 2
    monkeypatch.setattr(utils, "_VLLM_SEM", threading.BoundedSemaphore(cap))

    lock = threading.Lock()
    active = 0
    peak = 0

    def fake_post(url, headers=None, json=None, timeout=None):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)  # hold the slot long enough for contention to build
        with lock:
            active -= 1
        return SimpleNamespace(status_code=200, ok=True, text="", json=lambda: {})

    monkeypatch.setattr(utils.requests, "post", fake_post)

    def call():
        utils.post_with_retry("http://x", headers={}, payload={}, timeout=5, label="vLLM")

    threads = [threading.Thread(target=call) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert peak <= cap          # cap is never exceeded
    assert peak >= 2            # and calls really did run concurrently up to it


# ── vLLM lifecycle: shutdown, port handoff, GPU routing ──────────────────────

def _free_port() -> int:
    """An ephemeral port number that nothing is bound to right now."""
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_wait_down_waits_for_the_socket_not_for_readiness(monkeypatch):
    """A bound-but-unhealthy server must NOT read as 'down'.

    Regression: wait_down() polled is_ready(), which reports False the moment the
    engine dies even though the API front-end still owns the port. restart() then
    raced ahead into a start() that could not bind.
    """
    import socket
    from pipeline import config, vllm_server

    port = _free_port()
    monkeypatch.setattr(config, "VLLM_BASE_URL", f"http://127.0.0.1:{port}/v1")
    # Something is listening, but it answers nothing — i.e. not "ready" by any probe.
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    try:
        monkeypatch.setattr(vllm_server, "is_ready", lambda: False)
        with pytest.raises(RuntimeError, match="still bound"):
            vllm_server.wait_down(timeout=1)
    finally:
        srv.close()
    # Once the socket is released the port is free and wait_down returns promptly.
    vllm_server.wait_down(timeout=5)


def test_port_bindable_ignores_time_wait(monkeypatch):
    """A port whose last connection is in TIME_WAIT is still bindable by vLLM.

    _port_bindable sets SO_REUSEADDR to match uvicorn, so a just-stopped server's
    lingering connections don't strand wait_down() or drift restart() to a new port.
    """
    import socket
    from pipeline import vllm_server

    port = _free_port()
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    # A live listener really is occupied …
    assert vllm_server._port_bindable("127.0.0.1", port) is False
    client = socket.create_connection(("127.0.0.1", port))
    conn, _ = srv.accept()
    client.close()
    conn.close()
    srv.close()
    # … but its closed connections (TIME_WAIT) must not keep it looking occupied.
    assert vllm_server._port_bindable("127.0.0.1", port) is True


def test_is_ready_rejects_a_healthy_server_that_isnt_ours(monkeypatch):
    """A 200 from /health can't be trusted once our own subprocess has exited."""
    from pipeline import vllm_server

    monkeypatch.setattr(
        vllm_server.requests, "get",
        lambda *a, **k: SimpleNamespace(status_code=200),
    )
    # No managed process: any healthy server on the port is fair game.
    monkeypatch.setattr(vllm_server, "_proc", None)
    assert vllm_server.is_ready() is True
    # Our process died, so whatever is answering is an orphan serving unknown
    # weights — refuse it rather than train against it.
    monkeypatch.setattr(vllm_server, "_proc", SimpleNamespace(poll=lambda: 1))
    assert vllm_server.is_ready() is False
    # Still running → its 200 is ours.
    monkeypatch.setattr(vllm_server, "_proc", SimpleNamespace(poll=lambda: None))
    assert vllm_server.is_ready() is True


def test_stop_reaps_orphaned_engine_children(monkeypatch, tmp_path):
    """stop() must not return while VRAM-holding grandchildren are still alive.

    Models vLLM's shape: a launcher that exits immediately while its engine/worker
    children keep running in the same process group.
    """
    import contextlib
    import os
    import signal
    import subprocess
    import sys
    import time
    from pipeline import vllm_server

    # Launcher spawns a long-lived child, then exits at once — the child is
    # re-parented but stays in the launcher's process group, as vLLM's workers do.
    # The child IGNORES SIGTERM, so only the group-wide SIGKILL escalation clears
    # it: waiting on the launcher alone (the old behaviour) would leave it running
    # with its GPU memory still allocated.
    marker = tmp_path / "child.pid"
    child = ("import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
             f"open({str(marker)!r},'w').write(str(__import__('os').getpid())); "
             "time.sleep(120)")
    launcher = subprocess.Popen(
        [sys.executable, "-c",
         f"import subprocess,sys; subprocess.Popen([sys.executable,'-c',{child!r}])"],
        start_new_session=True,
    )
    pgid = os.getpgid(launcher.pid)
    # Wait for the grandchild to actually exist, so stop() is exercised against a
    # populated group rather than racing the launcher's spawn.
    deadline = time.time() + 10
    while time.time() < deadline and not marker.exists():
        time.sleep(0.05)
    assert marker.exists(), "grandchild never started; test would prove nothing"
    child_pid = int(marker.read_text())
    monkeypatch.setattr(vllm_server, "_proc", launcher)
    monkeypatch.setattr(vllm_server, "_pgid", pgid)
    # Keep the test quick: shorten the grace period before the SIGKILL escalation,
    # but keep the real polling behaviour (an immediate one-shot check would race
    # the kernel reaping the killed child).
    real_wait = vllm_server._wait_pgroup_gone
    monkeypatch.setattr(vllm_server, "_wait_pgroup_gone",
                        lambda pgid, timeout: real_wait(pgid, min(timeout, 1)))
    try:
        vllm_server.stop()
        # The whole group is gone, not just the launcher we could wait() on — the
        # SIGTERM-ignoring grandchild included.
        assert vllm_server._pgroup_alive(pgid) is False
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        assert vllm_server._proc is None
    finally:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)


def test_tensor_parallel_follows_the_visible_gpus(monkeypatch):
    """The serving degree tracks VLLM_GPUS, not the 120B-sized default."""
    from pipeline import config, vllm_server

    monkeypatch.setattr(config, "VLLM_TENSOR_PARALLEL", 2)
    # A 20B run pinned to one card: serving with the configured 2 would abort vLLM.
    monkeypatch.setattr(config, "VLLM_GPUS", "3")
    assert vllm_server._tensor_parallel_size() == 1
    # Matching config is passed through untouched.
    monkeypatch.setattr(config, "VLLM_GPUS", "0,1")
    assert vllm_server._tensor_parallel_size() == 2
    # Empty GPU list (inherit whatever is visible) falls back to the config value.
    monkeypatch.setattr(config, "VLLM_GPUS", "")
    assert vllm_server._tensor_parallel_size() == 2


@pytest.mark.parametrize("path,expected", [
    ("unsloth/gpt-oss-20b-BF16", "gpt-oss-20b"),
    ("unsloth/gpt-oss-20b", "gpt-oss-20b"),
    ("openai/gpt-oss-20b", "gpt-oss-20b"),
    ("/models/GPT-OSS-20B-MXFP4", "gpt-oss-20b"),
    ("unsloth/gpt-oss-120b-BF16", "gpt-oss-120b"),
    ("meta-llama/Llama-3-70B", None),
    ("", None),
])
def test_model_family_collapses_spellings(path, expected):
    """Every spelling of one model must route identically (launch.py placement)."""
    from pipeline import utils

    assert utils.model_family(path) == expected


def test_same_model_family_flags_only_confident_mismatches():
    from pipeline import utils

    # BF16 trainer base vs MXFP4 served release: the intended pairing.
    assert utils.same_model_family("unsloth/gpt-oss-20b-BF16", "unsloth/gpt-oss-20b")
    # 120B adapter on the default 20B base: the bug this guards.
    assert not utils.same_model_family("unsloth/gpt-oss-120b-BF16", "unsloth/gpt-oss-20b")
    # Unrecognised paths are not evidence of a mismatch.
    assert utils.same_model_family("/scratch/my-finetune", "unsloth/gpt-oss-20b")


def test_merge_gpu_count_is_sized_by_the_model(monkeypatch):
    """The in-process BF16 merge is sized by the model, not the serving degree.

    With HF_MAX_MEMORY empty — the default — the block is computed from the
    model's BF16 weight footprint against the cards the driver reports.
    """
    from pipeline import config, utils

    monkeypatch.setattr(config, "HF_MAX_MEMORY", "")
    monkeypatch.setattr(config, "HF_MIN_HEADROOM_GIB", 8.0)
    # Pretend to an 8x80 GB box so the arithmetic is checkable off-GPU.
    monkeypatch.setattr(utils, "gpu_memory_gib", lambda: {i: (80.0, 80.0) for i in range(8)})

    monkeypatch.setattr(config, "HF_MODEL_PATH", "unsloth/gpt-oss-20b-BF16")
    assert utils.hf_merge_gpu_count() == 1     # ~37 GiB fits one 72 GiB-usable card
    monkeypatch.setattr(config, "HF_MODEL_PATH", "unsloth/gpt-oss-120b-BF16")
    assert utils.hf_merge_gpu_count() == 4     # ~224 GiB over 72 GiB per card


def test_merge_gpu_count_honours_an_explicit_cap(monkeypatch):
    """An explicit HF_MAX_MEMORY wins: the operator has stated the block width.

    Guards the precedence the model-derived path must not override.
    """
    from pipeline import config, utils

    monkeypatch.setattr(config, "HF_MAX_MEMORY", "40,40,40,40,40,70")
    monkeypatch.setattr(config, "HF_MODEL_PATH", "unsloth/gpt-oss-20b-BF16")
    assert utils.hf_merge_gpu_count() == 6     # six caps, six cards, model ignored


def test_lock_path_survives_a_long_name(tmp_path, monkeypatch):
    """Path-derived lock names must not blow past the 255-byte filename limit."""
    from pipeline import utils

    monkeypatch.setattr(utils, "_LOCK_DIR", tmp_path)
    long_name = "merge_" + "a" * 400
    path = utils._lock_path(long_name)
    assert len(path.name) < 255
    # Distinct long names keep distinct locks (no collision onto one file) …
    assert utils._lock_path(long_name + "b") != path
    # … and the mapping is stable, so separate processes agree on the same file.
    assert utils._lock_path(long_name) == path
    # It is actually usable as a lock.
    with utils.file_lock(long_name):
        pass


def test_configure_serve_gpus_splits_merge_block_from_serving_cards(monkeypatch):
    """A wide merge block and a narrow serving list can coexist on one reservation.

    serve_checkpoint's 120B merge needs more cards than vLLM then serves on. All
    reserved cards become visible for the merge; VLLM_GPUS gets only the serving
    ones, so --tensor-parallel-size (derived from that list) still matches.
    """
    from pipeline import config, utils, vllm_server

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("VLLM_GPUS", raising=False)
    monkeypatch.setattr(config, "VLLM_TENSOR_PARALLEL", 2)
    monkeypatch.setattr(config, "VLLM_GPUS", "0,1")
    monkeypatch.setattr(utils, "reserve_open_gpus", lambda n, max_used_gib: list(range(n)))

    gpus = utils.configure_serve_gpus(set_visible=True, min_gpus=6)

    assert gpus == [0, 1, 2, 3, 4, 5]                    # whole merge block reserved
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5"   # merge sees them all
    assert config.VLLM_GPUS == "0,1"                     # vLLM serves on tp of them
    assert vllm_server._tensor_parallel_size() == 2      # and the degree agrees


# ── LR schedule (warmup × epoch decay) ──────────────────────────────────────

def test_warmup_factor_ramps_to_one_and_holds(monkeypatch):
    from pipeline import config, optimization_script as opt

    monkeypatch.setattr(config, "NUM_WARMUP_STEPS", 2)
    # Step 0 is the first update: the ramp uses (step + 1) so it is never zero.
    assert opt._warmup_factor(0) == pytest.approx(0.5)
    assert opt._warmup_factor(1) == pytest.approx(1.0)
    assert opt._warmup_factor(9) == pytest.approx(1.0)   # holds after warmup


def test_warmup_disabled_when_zero_steps(monkeypatch):
    from pipeline import config, optimization_script as opt

    monkeypatch.setattr(config, "NUM_WARMUP_STEPS", 0)
    assert opt._warmup_factor(0) == pytest.approx(1.0)


def test_decay_factor_starts_at_peak_and_anneals(monkeypatch):
    from pipeline import config, optimization_script as opt

    monkeypatch.setattr(config, "NUM_EPOCHS", 10)
    monkeypatch.setattr(config, "LR_DECAY_FLOOR", 0.1)

    assert opt._decay_factor(1) == pytest.approx(1.0)     # no decay in epoch 1
    # Monotonically decreasing across the run, and never below the floor.
    factors = [opt._decay_factor(e) for e in range(1, 11)]
    assert all(a > b for a, b in zip(factors, factors[1:]))
    assert min(factors) > config.LR_DECAY_FLOOR
    assert factors[-1] < 0.2


def test_decay_disabled_when_floor_is_one(monkeypatch):
    from pipeline import config, optimization_script as opt

    monkeypatch.setattr(config, "NUM_EPOCHS", 10)
    monkeypatch.setattr(config, "LR_DECAY_FLOOR", 1.0)
    assert [opt._decay_factor(e) for e in (1, 5, 10)] == [1.0, 1.0, 1.0]


def test_lr_lambda_is_warmup_times_decay(monkeypatch):
    from pipeline import config, optimization_script as opt

    monkeypatch.setattr(config, "NUM_WARMUP_STEPS", 2)
    monkeypatch.setattr(config, "NUM_EPOCHS", 10)
    monkeypatch.setattr(config, "LR_DECAY_FLOOR", 0.1)
    monkeypatch.setattr(opt, "_current_epoch", 4)

    assert opt._lr_lambda(5) == pytest.approx(opt._warmup_factor(5) * opt._decay_factor(4))


def test_schedule_spends_most_of_a_short_run_near_peak(monkeypatch):
    """Regression on the original bug: 15 warmup steps of a ~20-step run.

    The point of the new defaults is that a short run is no longer held below
    peak LR for most of its steps. Compare the summed multiplier over 20 steps.
    """
    from pipeline import config, optimization_script as opt

    monkeypatch.setattr(config, "NUM_EPOCHS", 10)
    monkeypatch.setattr(config, "LR_DECAY_FLOOR", 1.0)   # isolate the warmup shape

    monkeypatch.setattr(config, "NUM_WARMUP_STEPS", 15)
    old = sum(opt._warmup_factor(s) for s in range(20))
    monkeypatch.setattr(config, "NUM_WARMUP_STEPS", 2)
    new = sum(opt._warmup_factor(s) for s in range(20))

    assert old == pytest.approx(13.0, abs=0.1)    # 65% of peak on average
    assert new == pytest.approx(19.5, abs=0.1)    # 97.5%


def test_set_epoch_updates_live_lr_immediately(monkeypatch):
    """The new epoch's LR must apply to that epoch's FIRST step, not its second."""
    from pipeline import config, optimization_script as opt

    monkeypatch.setattr(config, "NUM_WARMUP_STEPS", 2)
    monkeypatch.setattr(config, "NUM_EPOCHS", 10)
    monkeypatch.setattr(config, "LR_DECAY_FLOOR", 0.1)

    param = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.AdamW([param], lr=2e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, opt._lr_lambda)
    monkeypatch.setattr(opt, "_optimizer", optimizer)
    monkeypatch.setattr(opt, "_scheduler", scheduler)

    opt.set_epoch(1)
    lr_epoch_1 = opt.current_lr()
    opt.set_epoch(8)
    assert opt.current_lr() < lr_epoch_1
    assert opt.current_lr() == pytest.approx(
        2e-5 * opt._warmup_factor(scheduler.last_epoch) * opt._decay_factor(8)
    )


# ── Token-budget penalty ────────────────────────────────────────────────────

def test_length_penalty_weight_is_zero_below_threshold(monkeypatch):
    from pipeline import config

    monkeypatch.setattr(config, "MAX_NEW_TOKENS", 1000)
    monkeypatch.setattr(config, "LENGTH_SOFT_FRAC", 0.7)
    monkeypatch.setattr(config, "LENGTH_PENALTY_POWER", 2.0)

    # Nothing below the soft threshold contributes — this is what keeps the term
    # from ever rewarding an early stop.
    assert config.length_penalty_weight(0) == 0.0
    assert config.length_penalty_weight(699) == 0.0
    # …and the ramp is sharp above it.
    assert config.length_penalty_weight(799) == pytest.approx((0.1 / 0.3) ** 2)
    assert config.length_penalty_weight(949) == pytest.approx((0.25 / 0.3) ** 2)
    assert config.length_penalty_weight(999) == pytest.approx(1.0)


def test_length_penalty_weight_clamps_past_the_budget(monkeypatch):
    from pipeline import config

    monkeypatch.setattr(config, "MAX_NEW_TOKENS", 100)
    monkeypatch.setattr(config, "LENGTH_SOFT_FRAC", 0.7)
    assert config.length_penalty_weight(500) == pytest.approx(1.0)


def test_length_penalty_matches_manual_weighted_nll():
    torch.manual_seed(0)
    logits = torch.randn(4, 16)
    end_ids = torch.tensor([3, 9])
    weights = torch.tensor([0.0, 0.25, 0.5, 1.0])

    got = loss_script.run_length_penalty(logits, end_ids, weights)

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    p_end = log_probs[:, end_ids].exp().sum(dim=-1)
    expected = (weights * -p_end.log()).mean()
    assert got.item() == pytest.approx(expected.item(), rel=1e-5)


def test_length_penalty_falls_as_end_token_becomes_likely():
    end_ids = torch.tensor([0])
    weights = torch.ones(1)
    reluctant = torch.tensor([[0.0, 10.0, 10.0]])
    willing = torch.tensor([[10.0, 0.0, 0.0]])

    assert (
        loss_script.run_length_penalty(willing, end_ids, weights).item()
        < loss_script.run_length_penalty(reluctant, end_ids, weights).item()
    )


def test_length_penalty_zero_weights_give_zero_loss_and_gradient():
    logits = torch.randn(3, 8, requires_grad=True)
    loss = loss_script.run_length_penalty(logits, torch.tensor([1]), torch.zeros(3))
    loss.backward()
    assert loss.item() == pytest.approx(0.0)
    assert torch.count_nonzero(logits.grad) == 0


def test_length_penalty_gradient_is_bounded_even_at_vanishing_probability():
    """Why the term is left uncapped: it is a cross-entropy, so however certain
    the model is about continuing, the logit-space gradient stays bounded."""
    logits = torch.tensor([[-100.0, 0.0, 0.0]], requires_grad=True)
    loss = loss_script.run_length_penalty(logits, torch.tensor([0]), torch.ones(1))
    loss.backward()
    assert torch.isfinite(loss)
    assert loss.item() > 50                      # value is large …
    assert logits.grad.abs().sum().item() <= 2.0  # … the gradient is not


def test_length_penalty_rejects_empty_end_token_set():
    with pytest.raises(ValueError, match="no turn-ending token"):
        loss_script.run_length_penalty(
            torch.randn(2, 8), torch.tensor([], dtype=torch.long), torch.ones(2)
        )


def test_length_penalty_rejects_weight_shape_mismatch():
    with pytest.raises(ValueError, match="position_weights"):
        loss_script.run_length_penalty(torch.randn(4, 8), torch.tensor([1]), torch.ones(3))


# ── inference_script token-budget plumbing ──────────────────────────────────

def _mock_vllm_response_with_tokens(message, finish_reason, token_ids, usage=None):
    body = {
        "choices": [{
            "message": message,
            "finish_reason": finish_reason,
            "logprobs": {"content": [{"token": f"token_id:{i}"} for i in token_ids]},
        }],
    }
    if usage is not None:
        body["usage"] = usage
    return SimpleNamespace(json=lambda: body)


def test_run_inference_reports_truncation_and_token_ids(monkeypatch):
    from pipeline import config

    monkeypatch.setattr(config, "LENGTH_PENALTY", True)
    monkeypatch.setattr(config, "MAX_NEW_TOKENS", 8)
    resp = _mock_vllm_response_with_tokens(
        {"content": "", "reasoning_content": "still thinking"}, "length", [11, 22, 33, 44]
    )
    with mock.patch.object(inference_script, "post_with_retry", return_value=resp):
        out = inference_script.run_inference("problem")

    assert out["finish_reason"] == "length"
    assert out["completion_tokens"] == [11, 22, 33, 44]
    assert out["n_completion_tokens"] == 4
    assert out["budget_frac"] == pytest.approx(0.5)


def test_run_inference_falls_back_to_usage_for_length(monkeypatch):
    """No logprobs (penalty off, or the server withheld them) must not lose the
    length signal entirely — budget_frac still comes from the usage block."""
    from pipeline import config

    monkeypatch.setattr(config, "LENGTH_PENALTY", False)
    monkeypatch.setattr(config, "MAX_NEW_TOKENS", 100)
    resp = _mock_vllm_response_with_tokens(
        {"content": "<answer>7</answer>"}, "stop", [], usage={"completion_tokens": 40}
    )
    with mock.patch.object(inference_script, "post_with_retry", return_value=resp):
        out = inference_script.run_inference("problem")

    assert out["completion_tokens"] == []
    assert out["n_completion_tokens"] == 40
    assert out["budget_frac"] == pytest.approx(0.4)


def test_run_inference_survives_unparseable_token_ids(monkeypatch):
    """A server that ignores return_tokens_as_token_ids costs us the length term
    for that item, not the item."""
    from pipeline import config

    monkeypatch.setattr(config, "LENGTH_PENALTY", True)
    resp = SimpleNamespace(json=lambda: {"choices": [{
        "message": {"content": "<answer>7</answer>"},
        "finish_reason": "stop",
        "logprobs": {"content": [{"token": " the"}]},
    }]})
    with mock.patch.object(inference_script, "post_with_retry", return_value=resp):
        out = inference_script.run_inference("problem")

    assert out["completion_tokens"] == []
    assert out["generated_answer"] == "7"


def test_run_inference_requests_token_ids_only_when_penalty_is_on(monkeypatch):
    from pipeline import config

    seen = {}

    def _capture(url, *, headers, payload, timeout, label):
        seen.update(payload)
        return _mock_vllm_response_with_tokens({"content": "x"}, "stop", [1])

    monkeypatch.setattr(config, "LENGTH_PENALTY", False)
    with mock.patch.object(inference_script, "post_with_retry", _capture):
        inference_script.run_inference("problem")
    assert "logprobs" not in seen

    seen.clear()
    monkeypatch.setattr(config, "LENGTH_PENALTY", True)
    with mock.patch.object(inference_script, "post_with_retry", _capture):
        inference_script.run_inference("problem")
    assert seen["logprobs"] is True and seen["return_tokens_as_token_ids"] is True
