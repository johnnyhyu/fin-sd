"""Student model + tokenizer construction for SFT.

Mirrors the main pipeline's HF/PEFT loading conventions (bf16 upcast, sharded
`device_map="auto"` with per-GPU memory ceilings, attention-Linear LoRA plus
optional gpt-oss packed-expert LoRA), but stands alone so an SFT run doesn't
touch the RL pipeline's singleton or its objective-specific settings.

One thing it does NOT inherit is the attention kernel. gpt-oss's autoselect lands
on `eager`, whose [batch, heads, seq, seq] score matrix is what OOMed the 20B
mid-epoch — 9.4 GiB for ONE 8875-token example, on top of the weights and the
optimizer state. That diagnosis and its fix now live in `pipeline.attention`,
shared with the RL pipeline (which had the same latent bug); see that module for
why picking the kernel is more than a one-liner. `SFT_ATTN_IMPLEMENTATION` still
pins it for this flow.
"""
from __future__ import annotations

from typing import Optional

import torch

from pipeline import attention
from pipeline.utils import logger, resolve_max_memory

from . import config

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

# Attention (+ dense-MLP) Linear modules that exist on most instruct models.
_LORA_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# gpt-oss MoE experts, packed as 3-D nn.Parameters (reached via target_parameters).
_LORA_EXPERT_PARAMS = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]


def _present_linear(model, candidates: list[str]) -> list[str]:
    present = {
        name.rsplit(".", 1)[-1]
        for name, m in model.named_modules()
        if name.rsplit(".", 1)[-1] in candidates and isinstance(m, torch.nn.Linear)
    }
    matched = [c for c in candidates if c in present]
    if not matched:
        raise RuntimeError(
            f"None of {candidates} are Linear modules of {type(model).__name__}; "
            "LoRA would train nothing."
        )
    return matched


def _present_params(model, candidates: list[str]) -> list[str]:
    names = [n for n, _ in model.named_parameters()]
    return [c for c in candidates if any(n.endswith(c) for n in names)]


def _num_experts(model) -> Optional[int]:
    cfg = getattr(model, "config", None)
    for attr in ("num_local_experts", "num_experts"):
        n = getattr(cfg, attr, None)
        if isinstance(n, int) and n > 0:
            return n
    return None


# ── Model ───────────────────────────────────────────────────────────────────


def build_model_and_tokenizer():
    """Load the student, attach LoRA (if enabled), and return (model, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = _DTYPES.get(config.HF_DTYPE, torch.bfloat16)
    # Per-card ceilings for the sharded load: derived from the cards' live free VRAM
    # minus the reserve a step needs, unless SFT_HF_MAX_MEMORY pins them (shared with
    # the RL pipeline — see pipeline.utils.resolve_max_memory).
    max_memory = resolve_max_memory(config.HF_MAX_MEMORY,
                                    model_path=config.STUDENT_MODEL,
                                    label="SFT_HF_MAX_MEMORY")
    attn = attention.resolve_implementation(
        config.ATTN_IMPLEMENTATION, env_var="SFT_ATTN_IMPLEMENTATION"
    )
    logger.info(
        "Loading student %s (dtype=%s, device_map=%s, max_memory=%s, attn=%s) …",
        config.STUDENT_MODEL, config.HF_DTYPE, config.HF_DEVICE, max_memory,
        attn or "auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(config.STUDENT_MODEL, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    load_kwargs = dict(
        torch_dtype=dtype,
        device_map=config.HF_DEVICE,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    if attn:
        load_kwargs["attn_implementation"] = attn

    model = AutoModelForCausalLM.from_pretrained(config.STUDENT_MODEL, **load_kwargs)
    model.config.use_cache = False

    # Read back what the model ACTUALLY loaded with (an unsupported value is
    # silently downgraded) and keep FlexAttention fused if that is what we got.
    attention.apply_to_loaded(model, attn)

    if config.USE_LORA:
        model = _attach_lora(model)
    else:
        logger.info("Full fine-tuning (no LoRA).")

    return model, tokenizer


def _attach_lora(model):
    from peft import LoraConfig, get_peft_model

    model.enable_input_require_grads()  # needed for grad checkpointing + frozen base
    lora_kwargs = dict(
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=_present_linear(model, _LORA_LINEAR),
    )

    if config.LORA_TARGET_EXPERTS and "target_parameters" in getattr(
        LoraConfig, "__dataclass_fields__", {}
    ):
        expert_params = _present_params(model, _LORA_EXPERT_PARAMS)
        if expert_params:
            n_exp = _num_experts(model) or 1
            expert_rank = max(1, config.LORA_RANK // n_exp)
            lora_kwargs["target_parameters"] = expert_params
            lora_kwargs["rank_pattern"] = {p: expert_rank for p in expert_params}
            logger.info(
                "LoRA adapting %d MoE expert group(s) at per-expert rank %d.",
                len(expert_params), expert_rank,
            )

    model = get_peft_model(model, LoraConfig(**lora_kwargs))
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
        logger.info(
            "LoRA attached: %s trainable / %s total (%.4f%%).",
            f"{trainable:,}", f"{total:,}", 100 * trainable / total,
        )
    return model
