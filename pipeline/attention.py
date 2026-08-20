"""Attention-kernel selection for the HF training/probing models.

gpt-oss declares `_supports_sdpa = False` — its attention sinks are an extra logit
per head that SDPA's fused kernels have no way to express — so transformers'
autoselect skips SDPA. With flash-attn absent (it is not in requirements.txt: the
wheel is built per-CUDA and the box may not have one) the fallback is `eager`,
which builds the score matrix in memory: `[batch, heads, seq, seq]` is
1 × 64 × 8192 × 8192 × 2 B = 8 GiB for ONE 8192-token example, and the softmax
intermediates multiply that. Measured on the 120B's per-layer geometry, eager at
8192 tokens costs ~30 GiB of transient VRAM per layer on top of everything else —
more than the entire rest of the layer's footprint — and it happens silently,
because eager is a legitimate choice for short sequences.

This module is the one place that picks the kernel, shared by the RL pipeline
(pipeline.probing_script, tools.profile_experts) and the SFT flow (sft.model),
which discovered the problem first and where the two functions below were
originally written. Keeping one copy means a fix found on either path applies to
both.

`resolve_implementation` picks the kernel explicitly, and `keep_flex_attention_fused`
keeps FlexAttention on the fused path it is worth using in the first place. See
those two for why each is more than a one-liner.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

from .utils import logger

# Preference order for the autoselect. Both are memory-linear in sequence length;
# flash-attn is faster where its wheel exists, FlexAttention ships with torch and
# is the one that actually expresses gpt-oss's sinks (transformers applies them
# after the kernel — see integrations/flex_attention.flex_attention_forward).
# `eager` is deliberately absent: falling into it is the bug this list exists for.
_ATTN_PREFERENCE = ("flash_attention_2", "flex_attention")

# Dynamo's per-code-object recompile ceiling while FlexAttention is in use.
#
# transformers compiles `flex_attention` ONCE into a process-wide singleton
# (integrations/flex_attention.WrappedFlexAttention), so every layer and every
# shard shares one code object and one cache. Guards trip on the query/key shapes
# and on the BlockMask's device, and both flows train variable-length examples at
# batch size 1, so each new length bucket costs an entry — the 20B burned all 8
# of the default budget inside 14 steps.
#
# What makes that a memory bug rather than a speed bug: on overflow dynamo does
# not raise, it drops to running `flex_attention` unfused, whose fallback
# materialises the same [batch, heads, seq, seq] score matrix eager would. The
# run reverts to the OOM profile mid-epoch with nothing in the log but a
# recompile warning. The ceiling is raised (not disabled) so a genuine
# recompile storm still surfaces instead of silently costing compile time.
_DYNAMO_CACHE_LIMIT = int(os.getenv("HF_DYNAMO_CACHE_LIMIT",
                                    os.getenv("SFT_DYNAMO_CACHE_LIMIT", "128")))

# Set once `keep_flex_attention_fused` has patched the attention entry, so a
# second model build in one process doesn't double-wrap it.
_FLEX_PATCHED = False


def is_available(name: str) -> bool:
    """Is this kernel importable AND usable on this box?"""
    from transformers.utils import (
        is_flash_attn_2_available,
        is_torch_flex_attn_available,
    )

    if name == "flash_attention_2":
        return bool(is_flash_attn_2_available())
    if name == "flex_attention":
        # FlexAttention's fused kernel is CUDA/Triton; on CPU it decomposes into
        # the dense matmul it exists to avoid, so it is not a fallback there.
        return bool(is_torch_flex_attn_available()) and torch.cuda.is_available()
    return True


def resolve_implementation(override: str = "", *, env_var: str = "HF_ATTN_IMPLEMENTATION") -> Optional[str]:
    """Choose the attention implementation to load a model under.

    `override` wins if set (including "eager", which is warned about rather than
    refused — it is the right choice for short sequences and the only one that
    supports `head_mask`). Otherwise take the first entry of `_ATTN_PREFERENCE`
    this box can run.

    Returns the string to hand `from_pretrained(attn_implementation=...)`, or
    None to leave transformers' own autoselect alone — which is only reached when
    nothing memory-linear is available, and is loudly flagged, because that run
    is the one that will OOM on a long example.

    `env_var` only names the knob in the messages, so each caller's log points at
    the variable its operator would actually set.
    """
    override = (override or "").strip()
    if override:
        if override == "eager":
            logger.warning(
                "%s=eager: attention will materialise a [batch, heads, seq, seq] "
                "score matrix (~8 GiB for one 8192-token example on the 120B, per "
                "layer). Expect OOM on long examples.", env_var,
            )
        elif not is_available(override):
            # Don't silently substitute: a pinned kernel that isn't installed
            # means the run would not be the one the operator asked to measure.
            raise RuntimeError(
                f"{env_var}={override!r} is not available in this environment. "
                "Install it, or unset the variable to autoselect from "
                f"{list(_ATTN_PREFERENCE)}."
            )
        logger.info("Attention implementation pinned to %r.", override)
        return override

    for name in _ATTN_PREFERENCE:
        if is_available(name):
            logger.info("Attention implementation: %r (autoselected).", name)
            return name

    logger.warning(
        "Neither %s is available; falling back to transformers' autoselect. For "
        "gpt-oss (_supports_sdpa = False) that means `eager`, whose score matrix "
        "is quadratic in sequence length — long examples will OOM. Install "
        "flash-attn or upgrade torch for FlexAttention.",
        " nor ".join(_ATTN_PREFERENCE),
    )
    return None


def apply_to_loaded(model, requested: Optional[str]) -> Optional[str]:
    """Check what a freshly loaded model ACTUALLY got, and patch flex if that's it.

    Reads back `config._attn_implementation` rather than trusting the request: an
    unsupported value is silently downgraded, and this is the line that tells a
    post-mortem which kernel the OOM happened under. Returns the effective name.
    """
    effective = getattr(getattr(model, "config", None), "_attn_implementation", None)
    if requested is not None and effective != requested:
        logger.warning("Requested attn_implementation=%r but the model loaded "
                       "with %r.", requested, effective)
    if effective == "flex_attention":
        keep_flex_attention_fused()
    elif effective == "eager":
        logger.warning(
            "Model loaded with EAGER attention — the quadratic score matrix is "
            "live. A long example will OOM; set the attention implementation "
            "explicitly or install a memory-linear kernel."
        )
    return effective


def keep_flex_attention_fused() -> None:
    """Make FlexAttention survive a sharded, variable-length training run.

    Two things break it, both silently, and both only once training is under way:

    1. Dynamo's recompile ceiling — see `_DYNAMO_CACHE_LIMIT`. Overflow reverts
       the kernel to its unfused decomposition, i.e. back to eager's memory.
    2. A single BlockMask shared across shards. The mask is built once per
       forward, on the inputs' device (masking_utils.flex_attention_mask passes
       `device=cache_position.device`), which under `device_map="auto"` is the
       FIRST shard only. Every layer accelerate placed on another card then calls
       the kernel with query/key on its own device and mask index tensors on
       card 0 — a cross-device access that aborts the compile for that layer.
       So the mask is re-homed per shard, cached ON the mask object so the copies
       die with it (one forward's mask is not reused by the next).

    Idempotent, and a no-op when FlexAttention isn't the selected kernel.
    """
    global _FLEX_PATCHED
    if _FLEX_PATCHED:
        return

    import torch._dynamo

    # `cache_size_limit` is per code object, `accumulated_cache_size_limit` the
    # budget across all of them; raising only the first still lets the second cap
    # the singleton once several shape buckets are live.
    for knob in ("cache_size_limit", "accumulated_cache_size_limit"):
        if getattr(torch._dynamo.config, knob, 0) < _DYNAMO_CACHE_LIMIT:
            setattr(torch._dynamo.config, knob, _DYNAMO_CACHE_LIMIT)
    logger.info("Dynamo recompile ceiling raised to %d (FlexAttention singleton "
                "is shared by every layer and shard).", _DYNAMO_CACHE_LIMIT)

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    inner = ALL_ATTENTION_FUNCTIONS["flex_attention"]

    def _shard_local_flex_attention(module, query, key, value, attention_mask,
                                    *args, **kwargs):
        return inner(module, query, key, value,
                     mask_on(attention_mask, query.device), *args, **kwargs)

    # `register` (not `interface[key] = …`) so the replacement lands in the class's
    # _global_mapping: item assignment only writes a _local_mapping on the one
    # instance, which the modeling files happen to share today but need not.
    ALL_ATTENTION_FUNCTIONS.register("flex_attention", _shard_local_flex_attention)
    _FLEX_PATCHED = True


def _rehome(value, device):
    """Copy `value` onto `device`, rebuilding any closure that captured a tensor.

    `BlockMask.to()` moves the block index tensors but NOT `mask_mod`, which is a
    plain Python closure. transformers builds that closure out of tensors bound to
    the mask's original device — `add_offsets_to_mask_function` captures
    `q_offset = cache_position[0]` (a 0-d tensor) and, whenever the batch is
    padded, `padding_mask_function` captures the 2-D padding mask. Left alone they
    make the traced `q_idx + q_offset` mix cuda:1 with cuda:0, which is the
    "found two different devices" abort that kills the compile on every shard but
    the first.

    Functions are rebuilt rather than wrapped because the capture has to be
    replaced in place: wrapping could only move the arguments or the result, and
    either way the addition still straddles two cards inside the traced region.
    """
    import types

    if torch.is_tensor(value):
        return value.to(device) if value.device != device else value
    if isinstance(value, types.FunctionType) and value.__closure__:
        cells = []
        for cell in value.__closure__:
            try:
                cells.append(types.CellType(_rehome(cell.cell_contents, device)))
            except ValueError:  # empty cell (recursive closure); keep it as-is
                cells.append(cell)
        rebuilt = types.FunctionType(value.__code__, value.__globals__,
                                     value.__name__, value.__defaults__,
                                     tuple(cells))
        rebuilt.__kwdefaults__ = value.__kwdefaults__
        return rebuilt
    if type(value) in (tuple, list):  # and_masks/or_masks hold their operands here
        return type(value)(_rehome(v, device) for v in value)
    return value


def _closure_tensors(value, depth: int = 0):
    """Yield the tensors a closure (or a tuple/list of them) has captured."""
    import types

    if depth > 6:
        return
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, types.FunctionType):
        for cell in (value.__closure__ or ()):
            try:
                contents = cell.cell_contents
            except ValueError:      # empty cell (recursive closure)
                continue
            yield from _closure_tensors(contents, depth + 1)
    elif type(value) in (tuple, list):
        for item in value:
            yield from _closure_tensors(item, depth + 1)


def mask_on(attention_mask, device):
    """Return `attention_mask` with everything on `device`, cached per device.

    Only BlockMasks need this — a plain tensor mask is indexed inside `score_mod`,
    which transformers already aligns.

    The subtle part is deciding whether a mask needs moving at all, because under
    `device_map="auto"` it arrives HALF moved. accelerate's AlignDevicesHook sends
    each module's arguments to that module's device, and BlockMask is a registered
    pytree, so its block index tensors are already on the right card by the time we
    see it — while `mask_mod`, a plain Python closure sitting in the pytree's
    context rather than among its leaves, still captures `q_offset` on the FIRST
    shard's device. Testing `kv_num_blocks.device` therefore reports "already
    home" for exactly the masks that are still broken, and the traced
    `q_idx + q_offset` then mixes cuda:1 with cuda:0 and aborts the compile —
    which is the failure this whole function exists to prevent, arriving through
    the one check that was supposed to skip it.

    So the closure is what we test. The returned mask is always a fresh BlockMask:
    the incoming one is shared with the other shards, and re-homing it in place
    would hand cuda:1's offsets to cuda:2. The copy is cached ON the mask so a
    shard pays for it once per forward rather than once per layer, and dies with
    it (one forward's mask is not reused by the next).
    """
    from torch.nn.attention.flex_attention import BlockMask

    if not isinstance(attention_mask, BlockMask):
        return attention_mask
    if attention_mask.kv_num_blocks.device == device and not any(
        t.device != device for t in _closure_tensors(attention_mask.mask_mod)
    ):
        return attention_mask
    cache = getattr(attention_mask, "_shard_masks", None)
    if cache is None:
        cache = {}
        # Survives on the mask for this forward only; BlockMask has no __slots__.
        attention_mask._shard_masks = cache
    if device not in cache:
        moved = attention_mask.to(device)       # a new object even when on-device
        moved.mask_mod = _rehome(attention_mask.mask_mod, device)
        cache[device] = moved
    return cache[device]
