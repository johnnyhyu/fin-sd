"""
probing_script.py — extracts aligned logit distributions under two conditions.

Condition A ("with hint"):
    prompt = [system] [user: problem] [assistant: truncated_reasoning + hint]
Condition B ("without hint"):
    prompt = [system] [user: problem] [assistant: truncated_reasoning]

Algorithm:
  1. Generate a greedy completion from Condition A (no gradient).
  2. Teacher-force those exact tokens through BOTH prefixes.
  3. Return the per-token logit distributions so that:
       logits_with_hint[i]    ≈ P( token_i | A-prefix, token_0..token_{i-1} )
       logits_without_hint[i] ≈ P( token_i | B-prefix, token_0..token_{i-1} )
     These are aligned: same token at each position, different conditioning context.

Both prefixes are rendered by prompts.continuation_prompt_ids, which leaves the
assistant turn OPEN mid-message on harmony's `analysis` channel, and the vLLM
calls send those exact ids to /v1/completions. Neither side may use a chat
request here: on gpt-oss the HF chat template renders a channel-less (invalid)
assistant turn, and vLLM's harmony renderer closes the message and opens a new
one — which would mean the completion is decoded from one context, the teacher
scored in a second and the student in a third. See prompts.py for the details.

The HuggingFace model is loaded once (singleton) and shared with optimization_script.py.
Note: after optimization_script updates weights, the vLLM server's weights become
stale. run_pipeline.py saves the LoRA adapter at the end of each epoch and
hot-loads it into vLLM (--enable-lora + VLLM_ALLOW_RUNTIME_LORA_UPDATING=True).
Since the bf16 training base is a lossless upcast of MXFP4, the adapter is
exactly consistent with the served base — do not merge-and-requantize.
"""
from __future__ import annotations

import contextlib
import threading
from typing import Optional

import torch

from . import attention, config, expert_lora, inference_script, prompts
from .utils import (
    logger,
    log_vram_budget,
    model_family,
    parse_vllm_token_id,
    post_with_retry,
    resolve_max_memory,
    same_model_family,
)

# ── Singletons shared with optimization_script ─────────────────────────────
_model: Optional[torch.nn.Module] = None
_tokenizer = None
# Guards the one-time model load: with concurrent item processing, several
# threads can reach get_model_and_tokenizer before the singleton is set, and an
# unguarded load would materialise the 120B model more than once.
_load_lock = threading.Lock()

# gpt-oss is MoE: bf16 router matmuls dispatch a slightly different token count per
# expert on the checkpoint recompute, which trips non-reentrant checkpointing's
# tensor-metadata consistency check whenever gradients flow through the experts
# (LORA_TARGET_EXPERTS, default-on). Reentrant skips that check (matching
# sft.train). transformers>=4.35 defaults gradient_checkpointing_enable() to
# use_reentrant=False, so we pass this explicitly at every enable site.
_GC_REENTRANT_KWARGS = {"gradient_checkpointing_kwargs": {"use_reentrant": True}}

# Gradient-hook handles from per-expert LoRA masking (Design B). Kept alive for the
# model's lifetime so the hooks keep firing; module-scoped because the masks belong
# to the _model singleton and must outlive _load_model_and_tokenizer's frame.
_expert_mask_handles: list = []


def is_loaded() -> bool:
    """True once the HF model singleton has been materialised."""
    return _model is not None


def _resolve_target_modules(model, candidates: list[str]) -> list[str]:
    """Keep only candidate names that exist as Linear submodules of `model`.

    gpt-oss stores its MoE expert weights as 3-D nn.Parameters
    (`experts.gate_up_proj` / `experts.down_proj`), not nn.Linear modules, so
    names like "gate_proj"/"up_proj"/"down_proj" match nothing there — those are
    adapted via target_parameters instead (see _resolve_target_parameters).
    PEFT only errors when *zero* names match, so without this check LoRA would
    quietly train fewer modules than the config claims.
    """
    present: set[str] = set()
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in candidates and isinstance(module, torch.nn.Linear):
            present.add(leaf)
    matched = [c for c in candidates if c in present]
    skipped = [c for c in candidates if c not in present]
    if not matched:
        raise RuntimeError(
            f"None of the candidate LoRA target modules {candidates} exist as "
            f"Linear layers in {type(model).__name__}. LoRA would train nothing."
        )
    if skipped:
        logger.info(
            "LoRA target modules not present as Linear layers in %s and skipped: "
            "%s (expected for gpt-oss, whose MoE experts are packed nn.Parameters "
            "and adapted via target_parameters). Linear modules adapted: %s.",
            type(model).__name__, skipped, matched,
        )
    return matched


def _resolve_target_parameters(model, candidates: list[str]) -> list[str]:
    """Return the candidate parameter-name suffixes that exist in `model`.

    gpt-oss stores its MoE experts as 3-D nn.Parameters (`mlp.experts.gate_up_proj`
    / `mlp.experts.down_proj`), which PEFT can only reach through
    LoraConfig.target_parameters (>= 0.17), not target_modules. Returns [] for a
    dense model where these names don't exist, so the caller can fall back to
    Linear-only coverage without erroring.
    """
    names = [name for name, _ in model.named_parameters()]
    present = [c for c in candidates if any(n.endswith(c) for n in names)]
    missing = [c for c in candidates if c not in present]
    if missing:
        logger.info(
            "Candidate expert LoRA parameters not found in %s and skipped: %s.",
            type(model).__name__, missing,
        )
    return present


def _num_experts(model) -> Optional[int]:
    """Best-effort read of the MoE expert count from the model config."""
    cfg = getattr(model, "config", None)
    for attr in ("num_local_experts", "num_experts"):
        n = getattr(cfg, attr, None)
        if isinstance(n, int) and n > 0:
            return n
    return None


def _iter_router_modules(model):
    """Yield (layer_idx, router_module) for each MoE router in `model`.

    gpt-oss names the per-layer gate `…layers.{i}.mlp.router`; some checkpoints
    call it `.gate`. We match on the leaf name and recover the layer index from
    the module path, so the caller can key activation stats by layer without
    assuming a fixed attribute chain.
    """
    import re
    seen = []
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in ("router", "gate"):
            continue
        m = re.search(r"layers\.(\d+)\.", name)
        if m is None:
            continue
        seen.append((int(m.group(1)), name, module))
    # A layer can expose both .mlp.router and an unrelated .gate; prefer 'router'.
    by_layer: dict[int, tuple] = {}
    for li, name, module in seen:
        if li not in by_layer or name.rsplit(".", 1)[-1] == "router":
            by_layer[li] = (name, module)
    for li in sorted(by_layer):
        yield li, by_layer[li][1]


def _full_router_logits(mod, inp, out, n_exp, gate_k, _torch):
    """Best-effort recovery of the dense per-token router logits over all experts.

    gpt-oss's router returns (router_scores, router_indices) where router_scores is
    softmaxed over ONLY the top-k selected experts (zeros elsewhere) — not a full
    distribution, so its entropy can't measure per-token routing sharpness over all
    experts. We recover the dense logits by:
      1. using the module output if it already looks dense (values off the top-k
         support — negatives or more than gate_k nonzeros per row), else
      2. reconstructing logits = hidden @ router_weight.T (+bias) from the hooked
         input and the router's own weight (gpt-oss GptOssTopKRouter.weight).
    Returns a [tokens, n_exp] float tensor, or None if neither path applies (caller
    then reports per-token sharpness as unavailable).
    """
    cand = out[0] if isinstance(out, (tuple, list)) and out else out
    if isinstance(cand, _torch.Tensor) and cand.shape[-1] == n_exp:
        flat = cand.reshape(-1, n_exp)
        nnz = (flat != 0).sum(-1)
        if bool((flat < 0).any()) or bool((nnz > gate_k).any()):
            return flat.float()          # already dense logits / full distribution
    w = getattr(mod, "weight", None)
    x = inp[0] if isinstance(inp, (tuple, list)) and inp else inp
    if (isinstance(w, _torch.Tensor) and w.dim() == 2 and w.shape[0] == n_exp
            and isinstance(x, _torch.Tensor)):
        b = getattr(mod, "bias", None)
        return _torch.nn.functional.linear(
            x.reshape(-1, x.shape[-1]).float(), w.float(),
            b.float() if isinstance(b, _torch.Tensor) else None,
        )
    return None


def profile_expert_activations(model, tokenizer, problems, *, max_items=256, top_k=None):
    """Forward `problems` through the frozen router and profile expert routing.

    Runs forward-only (no grad, no LoRA needed — call on the base model). For each
    MoE layer, hooks the router and accumulates:
      • hard selection counts per expert (which experts fire, and how often), and
      • when dense logits are recoverable (see _full_router_logits), the soft
        per-token distribution — used to separate a genuinely uniform layer from
        one that is specialized per token but flat in aggregate.

    Returns (counts, stats):
      counts — {layer_idx: list[int] length n_exp} hard selection tallies.
      stats  — {layer_idx: {"h_tok": float, "h_marg": float, "n_tok": int}} where
               h_tok is the mean per-token routing entropy (bits) and h_marg is the
               entropy of the soft marginal (mean per-token distribution). Their gap
               h_marg - h_tok ≈ mutual information I(token; expert): ~0 means the
               layer is uniform per token (safe to treat as generic); large means
               tokens route sharply but to different experts (specialized — do not
               drop). Empty {} when dense logits were not recoverable.
    """
    import torch as _torch

    device = next(model.parameters()).device
    n_exp = _num_experts(model) or 1
    gate_k = getattr(model.config, "num_experts_per_tok", None) or \
        getattr(model.config, "experts_per_token", None) or top_k or 4

    counts: dict[int, _torch.Tensor] = {}
    psum: dict[int, _torch.Tensor] = {}       # Σ_token softmax(logits) → soft marginal
    hsum: dict[int, _torch.Tensor] = {}       # Σ_token H(p_token), bits
    ntok: dict[int, int] = {}
    have_soft = {"any": False}

    def _make_hook(li):
        def _hook(_mod, _inp, out):
            sel = None
            if isinstance(out, (tuple, list)):
                for t in out:                 # gpt-oss → (scores, indices); pick the int tensor
                    if isinstance(t, _torch.Tensor) and t.dtype in (
                        _torch.long, _torch.int32, _torch.int64
                    ):
                        sel = t
                        break
            logits = _full_router_logits(_mod, _inp, out, n_exp, gate_k, _torch)
            if sel is None and logits is not None:
                sel = logits.topk(gate_k, dim=-1).indices
            if sel is not None:
                flat = sel.reshape(-1).to(counts[li].device)
                counts[li].scatter_add_(0, flat, _torch.ones_like(flat, dtype=counts[li].dtype))
            if logits is not None:
                p = _torch.softmax(logits, dim=-1)                 # [tokens, n_exp]
                ent = -(p * p.clamp_min(1e-12).log2()).sum(-1)     # [tokens] bits
                psum[li] += p.sum(0).to(psum[li].device)
                hsum[li] += ent.sum().to(hsum[li].device)
                ntok[li] += ent.numel()
                have_soft["any"] = True
        return _hook

    handles = []
    for li, router in _iter_router_modules(model):
        counts[li] = _torch.zeros(n_exp, dtype=_torch.long, device=device)
        psum[li] = _torch.zeros(n_exp, dtype=_torch.float64, device=device)
        hsum[li] = _torch.zeros((), dtype=_torch.float64, device=device)
        ntok[li] = 0
        handles.append(router.register_forward_hook(_make_hook(li)))

    if not counts:
        raise RuntimeError(
            "No MoE router modules found (looked for '.mlp.router'/'.gate' under "
            f"layers.* in {type(model).__name__}). This model may be dense, or the "
            "router attribute name differs — inspect model.named_modules()."
        )

    was_training = model.training
    model.eval()
    try:
        with _torch.no_grad():
            for problem in list(problems)[:max_items]:
                ids = _build_prefix_ids(problem, "", tokenizer).unsqueeze(0).to(device)
                model(ids)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    counts_out = {li: c.tolist() for li, c in counts.items()}
    stats_out: dict[int, dict] = {}
    if have_soft["any"]:
        for li in counts:
            n = ntok[li]
            if not n:
                continue
            marg = (psum[li] / n).clamp_min(1e-12)
            h_marg = float(-(marg * marg.log2()).sum().item())
            h_tok = float((hsum[li] / n).item())
            stats_out[li] = {"h_tok": h_tok, "h_marg": h_marg, "n_tok": int(n)}
    return counts_out, stats_out


def _selected_experts_from_profile(threshold, n_experts=None):
    """Per-layer top-utilization expert sets from the cached activation profile.

    Reads config.EXPERT_PROFILE_PATH (written by tools/profile_experts.py) and, for
    each MoE layer, keeps the busiest experts whose CUMULATIVE hard-selection mass
    first reaches `threshold` of that layer's total. This is a variable count per
    layer, not a fixed top-k: a near-uniform early layer needs ~20+ experts to cover
    the mass, a concentrated late layer only 6–8. Selection is on absolute
    utilization (raw selection counts), never finance lift.

    Returns {layer_idx: sorted list[int] of kept expert indices}, or None when the
    cache is absent/unusable (caller then adapts every expert). A layer with zero
    recorded routing is omitted (left unmasked → all experts adapt).
    """
    import json
    import os
    path = config.EXPERT_PROFILE_PATH
    if not os.path.exists(path):
        logger.warning(
            "PROFILE_EXPERT_ACTIVATIONS is on but no profile cache at %s; run "
            "tools/profile_experts.py first. Adapting every expert.",
            path,
        )
        return None
    with open(path) as fh:
        profile = json.load(fh)
    per_layer = profile.get("counts") or {}
    if not per_layer:
        logger.warning("Profile cache %s has no 'counts'; adapting every expert.", path)
        return None

    # A profile is keyed to the model it was built on, and nothing downstream can
    # tell that it wasn't. Applied to a different model it does not error, it
    # MIS-SELECTS: the 20B's cache on the 120B gave layers 0-23 the 20B's expert
    # indices (of which only 32 of 128 exist) and left layers 24-35 unprofiled, so
    # they adapted every expert — a run that looks configured and trains the wrong
    # experts. launch.py rebuilds a stale cache before starting, but nothing
    # protected a direct `python run_pipeline.py`, a resumed run, or the serve
    # tools. Both mismatches are cheap to see here, so refuse the cache instead.
    cached_model = profile.get("model")
    if cached_model and not same_model_family(cached_model, config.HF_MODEL_PATH):
        logger.warning(
            "Profile cache %s was built for %r (%s) but this run trains %r (%s); "
            "ignoring it and adapting every expert. Rebuild it with "
            "tools/profile_experts.py to restore the profiled selection.",
            path, cached_model, model_family(cached_model),
            config.HF_MODEL_PATH, model_family(config.HF_MODEL_PATH),
        )
        return None
    if n_experts:
        widths = {len(counts) for counts in per_layer.values()}
        if widths != {n_experts}:
            logger.warning(
                "Profile cache %s records %s expert(s) per layer but this model has "
                "%d; ignoring it and adapting every expert.",
                path, "/".join(str(w) for w in sorted(widths)), n_experts,
            )
            return None

    threshold = min(max(float(threshold), 0.0), 1.0)
    selected: dict[int, list[int]] = {}
    for li, counts in per_layer.items():
        total = sum(counts)
        if total <= 0:
            continue                                # dead layer: leave unmasked
        order = sorted(range(len(counts)), key=lambda e: counts[e], reverse=True)
        acc, kept = 0, []
        for e in order:
            kept.append(e)
            acc += counts[e]
            if acc >= threshold * total:            # cumulative mass covered
                break
        selected[int(li)] = sorted(kept)
    return selected or None


def _install_expert_masks(model, selected_by_layer, adapter_name):
    """Restrict expert-LoRA adaptation to `selected_by_layer` experts (Design B).

    PEFT's ParamWrapper attaches one rank-r adapter to the whole fused expert tensor,
    so every expert is adapted uniformly — there is no per-expert rank knob. To adapt
    only the top-utilization experts, we zero each non-selected expert's lora_A/lora_B
    blocks at init and register a gradient hook that keeps them at zero. lora_B is
    zero-initialised, so a frozen-zero block contributes no delta (delta_e = B_e @ A_e):
    those experts stay at the base weights while the selected experts train. Block
    layout (empirically verified): expert e occupies lora_A rows {row // r == e}
    (contiguous) and lora_B columns {col % num_experts == e} (strided). A layer
    exposes two ParamWrappers (gate_up_proj, down_proj); both are masked.

    Returns (handles, kept_counts): the grad-hook handles (kept alive by the caller so
    the hooks keep firing) and {layer_idx: n_kept_experts} for logging.
    """
    import re
    import torch as _torch
    from peft.tuners.lora.layer import ParamWrapper

    handles: list = []
    kept_counts: dict[int, int] = {}
    for name, module in model.named_modules():
        if not isinstance(module, ParamWrapper):
            continue
        m = re.search(r"layers\.(\d+)\.", name)
        if m is None:
            continue
        keep = selected_by_layer.get(int(m.group(1)))
        if keep is None:
            continue                                    # unprofiled/dead layer: adapt all
        ne = getattr(module, "num_experts", 1) or 1
        r = module.r.get(adapter_name)
        if ne <= 1 or not r:
            continue
        keep_mask = _torch.zeros(ne, dtype=_torch.bool)
        keep_mask[_torch.tensor(sorted(keep), dtype=_torch.long)] = True
        wA = module.lora_A[adapter_name].weight         # (r*ne, in_features)
        wB = module.lora_B[adapter_name].weight         # (out_features, r*ne)
        idx = _torch.arange(r * ne)
        maskA = keep_mask[idx // r].to(wA.dtype).view(-1, 1).to(wA.device)
        maskB = keep_mask[idx % ne].to(wB.dtype).view(1, -1).to(wB.device)
        with _torch.no_grad():
            wA.mul_(maskA)
            wB.mul_(maskB)
        handles.append(wA.register_hook(lambda g, msk=maskA: g * msk))
        handles.append(wB.register_hook(lambda g, msk=maskB: g * msk))
        kept_counts[int(m.group(1))] = int(keep_mask.sum())
    return handles, kept_counts


def _peft_targets_parameters(model) -> bool:
    """True if any active PEFT adapter targets packed nn.Parameters (experts)."""
    cfgs = getattr(model, "peft_config", None)
    if not cfgs:
        return False
    return any(getattr(c, "target_parameters", None) for c in cfgs.values())


def get_model_and_tokenizer():
    """Lazily load (or return cached) HuggingFace model + tokenizer."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    with _load_lock:
        # Re-check under the lock: another thread may have loaded while we waited.
        if _model is not None:
            return _model, _tokenizer
        return _load_model_and_tokenizer()


def _load_model_and_tokenizer():
    """Materialise the HF model + tokenizer singleton. Caller holds _load_lock."""
    global _model, _tokenizer
    # Import here to avoid hard dep at module level when only using OpenRouter scripts
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config.HF_DTYPE, torch.bfloat16)

    # Attention (and, on dense models, MLP) Linear modules.
    lora_candidates = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
    # gpt-oss MoE experts, packed as 3-D nn.Parameters (reached via target_parameters).
    expert_param_candidates = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    # Per-GPU memory ceilings for device_map="auto" sharding: each visible card's
    # live free VRAM minus the per-card reserve a training step needs, unless
    # HF_MAX_MEMORY pins them explicitly (see utils.resolve_max_memory). HF only
    # honors max_memory when it computes the placement itself, so this is a no-op
    # (and ignored) unless device_map is "auto"/"balanced"/etc.
    max_memory = resolve_max_memory(config.HF_MAX_MEMORY,
                                    model_path=config.HF_MODEL_PATH)

    # Pick the attention kernel explicitly. gpt-oss's autoselect lands on `eager`,
    # whose [batch, heads, seq, seq] score matrix costs ~30 GiB per layer at the
    # 8192-token sequences the length-penalty probe forwards on the 120B — see
    # pipeline/attention.py for the full diagnosis.
    attn = attention.resolve_implementation(config.HF_ATTN_IMPLEMENTATION)
    logger.info("Loading via HF+PEFT from %s (dtype=%s, device_map=%s, max_memory=%s, attn=%s) …",
                config.HF_MODEL_PATH, config.HF_DTYPE, config.HF_DEVICE, max_memory,
                attn or "auto")
    _tokenizer = AutoTokenizer.from_pretrained(config.HF_MODEL_PATH, use_fast=True)
    load_kwargs = dict(
        torch_dtype=dtype,
        device_map=config.HF_DEVICE,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    if attn:
        load_kwargs["attn_implementation"] = attn
    _model = AutoModelForCausalLM.from_pretrained(config.HF_MODEL_PATH, **load_kwargs)
    attention.apply_to_loaded(_model, attn)
    # Reentrant checkpointing is required for gpt-oss's MoE experts under grad
    # (see _GC_REENTRANT_KWARGS); non-reentrant is the transformers default.
    _model.gradient_checkpointing_enable(**_GC_REENTRANT_KWARGS)
    _model.config.use_cache = False
    _model.enable_input_require_grads()

    # Report device placement (helps debug multi-GPU / TP splits).
    _lm_head_devices = sorted(
        {str(p.device) for n, p in _model.named_parameters() if "lm_head" in n}
    )
    logger.info(
        "Device map: %s | lm_head on: %s",
        getattr(_model, "hf_device_map", "<single device>"),
        ", ".join(_lm_head_devices) or "<not found>",
    )

    lora_kwargs = dict(
        r=config.LORA_RANK,
        target_modules=_resolve_target_modules(_model, lora_candidates),
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── MoE expert coverage (the bulk of gpt-oss's parameters) ──────────────
    # Per-expert selection (Design B) is applied as grad masks AFTER get_peft_model,
    # since PEFT ranks the whole fused expert tensor uniformly; computed here so the
    # rank_pattern / logging stay together.
    selected_experts = None
    if config.LORA_TARGET_EXPERTS:
        supports_params = "target_parameters" in getattr(
            LoraConfig, "__dataclass_fields__", {}
        )
        if not supports_params:
            logger.warning(
                "LORA_TARGET_EXPERTS is set but the installed PEFT lacks "
                "target_parameters (needs >= 0.17). Training attention-only; "
                "the MoE experts will NOT be adapted. Upgrade peft to fix."
            )
        else:
            expert_params = _resolve_target_parameters(_model, expert_param_candidates)
            if expert_params:
                n_exp = _num_experts(_model) or 1
                expert_rank = config.LORA_EXPERT_RANK or max(1, config.LORA_RANK // n_exp)
                lora_kwargs["target_parameters"] = expert_params
                # A rank-r adapter is applied per expert; cap the per-expert rank
                # so the total expert budget stays ≈ one dense rank-r layer. Every
                # layer/param gets the same rank; per-expert restriction is a mask.
                lora_kwargs["rank_pattern"] = {p: expert_rank for p in expert_params}
                if config.PROFILE_EXPERT_ACTIVATIONS:
                    selected_experts = _selected_experts_from_profile(
                        config.PROFILE_EXPERT_MASS_THRESHOLD, n_experts=n_exp
                    )
                if selected_experts is not None:
                    counts = sorted(len(v) for v in selected_experts.values())
                    logger.info(
                        "Profiled expert LoRA (Design B): per-expert rank %d, keeping "
                        "the busiest experts up to %.0f%% cumulative mass per layer — "
                        "%d–%d experts/layer (mean %.1f) of %d, across %d layer(s); "
                        "the rest are frozen (attention rank=%d).",
                        expert_rank, 100 * config.PROFILE_EXPERT_MASS_THRESHOLD,
                        counts[0], counts[-1], sum(counts) / len(counts), n_exp,
                        len(counts), config.LORA_RANK,
                    )
                else:
                    logger.info(
                        "LoRA adapting %d MoE expert parameter group(s) at per-expert "
                        "rank %d (model reports %d experts; attention rank=%d).",
                        len(expert_params), expert_rank, n_exp, config.LORA_RANK,
                    )
            else:
                logger.warning(
                    "LORA_TARGET_EXPERTS is set but no packed expert parameters "
                    "were found in %s; adapting Linear modules only.",
                    type(_model).__name__,
                )

    _model = get_peft_model(_model, LoraConfig(**lora_kwargs))
    _model.train()

    adapter = getattr(_model, "active_adapter", "default")
    if isinstance(adapter, (list, tuple)):
        adapter = adapter[0] if adapter else "default"

    # Move the expert adapter off PEFT's fused-delta path and onto the
    # activation-space one, which also lets the profiled selection be an
    # ALLOCATION rather than a mask (see pipeline/expert_lora.py). Returns None —
    # having changed nothing — for a model or a PEFT version it doesn't recognise,
    # in which case the grad-mask path below still applies the same selection.
    activation_space = None
    if config.LORA_TARGET_EXPERTS and expert_lora.is_enabled():
        activation_space = expert_lora.install(_model, adapter, selected_experts)

    # Freeze the adapter for non-selected experts (Design B). Must run after
    # get_peft_model so the ParamWrappers (and their lora_A/lora_B) exist. Only
    # needed on the fused-delta fallback: on the activation-space path the frozen
    # experts have no parameters to mask in the first place.
    if selected_experts is not None and activation_space is None:
        global _expert_mask_handles
        _expert_mask_handles, kept_counts = _install_expert_masks(
            _model, selected_experts, adapter
        )
        if kept_counts:
            vals = sorted(kept_counts.values())
            logger.info(
                "Per-expert LoRA masks installed on %d layer(s): %d–%d experts "
                "adapted per layer (mean %.1f); remaining experts frozen.",
                len(kept_counts), vals[0], vals[-1], sum(vals) / len(vals),
            )
        else:
            logger.warning(
                "Per-expert selection produced no maskable ParamWrappers; every "
                "expert will adapt (check the profile cache and PEFT version)."
            )

    if hasattr(_model, "get_nb_trainable_parameters"):
        trainable, total = _model.get_nb_trainable_parameters()
        logger.info(
            "LoRA attached: %s trainable / %s total parameters (%.4f%%).",
            f"{trainable:,}", f"{total:,}", 100 * trainable / total,
        )

    if _tokenizer.pad_token_id is None:
        _tokenizer.pad_token_id = _tokenizer.eos_token_id

    # Decide NOW whether this layout can survive a training step, while the answer
    # is still a placement problem the operator can fix, rather than at the first
    # long sequence of the first epoch (see utils.log_vram_budget).
    log_vram_budget("loading the student + LoRA", config.HF_MIN_HEADROOM_GIB)

    logger.info("HF model loaded and ready.")
    return _model, _tokenizer


# ── Token sequence builders ─────────────────────────────────────────────────

def _build_prefix_ids(
    problem: str,
    assistant_prefix: str,
    tokenizer,
) -> torch.Tensor:
    """Build the token sequence both forwards are conditioned on, as a 1-D tensor.

    Ends MID-message inside an open `analysis` assistant turn, so teacher-forcing
    a completion onto it scores the tokens as a continuation of the truncated
    reasoning — the alignment this module's contract (see the header) assumes.

    Rendered by prompts.continuation_prompt_ids rather than the chat template
    directly. On gpt-oss the template cannot express this: its no-`thinking`
    branch emits a channel-less `<|start|>assistant<|message|>`, which is not
    valid harmony and which the model degenerates on. Going through prompts also
    means these are the exact ids the vLLM calls below send as their raw prompt,
    so teacher and student share one conditioning context by construction.
    """
    ids = prompts.continuation_prompt_ids(
        config.INFERENCE_SYSTEM_PROMPT, problem, assistant_prefix, tokenizer
    )
    return torch.tensor(ids, dtype=torch.long)


def _stop_ids(tokenizer) -> list[int]:
    """Stop-token ids that end a probe completion at its message boundary.

    On gpt-oss this adds `<|end|>` to the generation_config stop set, so a probe
    ends when the analysis message it is continuing closes — without it the
    completion runs on through `<|start|>assistant<|channel|>final…` and the
    distilled tokens stop being "the next step of the reasoning". Empty (falsy)
    for non-harmony models, where the caller's plain eos handling applies.
    """
    return prompts.harmony_stop_token_ids(tokenizer)


# ── Logit extraction helper ─────────────────────────────────────────────────

def _get_completion_logits(
    model,
    prefix_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    device: torch.device,
    no_grad: bool,
) -> torch.Tensor:
    """Return logit slice over the completion tokens.

    For a sequence [prefix_0 … prefix_{p-1}  c_0 … c_{n-1}],
    the logit predicting c_i sits at position (p-1+i) in the model output.
    The completion sits at the tail of the sequence, so we ask the model to
    project the lm_head over only the last n+1 positions (logits_to_keep=n+1,
    absolute indices p-1 … p+n-1) instead of all p+n. This skips the lm_head
    matmul over the ~201k-token vocab for every prefix position — a large saving
    on long prefixes — and keeps those positions out of the student pass's
    autograd graph. The first n of the kept rows predict c_0 … c_{n-1}.

    Args:
        no_grad: If True, run inside torch.no_grad() (teacher/target pass).
                 If False, gradients are tracked (student pass for optimisation).
    """
    full_ids = torch.cat([prefix_ids, completion_ids]).unsqueeze(0).to(device)
    n = completion_ids.shape[0]

    if no_grad:
        ctx = torch.no_grad()
    else:
        ctx = torch.enable_grad()

    with ctx:
        # Keep the last n+1 positions: absolute indices p-1 … p+n-1.
        out = model(full_ids, logits_to_keep=n + 1)
        logits = out.logits  # (1, n+1, vocab)

    # Kept row j is absolute position (p-1+j); rows 0..n-1 predict c_0..c_{n-1}.
    sliced = logits[0, :n, :]  # (n, vocab)
    if no_grad:
        # The slice is a view that pins the (1, n+1, vocab) output's extra row;
        # clone so the teacher pass retains only the (n, vocab) block.
        sliced = sliced.clone()
    return sliced


def _batched_completion_logits(
    model,
    full_seqs: list[torch.Tensor],
    completion_lens: list[int],
    device: torch.device,
    no_grad: bool,
    pad_id: int,
) -> list[torch.Tensor]:
    """Run ONE batched forward over B sequences; return per-item completion logits.

    This is the multi-item analogue of _get_completion_logits: instead of B
    sequential batch-size-1 forwards, it left-pads the B sequences into a single
    (B, L) batch and projects the lm_head once. Left padding flushes every
    completion against the tail, so:
      • the logits_to_keep tail-window optimization still applies to the whole
        batch (keep the last max_n+1 positions), and
      • each item's completion logits are a contiguous block ending at the tail.

    Because the sequences are left-padded, the default arange position ids (valid
    only at batch size 1) would be wrong, so explicit position ids that reset
    after the pad are passed alongside the attention mask — required for rotary
    models like gpt-oss.

    Args:
        full_seqs:       B 1-D LongTensors, each [prefix_i ++ completion_i].
        completion_lens: B ints n_i — the tail length of each seq that is the
                         completion (rows we return logits for).
        no_grad:         If True, run inside torch.no_grad() (teacher pass) and
                         clone the slices so the (B, max_n+1, V) buffer is freed.
        pad_id:          Token id used to fill the left padding (masked out).

    Returns:
        List of B tensors; item j is (n_j, vocab), aligned so row i predicts the
        j-th item's completion token c_i.
    """
    lengths = [int(s.shape[0]) for s in full_seqs]
    L = max(lengths)
    max_n = max(completion_lens)
    B = len(full_seqs)

    input_ids = torch.full((B, L), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, L), dtype=torch.long, device=device)
    for j, seq in enumerate(full_seqs):
        input_ids[j, L - lengths[j]:] = seq.to(device)
        attention_mask[j, L - lengths[j]:] = 1

    # Left-padding-correct position ids: 0,1,2,… over the real tokens, with pad
    # positions pinned to 0 (they are masked out of attention anyway).
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.clamp_(min=0)

    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        # Keep the last max_n+1 positions for the whole batch: since completions
        # are tail-aligned, this window covers every item's completion.
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            logits_to_keep=max_n + 1,
        )
        out = outputs.logits  # (B, max_n+1, vocab)

    results: list[torch.Tensor] = []
    for j, n in enumerate(completion_lens):
        # Completion occupies absolute indices L-n .. L-1; the predicting logits
        # sit one position earlier, mapping to kept-window rows max_n-n .. max_n-1.
        sliced = out[j, max_n - n: max_n, :]  # (n, vocab)
        if no_grad:
            sliced = sliced.clone()
        results.append(sliced)
    return results


# ── Generation-mode context manager ─────────────────────────────────────────

class _inference_mode:
    """Temporarily put the model in inference configuration for generation.

    With gradient checkpointing enabled in train mode, transformers disables
    the KV cache, making generate() re-run the full forward per decoded token
    (quadratic — hours per item on a 120B model). Switch to eval +
    checkpointing-off for the generation, then restore training state.

    When LoRA targets the MoE experts, PEFT materializes the adapter delta for
    *every* expert at each decode step under the KV cache, a large per-token
    slowdown. If LORA_MERGE_FOR_GEN is set we merge the adapter into the base
    for the generation window and unmerge afterwards (the PEFT-recommended fix),
    leaving the un-merged adapter active for the gradient-tracked probing passes.
    """

    def __init__(self, model, skip_merge: bool = False):
        self.model = model
        # skip_merge: the caller has disabled the adapter (frozen-teacher decode),
        # so there is no per-token expert delta to avoid — and merging would bake
        # the delta into the base, defeating the freeze. Skip the merge entirely.
        self.skip_merge = skip_merge
        self._merged = False

    def __enter__(self):
        self.model.eval()
        self.model.gradient_checkpointing_disable()
        self.model.config.use_cache = True
        if (not self.skip_merge
                and config.LORA_MERGE_FOR_GEN
                and _peft_targets_parameters(self.model)
                and hasattr(self.model, "merge_adapter")):
            try:
                self.model.merge_adapter()
                self._merged = True
            except Exception as exc:  # fall back to un-merged (slower) generation
                logger.warning(
                    "Could not merge adapter for generation (%s); continuing "
                    "un-merged (slower per-token decode).", exc,
                )
        return self.model

    def __exit__(self, *_):
        if self._merged:
            try:
                self.model.unmerge_adapter()
            except Exception as unmerge_exc:
                logger.error(
                    "Failed to unmerge adapter after generation (%s); base "
                    "weights may be polluted for subsequent steps.", unmerge_exc,
                )
            self._merged = False
        try:
            self.model.gradient_checkpointing_enable(**_GC_REENTRANT_KWARGS)
            self.model.config.use_cache = False
            self.model.train()
        except Exception as cleanup_exc:
            logger.error("Failed to restore training state after generation (%s).", cleanup_exc)
        return False


# ── Frozen-teacher helper ─────────────────────────────────────────────────────

def _teacher_adapter_disabled(model):
    """Context manager that disables the LoRA adapter for the teacher pass.

    With config.FROZEN_TEACHER the teacher (with-hint) distribution must come from
    the original base weights, not the live student weights. Since training is
    LoRA, the base is already frozen — disabling the adapter inside this context
    yields the frozen teacher for free (no second model, no extra memory). A
    no-op (nullcontext) when FROZEN_TEACHER is off or the model is not PEFT-wrapped.

    Wrap BOTH the greedy with-hint decode (it defines the target trajectory) and
    the no-grad teacher forward in this; leave the gradient-tracked student
    forward OUTSIDE it so the student keeps using the live adapter.
    """
    if config.FROZEN_TEACHER and hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    return contextlib.nullcontext()


def _base_adapter_disabled(model):
    """Context manager that unconditionally disables the LoRA adapter.

    Unlike _teacher_adapter_disabled (gated on config.FROZEN_TEACHER), this always
    disables the adapter when the model is PEFT-wrapped, yielding the frozen base
    distribution. Used for the EXOPD reference π_ref, which must be the base model
    regardless of the teacher mode. A no-op when the model is not PEFT-wrapped.
    """
    if hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    return contextlib.nullcontext()


# ── Public API ──────────────────────────────────────────────────────────────

def _build_assistant_prefixes(truncated_reasoning: str, hint: str) -> tuple[str, str]:
    """Build the (with-hint, without-hint) assistant continuation prefixes.

    Shared by both the full-vocab HF path and the vLLM top-k path so the two
    conditions are constructed identically regardless of who runs the forward.
    """
    # Match the rollout format: the original assistant message opened with a
    # <reasoning> block (per INFERENCE_SYSTEM_PROMPT), so the probe prefixes
    # must too — otherwise the continuation is off-distribution.

    # Normalise: strip any <reasoning> wrapper that may have been included in
    # truncated_reasoning (e.g. when inference used a native reasoning_content
    # channel that itself echoed the XML tags). Double-wrapping would produce
    # <reasoning>\n<reasoning>\n… which is off-distribution and confuses the
    # model. Also strip trailing whitespace to avoid spurious blank lines.
    _tr = truncated_reasoning.strip()
    if _tr.startswith("<reasoning>"):
        _tr = _tr[len("<reasoning>"):].lstrip("\n")
    if "</reasoning>" in _tr:
        _tr = _tr[: _tr.index("</reasoning>")].rstrip()
    truncated_reasoning = _tr.rstrip()

    if not truncated_reasoning:
        raise RuntimeError(
            "truncated_reasoning is empty after stripping. State-finder returned "
            "a sentence that reduced the prefix to nothing; cannot build a meaningful "
            "probing prompt. Check state_finder_script output for this item."
        )

    # The hint is injected mid-reasoning as a continuation prefix: the model
    # must apply it and keep generating in-distribution. Frame it explicitly so
    # the continuation is a clean corrected next step rather than an apology or
    # restatement (which would pollute the teacher distribution and waste the
    # completion budget):
    #   - the trace so far is fully correct (don't let the model re-audit it);
    #   - the hint flags an error originally made at THIS step, not yet visible
    #     in the trace above (so the fix belongs at the current point, not earlier);
    #   - it must not spend tokens acknowledging the hint or naming the error.
    with_hint_prefix = (
        f"<reasoning>\n{truncated_reasoning}\n\n"
        "[The reasoning above is fully correct. The following hint refers to an "
        "error originally made at this current step that is not yet visible in the "
        "trace above. Do not apologize, acknowledge this hint, or explain the error. "
        "Silently apply the correction and continue directly with the next step of "
        "the calculation.]\n"
        f"[Hint]: {hint}"
    )
    # Mirror the with-hint framing minus the hint itself, so the two conditions
    # differ only by the correction content (not by the presence of an
    # instruction block): same "trace is correct, continue silently" steer.
    without_hint_prefix = (
        f"<reasoning>\n{truncated_reasoning}\n\n"
        "[The reasoning above is fully correct. Continue directly with "
        "the next step of the calculation.]"
    )
    return with_hint_prefix, without_hint_prefix


def run_probing(
    problem: str,
    truncated_reasoning: str,
    hint: str,
) -> dict:
    """Extract aligned logit distributions under the two prompting conditions.

    Args:
        problem:             Original problem text.
        truncated_reasoning: Reasoning chain truncated at the mistake point.
        hint:                Constructive hint generated by hint_script.

    Returns:
        {
            "logits_with_hint":    Tensor (N, vocab) — teacher distribution, no grad,
            "logits_without_hint": Tensor (N, vocab) — student distribution, has grad,
            "completion_tokens":   list[int]          — the N generated token ids,
        }
    """
    model, tokenizer = get_model_and_tokenizer()
    device = next(model.parameters()).device

    with_hint_prefix, without_hint_prefix = _build_assistant_prefixes(
        truncated_reasoning, hint
    )

    prefix_with_ids = _build_prefix_ids(problem, with_hint_prefix, tokenizer).to(device)
    prefix_without_ids = _build_prefix_ids(problem, without_hint_prefix, tokenizer).to(device)

    # ── Steps 1-2: decode + teacher forward, optionally on frozen base weights ─
    # With FROZEN_TEACHER the adapter is disabled for BOTH the with-hint decode
    # (so the target trajectory comes from the base model) and the teacher forward.
    with _teacher_adapter_disabled(model):
        # ── Step 1: generate completion greedily from "with hint" context ────
        logger.debug("Generating completion from 'with hint' prefix (%d tokens) …", prefix_with_ids.shape[0])
        with _inference_mode(model, skip_merge=config.FROZEN_TEACHER), torch.no_grad():
            generated = model.generate(
                prefix_with_ids.unsqueeze(0),
                attention_mask=torch.ones(1, prefix_with_ids.shape[0], dtype=torch.long, device=device),
                max_new_tokens=config.MAX_PROBE_TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=_stop_ids(tokenizer) or tokenizer.eos_token_id,
            )
        completion_ids = generated[0, prefix_with_ids.shape[0]:]

        if completion_ids.numel() == 0:
            raise RuntimeError("Model generated zero completion tokens from the 'with hint' prefix.")

        logger.debug("Completion: %d tokens generated.", completion_ids.shape[0])

        # ── Step 2: forward pass WITH hint (no grad — teacher/target) ───────
        logits_with = _get_completion_logits(
            model, prefix_with_ids, completion_ids, device, no_grad=True
        )

    # ── Step 3: forward pass WITHOUT hint (grad enabled — student) ──────────
    logits_without = _get_completion_logits(
        model, prefix_without_ids, completion_ids, device, no_grad=False
    )

    assert logits_with.shape == logits_without.shape, (
        f"Logit shape mismatch: {logits_with.shape} vs {logits_without.shape}"
    )

    logger.debug(
        "Probing complete. Logit shape: %s (N tokens × vocab).",
        logits_with.shape,
    )

    return {
        "logits_with_hint": logits_with,        # (N, V) — no grad
        "logits_without_hint": logits_without,  # (N, V) — has grad
        "completion_tokens": completion_ids.tolist(),
    }


def _vllm_student_rollouts(
    prefix_ids: torch.Tensor,
    n: int,
    stop_ids: list[int],
    seed: Optional[int] = None,
) -> list[list[int]]:
    """Sample n student completions continuing the without-hint prefix via vLLM.

    The on-policy rollouts are drawn from the *served* student weights (the active
    hot-swapped / merged checkpoint) with temperature/top-p sampling, in a single
    batched n-way request, rather than decoded one-at-a-time on the live HF model.
    Each chosen token is recovered as an HF-aligned vocab id via
    return_tokens_as_token_ids (same mechanism as the top-k teacher path), so the
    ids can be teacher-forced on the HF model in student_rollout_logits().

    Driven through /v1/completions with `prefix_ids` sent verbatim as the prompt,
    NOT through /v1/chat/completions with an assistant message: on gpt-oss vLLM
    renders chat requests with openai_harmony, which closes any assistant message
    it is given and opens a fresh turn after it, so `continue_final_message` never
    reaches a renderer that could honour it and the rollout would be sampled from
    a different context than the one it is later scored under. Passing raw ids
    also removes the last renderer disagreement: the tokens conditioning the
    server are literally the tensor the HF forwards use (prompts.py).

    NOTE: vLLM serves the weights from the last epoch boundary (merge+restart) or
    hot-swap, so mid-epoch these rollouts come from a *stale* student relative to
    the live HF model being trained — the same staleness the top-k teacher path
    already accepts (see config.PROBE_TOPK_VIA_VLLM). The objective stays
    on-policy at each epoch boundary, when the served weights are refreshed.
    """
    url = f"{config.VLLM_BASE_URL}/completions"
    headers = {
        "Authorization": f"Bearer {config.VLLM_API_KEY}",
        "Content-Type": "application/json",
    }
    # Rollouts come from the student, i.e. the live/served model (NOT the frozen
    # teacher), so always target the active served model.
    student_model = inference_script.get_active_model() or config.VLLM_MODEL
    payload = {
        "model": student_model,
        "prompt": [int(t) for t in prefix_ids.tolist()],
        "temperature": config.STUDENT_ROLLOUT_TEMPERATURE,
        "top_p": config.STUDENT_ROLLOUT_TOP_P,
        "max_tokens": config.MAX_PROBE_TOKENS,
        "n": n,
        # We only need the chosen token ids (no alternatives), recovered as
        # "token_id:N" strings. On /v1/completions `logprobs` is the NUMBER of
        # alternatives to report, not a bool — 0 means "sampled token only".
        "logprobs": 0,
        "return_tokens_as_token_ids": True,
    }
    if stop_ids:
        payload["stop_token_ids"] = stop_ids
    # A per-item seed (config.SEED + epoch/idx offset, derived by the caller) makes
    # runs reproducible while keeping rollouts diverse across items — a single
    # fixed seed reused everywhere would collapse the diversity the reverse-KL
    # objective needs. Within one request vLLM still draws n distinct sequences.
    if seed is not None:
        payload["seed"] = seed
    resp = post_with_retry(url, headers=headers, payload=payload, timeout=300,
                           label="vLLM-rollout")
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"vLLM-rollout returned an error body: {str(data['error'])[:500]}")
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"vLLM-rollout response had no choices: {str(data)[:500]}")

    completions: list[list[int]] = []
    n_length_truncated = 0
    for choice in choices:
        tokens = ((choice.get("logprobs") or {}).get("tokens")) or []
        ids = [_parse_token_id(tok) for tok in tokens]
        if choice.get("finish_reason") == "length":
            n_length_truncated += 1
        if ids:
            completions.append(ids)
    if n_length_truncated:
        logger.warning(
            "vLLM-rollout: %d/%d rollouts hit the %d-token limit; those completions "
            "may be cut off.", n_length_truncated, len(choices), config.MAX_PROBE_TOKENS,
        )
    return completions


def sample_student_rollouts(
    problem: str,
    truncated_reasoning: str,
    hint: str,
    seed: Optional[int] = None,
) -> dict:
    """Sample student rollouts (the cheap, no-grad half of the on-policy path).

    The on-policy counterpart to run_probing (config.STUDENT_ROLLOUT). Instead of
    greedily decoding ONE completion from the with-hint (teacher) prefix, the
    student samples config.NUM_STUDENT_ROLLOUTS completions from its OWN
    (without-hint) prefix with temperature/top-p sampling on vLLM (the served
    student weights), in a single batched request (see _vllm_student_rollouts).

    The teacher (with-hint) target logits for ALL rollouts are computed here in
    ONE batched no-grad forward (see _batched_completion_logits) rather than one
    per rollout: they carry no autograd graph, so batching only costs a transient
    (B, max_n+1, V) logit buffer — a near-free win over N separate batch-1
    forwards. The grad-carrying student forwards stay deferred to
    student_rollout_logits(), called once per rollout, so that only ONE student
    autograd graph is resident on the GPU at a time. (The old eager version
    teacher-forced every rollout up front AND held NUM_STUDENT_ROLLOUTS full-model
    backward graphs simultaneously — ~N× the peak memory, the dominant
    student-rollout OOM source.)

    Args:
        problem:             Original problem text.
        truncated_reasoning: Reasoning chain truncated at the mistake point.
        hint:                Constructive hint generated by hint_script.
        seed:                Per-item vLLM sampling seed for reproducibility, or
                             None to leave sampling unseeded (config.SEED < 0).

    Returns:
        A dict carrying everything student_rollout_logits() needs:
        {
            "completions":        list[Tensor] — non-empty completion id tensors,
            "teacher_logits":     list[Tensor] — per-rollout (N, V) teacher logits,
                                                 no grad, aligned with completions,
            "reference_logits":   list[Tensor] | None — per-rollout (N, V) base
                                                 (adapter-disabled) without-hint
                                                 logits for EXOPD (config.EXOPD),
                                                 no grad; None when EXOPD is off,
            "prefix_with_ids":    Tensor       — the with-hint (teacher) prefix,
            "prefix_without_ids": Tensor       — the without-hint (student) prefix,
            "device":             torch.device,
        }

    Raises:
        RuntimeError: if every sampled rollout is empty (zero completion tokens).
    """
    model, tokenizer = get_model_and_tokenizer()
    device = next(model.parameters()).device

    with_hint_prefix, without_hint_prefix = _build_assistant_prefixes(
        truncated_reasoning, hint
    )

    prefix_with_ids = _build_prefix_ids(problem, with_hint_prefix, tokenizer).to(device)
    prefix_without_ids = _build_prefix_ids(problem, without_hint_prefix, tokenizer).to(device)

    # ── Sample N completions from the "without hint" (student) prefix on vLLM ──
    n_rollouts = config.NUM_STUDENT_ROLLOUTS
    logger.debug(
        "Sampling %d student rollouts via vLLM from 'without hint' prefix, "
        "temp=%.2f, top_p=%.2f …",
        n_rollouts, config.STUDENT_ROLLOUT_TEMPERATURE, config.STUDENT_ROLLOUT_TOP_P,
    )

    completion_id_lists = _vllm_student_rollouts(
        prefix_without_ids, n_rollouts, _stop_ids(tokenizer), seed=seed
    )
    completions: list[torch.Tensor] = [
        torch.tensor(ids, dtype=torch.long, device=device)
        for ids in completion_id_lists if ids
    ]

    if not completions:
        raise RuntimeError(
            "All student rollouts were empty (zero completion tokens). The "
            "without-hint prefix may be terminating immediately."
        )

    logger.debug(
        "Student rollouts: %d/%d non-empty; completion lengths %s.",
        len(completions), n_rollouts, [c.shape[0] for c in completions],
    )

    # ── Teacher (with-hint) forwards for ALL rollouts in ONE batched pass ──────
    # No-grad target distributions, so batching holds no autograd graphs — only a
    # transient (B, max_n+1, V) buffer that _batched_completion_logits clones and
    # frees. FROZEN_TEACHER disables the adapter so the teacher is the base model,
    # matching the per-rollout single-forward path it replaces.
    with _teacher_adapter_disabled(model):
        teacher_logits = _batched_completion_logits(
            model,
            [torch.cat([prefix_with_ids, c]) for c in completions],
            [int(c.shape[0]) for c in completions],
            device,
            no_grad=True,
            pad_id=tokenizer.pad_token_id,
        )

    # ── Reference (π_ref) forwards for EXOPD, adapter disabled, without-hint ────
    # The base model's without-hint distribution over the same rollout tokens, in
    # one batched no-grad pass (same transient-buffer cost as the teacher pass).
    # Always disables the adapter (the reference is the base regardless of the
    # teacher mode — see config.EXOPD / loss_script.run_exopd_loss).
    reference_logits = None
    if config.EXOPD:
        with _base_adapter_disabled(model):
            reference_logits = _batched_completion_logits(
                model,
                [torch.cat([prefix_without_ids, c]) for c in completions],
                [int(c.shape[0]) for c in completions],
                device,
                no_grad=True,
                pad_id=tokenizer.pad_token_id,
            )

    return {
        "completions": completions,
        "teacher_logits": teacher_logits,
        "reference_logits": reference_logits,
        "prefix_with_ids": prefix_with_ids,
        "prefix_without_ids": prefix_without_ids,
        "device": device,
    }


def student_rollout_logits(
    sampled: dict, completion_ids: torch.Tensor, teacher_logits: torch.Tensor
) -> dict:
    """Run ONE student rollout's grad forward; pair it with its teacher logits.

    The grad-carrying half of the on-policy path. Called once per completion from
    sample_student_rollouts()["completions"], immediately before the caller
    computes the reverse KL and backpropagates. The teacher (with-hint) target
    logits were already computed for every rollout in one batched no-grad forward
    by sample_student_rollouts (passed in here as teacher_logits), so this call
    adds only the single grad-carrying student forward — peak GPU memory is still
    one full-model autograd graph at a time.

    Args:
        sampled:        The dict returned by sample_student_rollouts().
        completion_ids: One entry from sampled["completions"].
        teacher_logits: The matching entry from sampled["teacher_logits"] — the
                        no-grad teacher distribution for this rollout.

    Returns:
        {
            "logits_with_hint":    Tensor (N, vocab) — teacher distribution, no grad,
            "logits_without_hint": Tensor (N, vocab) — student distribution, has grad,
            "completion_tokens":   list[int]          — the N sampled token ids,
        }
    """
    model, _ = get_model_and_tokenizer()
    device = sampled["device"]

    # ── forward WITHOUT hint (grad enabled — student) ─────────────────────────
    # The teacher forward was already done (batched, no-grad) in
    # sample_student_rollouts; only the live student forward remains here.
    logits_without = _get_completion_logits(
        model, sampled["prefix_without_ids"], completion_ids, device, no_grad=False
    )
    assert teacher_logits.shape == logits_without.shape, (
        f"Logit shape mismatch: {teacher_logits.shape} vs {logits_without.shape}"
    )
    return {
        "logits_with_hint": teacher_logits,     # (N, V) — no grad
        "logits_without_hint": logits_without,  # (N, V) — has grad
        "completion_tokens": completion_ids.tolist(),
    }


# ── vLLM top-k probing path ──────────────────────────────────────────────────

def _parse_token_id(token: str) -> int:
    """Parse a vLLM "token_id:NNN" logprob token string into its integer id.

    Thin alias for utils.parse_vllm_token_id, kept because the top-k teacher path
    below reads ids out of both `tokens` and the `top_logprobs` keys.
    """
    return parse_vllm_token_id(token)


def _vllm_teacher_topk(
    prefix_ids: torch.Tensor, k: int, stop_ids: list[int]
) -> tuple[list[int], list[list[int]], list[list[float]]]:
    """Greedily continue the with-hint prefix on vLLM, returning teacher top-k.

    Returns (completion_ids, topk_ids, topk_logprobs) where:
        completion_ids[i]   — the token id greedily chosen at position i,
        topk_ids[i]         — the teacher's top-k token ids at position i,
        topk_logprobs[i]    — their (full-vocab) log-probabilities.
    The distribution at position i is P(· | with-hint prefix, tokens 0..i-1),
    i.e. the teacher distribution aligned to the same completion the HF student
    pass is teacher-forced on.

    That alignment is what makes the KL meaningful, and it is why this goes
    through /v1/completions with `prefix_ids` as a raw prompt — see
    _vllm_student_rollouts for the full reason a chat request cannot express a
    mid-message continuation on gpt-oss. Sending the same ids the student forward
    uses means the two distributions differ only by the hint, which is the one
    difference the objective is supposed to measure.
    """
    url = f"{config.VLLM_BASE_URL}/completions"
    headers = {
        "Authorization": f"Bearer {config.VLLM_API_KEY}",
        "Content-Type": "application/json",
    }
    # With FROZEN_TEACHER, target the un-adapted base model (config.VLLM_MODEL)
    # rather than the hot-loaded adapter, so the teacher top-k comes from the
    # frozen base. (Only truly frozen in the hot-swap LoRA serving path; see the
    # config.FROZEN_TEACHER note for the merge+restart expert-LoRA caveat.)
    teacher_model = (
        config.VLLM_MODEL if config.FROZEN_TEACHER
        else (inference_script.get_active_model() or config.VLLM_MODEL)
    )
    payload = {
        "model": teacher_model,
        "prompt": [int(t) for t in prefix_ids.tolist()],
        "temperature": 0.0,
        "max_tokens": config.MAX_PROBE_TOKENS,
        # On /v1/completions `logprobs` is the NUMBER of alternatives per position
        # (the chat endpoint's logprobs+top_logprobs pair does not exist here).
        "logprobs": k,
        "return_tokens_as_token_ids": True,
    }
    if stop_ids:
        payload["stop_token_ids"] = stop_ids
    resp = post_with_retry(url, headers=headers, payload=payload, timeout=300, label="vLLM-probe")
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"vLLM-probe returned an error body: {str(data['error'])[:500]}")
    try:
        choice = data["choices"][0]
        lp = choice["logprobs"]
        tokens = lp["tokens"]
        alts_per_pos = lp.get("top_logprobs") or [None] * len(tokens)
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"vLLM-probe response missing logprobs.tokens: {str(data)[:500]}"
        ) from exc

    if choice.get("finish_reason") == "length":
        logger.warning(
            "vLLM-probe completion hit the %d-token limit; teacher path may be cut off.",
            config.MAX_PROBE_TOKENS,
        )

    completion_ids: list[int] = []
    topk_ids: list[list[int]] = []
    topk_logprobs: list[list[float]] = []
    for token, alts in zip(tokens, alts_per_pos):
        completion_ids.append(_parse_token_id(token))
        # /v1/completions reports the alternatives as a {token: logprob} mapping,
        # and per the OpenAI contract it always includes the SAMPLED token — so a
        # position can carry k+1 entries when the sample fell outside the top-k
        # (possible only if a future caller drops temperature=0). Rank by logprob
        # and keep k, so the row is the teacher's true top-k either way.
        ranked = sorted((alts or {}).items(), key=lambda kv: kv[1], reverse=True)[:k]
        topk_ids.append([_parse_token_id(tok) for tok, _ in ranked])
        topk_logprobs.append([float(v) for _, v in ranked])
    return completion_ids, topk_ids, topk_logprobs


def prepare_probing_topk(
    problem: str,
    truncated_reasoning: str,
    hint: str,
) -> dict:
    """Do everything for the top-k probe EXCEPT the HF student forward.

    Splitting the per-item teacher work (the vLLM decode + top-k logprobs) from
    the student forward lets the caller buffer several prepared items and run
    their student forwards together as one real batch (see
    student_forward_topk_batch). The teacher reflects vLLM's served weights,
    which only refresh at epoch end — mid-epoch they are stale relative to the
    live HF student (see config.PROBE_TOPK_VIA_VLLM).

    Returns:
        {
            "teacher_topk_ids":      LongTensor (N, k)  — teacher top-k ids, no grad,
            "teacher_topk_logprobs": FloatTensor (N, k) — teacher log-probs, no grad,
            "teacher_topk_mask":     FloatTensor (N, k) — 1 for valid entries, 0 for padding,
            "completion_tokens":     list[int]          — the N decoded token ids,
            "_full_ids":             LongTensor (P+N,)  — [without-hint prefix ++ completion],
            "_n_completion":         int                — N, the completion length,
        }
        The leading-underscore keys feed student_forward_topk_batch; that call
        adds "student_logits" (N, V, grad) before the loss is computed.
    """
    model, tokenizer = get_model_and_tokenizer()
    device = next(model.parameters()).device

    with_hint_prefix, without_hint_prefix = _build_assistant_prefixes(
        truncated_reasoning, hint
    )
    # Both prefixes are built here, up front, so the ids sent to vLLM are the same
    # objects the student forward is conditioned on — the teacher and student
    # contexts can no longer drift apart through two different renderers.
    prefix_with_ids = _build_prefix_ids(problem, with_hint_prefix, tokenizer).to(device)
    prefix_without_ids = _build_prefix_ids(problem, without_hint_prefix, tokenizer).to(device)

    # ── Step 1+2: vLLM decodes the completion AND returns teacher top-k ──────
    k = config.PROBE_TOPK_K
    completion_ids_list, topk_ids_list, topk_lp_list = _vllm_teacher_topk(
        prefix_with_ids, k, _stop_ids(tokenizer)
    )
    if not completion_ids_list:
        raise RuntimeError("vLLM-probe generated zero completion tokens from the 'with hint' prefix.")
    logger.debug("vLLM-probe: %d completion tokens, top-%d teacher logprobs.",
                 len(completion_ids_list), k)

    completion_ids = torch.tensor(completion_ids_list, dtype=torch.long, device=device)

    # Pad the (possibly ragged) top-k rows into rectangular tensors. A position
    # with fewer than k alternatives gets zero-weight padding via the mask, so
    # padded entries contribute nothing to the partial-sum KL.
    n = len(topk_ids_list)
    teacher_ids = torch.zeros(n, k, dtype=torch.long)
    teacher_lp = torch.zeros(n, k, dtype=torch.float)
    mask = torch.zeros(n, k, dtype=torch.float)
    for i, (ids_row, lp_row) in enumerate(zip(topk_ids_list, topk_lp_list)):
        m = min(len(ids_row), k)
        if m:
            teacher_ids[i, :m] = torch.tensor(ids_row[:m], dtype=torch.long)
            teacher_lp[i, :m] = torch.tensor(lp_row[:m], dtype=torch.float)
            mask[i, :m] = 1.0

    # Full token sequence for the (deferred) student forward, without hint.
    full_ids = torch.cat([prefix_without_ids, completion_ids])

    return {
        "teacher_topk_ids": teacher_ids.to(device),
        "teacher_topk_logprobs": teacher_lp.to(device),
        "teacher_topk_mask": mask.to(device),
        "completion_tokens": completion_ids_list,
        "_full_ids": full_ids,
        "_n_completion": int(completion_ids.shape[0]),
    }


def prepare_length_probe(problem: str, completion_token_ids: list[int]) -> Optional[dict]:
    """Prepare the token-budget probe for one over-long rollout (no forward yet).

    The counterpart to prepare_probing_topk for rollouts that ran into (or close
    to) config.MAX_NEW_TOKENS. There is no teacher here: the rollout was cut off
    rather than reasoned wrongly, so nothing was distilled onto it. What this
    builds is the input to loss_script.run_length_penalty — the student's own
    rollout, re-conditioned on the prompt it was generated under, plus the
    per-position severity weights.

    Only the LAST config.LENGTH_PENALTY_WINDOW positions are scored. Those are the
    positions with the largest weights (w_i rises monotonically with index), and
    bounding them bounds the (M, V) logits + backward buffer. The forward itself
    still covers the whole rollout, because the scored positions have to be
    conditioned on everything that preceded them — that forward is the expensive
    part of this path, and it is why gradient checkpointing (enabled at load) is
    load-bearing here.

    Mirrors the generation context exactly: prompts.serving_prompt_ids is what
    vLLM conditioned the chat rollout on, and completion_token_ids are the ids it
    returned, so the scored positions are the same ones the model actually chose
    tokens at.

    Args:
        problem:             The problem text the rollout answered.
        completion_token_ids: The rollout's generated token ids, in order.

    Returns:
        A probe dict carrying "_full_ids" / "_n_completion" (the keys
        student_forward_batch consumes), "position_weights" (M,),
        "end_token_ids" (S,) and "n_generated". None when the probe cannot be
        built — no ids recovered, no turn-ending token in the vocabulary, or every
        scored position falling below the soft threshold — in which case the
        caller simply trains nothing for this item's length.
    """
    if not completion_token_ids:
        return None

    model, tokenizer = get_model_and_tokenizer()
    device = next(model.parameters()).device

    end_ids = prompts.turn_end_token_ids(tokenizer)
    if not end_ids:
        logger.warning(
            "No turn-ending token id available for %s; skipping the token-budget "
            "penalty (see prompts.turn_end_token_ids).", config.HF_MODEL_PATH,
        )
        return None

    n_generated = len(completion_token_ids)
    window = min(max(1, config.LENGTH_PENALTY_WINDOW), n_generated)
    start = n_generated - window

    # Weights are keyed to the position's index in the FULL rollout (not in the
    # window), so a window that only partly clears the soft threshold is only
    # partly penalised.
    weights = [config.length_penalty_weight(start + i) for i in range(window)]
    if not any(w > 0.0 for w in weights):
        return None

    prefix_ids = torch.tensor(
        prompts.serving_prompt_ids(config.INFERENCE_SYSTEM_PROMPT, problem, tokenizer),
        dtype=torch.long,
    )
    completion_ids = torch.tensor(completion_token_ids, dtype=torch.long)
    full_ids = torch.cat([prefix_ids, completion_ids])

    return {
        "_full_ids": full_ids,
        "_n_completion": window,
        "position_weights": torch.tensor(weights, dtype=torch.float, device=device),
        "end_token_ids": torch.tensor(end_ids, dtype=torch.long, device=device),
        "n_generated": n_generated,
    }


def student_forward_topk_batch(prepared: list[dict]) -> None:
    """Run the HF student forward for a batch of prepared probes, in place.

    Takes the list produced by prepare_probing_topk (or prepare_length_probe —
    the two agree on the "_full_ids" / "_n_completion" contract) and attaches
    "student_logits" ((N_i, V), grad-tracked) to each dict using a SINGLE batched
    forward. This is where actual batching happens: B items share one (B, L)
    forward + one shared autograd graph, instead of B sequential batch-size-1
    forwards.
    """
    if not prepared:
        return
    model, tokenizer = get_model_and_tokenizer()
    device = next(model.parameters()).device

    full_seqs = [p["_full_ids"] for p in prepared]
    completion_lens = [p["_n_completion"] for p in prepared]
    student_logits = _batched_completion_logits(
        model, full_seqs, completion_lens, device, no_grad=False,
        pad_id=tokenizer.pad_token_id,
    )
    for p, logits in zip(prepared, student_logits):
        p["student_logits"] = logits  # (N, V) — has grad


# The batched forward is shape-generic; "topk" in the name above is historical.
# Length probes (prepare_length_probe) go through the same call under this name.
student_forward_batch = student_forward_topk_batch
