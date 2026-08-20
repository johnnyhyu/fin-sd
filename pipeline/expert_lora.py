"""Memory-efficient expert LoRA for gpt-oss's fused MoE parameters.

PEFT reaches gpt-oss's experts through `LoraConfig.target_parameters`, and its
`ParamWrapper` implements the adapter the only way a parameter-level wrapper can:
it builds the FULL delta for the fused tensor and registers it as a
parametrization, so the base parameter is `W + ΔW` for the duration of the
forward. On the 120B that is ruinous. `mlp.experts.gate_up_proj` is
[128, 2880, 5760] — 4.25 GiB in bf16 — so each MoE layer materialises 4.25 GiB of
delta plus 4.25 GiB of summed weight, and `down_proj` adds 2 × 2.12 GiB. Measured
on one layer at 120B geometry (fwd+bwd, flex attention, 2048 tokens):

    PEFT ParamWrapper .................. 18.1 GiB transient, 209 ms
    this module, all 128 experts ....... 0.69 GiB transient, 302 ms
    this module, 20 profiled experts ... 0.62 GiB transient, 194 ms

and that transient is paid on every card, since `device_map="auto"` runs one
layer at a time. It is what forced HF_MAX_MEMORY down to 40 GiB per 80 GiB card.

The delta is never needed as a tensor. `GptOssExperts.forward` (transformers'
training branch) already loops over the experts a batch actually routed to,
slicing ONE expert's weights per iteration, so the adapter can be applied in
activation space where LoRA is normally applied:

    x @ (W_e + ΔW_e)  ==  x @ W_e + ((x @ B_e) @ A_e) · scaling

which touches only the ~64 tokens routed to that expert and two rank-8 factors.
Verified against PEFT's own `get_delta_weight` to 3.6e-7 in fp32.

Two consequences make this more than an optimisation:

  • Sparse allocation becomes possible. `PROFILE_EXPERT_ACTIVATIONS` already
    restricts adaptation to the busiest experts per layer, but PEFT can only
    express that as a MASK over a full-width adapter — every frozen expert still
    carries parameters, gradients and AdamW moments (~7 GiB on the 120B) and is
    still multiplied through on every step. Here the frozen experts simply have
    no parameters, which is also why the 20-expert row above is FASTER than PEFT.
  • Merging gets 128× lighter, since it accumulates per expert instead of
    building the fused delta (this is what `save_merged_model` and
    `tools/serve_checkpoint.py` run at every epoch boundary).

What deliberately does NOT change is the checkpoint. PEFT still owns the
`lora_A`/`lora_B` tensors, `adapter_config.json`, and the merge entry points, and
`install()` re-expands the sparse tensors to their full [r·n_experts, …] shape on
save (and compresses on load), so `adapter_model.safetensors` is byte-compatible
with what the pipeline wrote before — `tools/serve_checkpoint.py`,
`tools/serve_epoch.py` and every existing checkpoint keep working untouched.

If anything about PEFT's layout stops matching what `_describe` asserts (a PEFT
upgrade, a different MoE model), `install()` returns None having changed nothing,
and the caller falls back to stock PEFT behaviour plus the old grad masks. The
expensive path is the one that still works.
"""
from __future__ import annotations

import contextlib
import re
import types
from dataclasses import dataclass
from typing import Optional

import torch

from . import config
from .utils import logger

# The fused expert parameters this module knows how to adapt. Anything else
# targeted via target_parameters is left to PEFT.
_EXPERT_PARAMS = ("gate_up_proj", "down_proj")

# Set once GptOssExperts.forward has been patched, so a second model build in one
# process doesn't wrap it twice.
_FORWARD_PATCHED = False


@dataclass
class _ExpertAdapter:
    """One (wrapper, parameter) pair's adapter, in the layout the forward needs.

    `wrapper` is PEFT's ParamWrapper — held live rather than copied out, so the
    forward sees optimizer updates, `disable_adapter()` (FROZEN_TEACHER, the EXOPD
    reference pass) and `merge_adapter()` without any bookkeeping of our own.

    `slots` maps an expert index to its row-block in the (possibly sparse)
    factors, or -1 for an expert this layer does not adapt. It is a plain Python
    list, not a tensor: the forward looks it up once per routed expert, and
    reading an element of a CUDA tensor there would force a host synchronisation
    inside the expert loop — 128 of them per layer.
    """

    wrapper: torch.nn.Module
    adapter: str
    slots: list                        # len n_experts, -1 where frozen
    n_slots: int                       # experts actually carrying parameters
    n_experts: int                     # experts in the base parameter
    rank: int

    @property
    def active(self) -> bool:
        """Should the forward add this adapter's delta at all?

        Mirrors ParamWrapper.forward's own branches: a disabled adapter
        contributes nothing, and a merged one is already inside the base weights
        (adding it again would double it).
        """
        return not self.wrapper.disable_adapters and not self.wrapper.merged

    def factors(self, slot: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        """(B_e, A_e, scaling) for one slot — views, never copies.

        PEFT stores expert `e`'s block contiguously in lora_A's rows
        (`row // r == e`) and strided in lora_B's columns (`col % n_slots == e`);
        `_install_expert_masks` documents the same layout from the masking side.
        With the (experts, in, out) parameter layout PEFT swaps in/out features,
        which makes lora_B the input-side factor and lora_A the output-side one —
        i.e. the delta is B_e @ A_e, not the other way round.
        """
        wA = self.wrapper.lora_A[self.adapter].weight      # (r·n_slots, out_dim)
        wB = self.wrapper.lora_B[self.adapter].weight      # (in_dim, r·n_slots)
        a = wA.view(self.n_slots, self.rank, wA.shape[-1])[slot]        # (r, out)
        b = wB.view(wB.shape[0], self.rank, self.n_slots)[:, :, slot]   # (in, r)
        return b, a, self.wrapper.scaling[self.adapter]

    def delta(self, slot: int) -> torch.Tensor:
        """One expert's dense ΔW (in, out) — used by merge, not by the forward."""
        b, a, scaling = self.factors(slot)
        return (b @ a) * scaling

    def apply(self, slot: int, x: torch.Tensor) -> torch.Tensor:
        """This expert's contribution to `x`'s output: ((x @ B_e) @ A_e) · scaling.

        The factors are fp32 (PEFT's `autocast_adapter_dtype`, so the adapter keeps
        its precision under a bf16 base) while `x` is bf16, so the product is
        computed in the factors' dtype and cast back — the same discipline
        lora.Linear.forward uses, and the activation-space counterpart of PEFT
        casting its fp32 delta down to the parameter dtype before adding it.
        """
        b, a, scaling = self.factors(slot)
        out = ((x.to(b.dtype) @ b) @ a) * scaling
        return out.to(x.dtype)


def _describe(wrapper, adapter: str) -> Optional[tuple[int, int, int, int]]:
    """Validate PEFT's layout for one wrapper; return (n_experts, r, in, out).

    Everything this module does rests on the shapes PEFT chose for the factors
    and on the in/out swap it applies to 3-D parameters. Rather than trusting
    that across PEFT versions, check it: a mismatch returns None and the caller
    leaves the stock (expensive but correct) path in place.
    """
    param = wrapper.get_param()
    if param.ndim != 3:
        return None
    n_experts, dim_in, dim_out = param.shape
    r = wrapper.r.get(adapter)
    if not r or adapter not in wrapper.lora_A:
        return None
    if not getattr(wrapper, "_did_swap_in_out_features", False):
        return None                     # (experts, out, in) layout — not ours
    wA = wrapper.lora_A[adapter].weight
    wB = wrapper.lora_B[adapter].weight
    if tuple(wA.shape) != (r * n_experts, dim_out):
        return None
    if tuple(wB.shape) != (dim_in, r * n_experts):
        return None
    return n_experts, r, dim_in, dim_out


def _full_index(kind: str, keep: list[int], n_experts: int, r: int) -> torch.Tensor:
    """Positions in the FULL-width factor that the compact factor maps onto.

    `kind` is "A" (rows, expert-major: e·r + j) or "B" (columns, rank-major:
    j·n_experts + e). Used to expand on save and to compress on load, so the two
    directions can never disagree about the layout.
    """
    if kind == "A":
        return torch.tensor([e * r + j for e in keep for j in range(r)], dtype=torch.long)
    return torch.tensor([j * n_experts + e for j in range(r) for e in keep], dtype=torch.long)


def _sparsify(wrapper, adapter: str, keep: list[int], n_experts: int, r: int) -> None:
    """Shrink this wrapper's factors to cover only `keep`, preserving their values.

    The frozen experts' blocks are zero (PEFT zero-initialises lora_B, and the
    profiled selection freezes the rest), so dropping them changes no arithmetic —
    it only stops allocating parameters, gradients and optimizer moments for
    experts whose delta is identically zero. `install()` re-expands them on save.
    """
    lin_a, lin_b = wrapper.lora_A[adapter], wrapper.lora_B[adapter]
    wA, wB = lin_a.weight, lin_b.weight
    rows = _full_index("A", keep, n_experts, r).to(wA.device)
    # lora_B's expert blocks are strided (col % n_experts == e), so the kept
    # columns must be listed in the same (rank-major, expert-minor) order the
    # compacted view reads them back in.
    cols = _full_index("B", keep, n_experts, r).to(wB.device)
    with torch.no_grad():
        new_a = wA.index_select(0, rows).clone()
        new_b = wB.index_select(1, cols).clone()
    lin_a.weight = torch.nn.Parameter(new_a, requires_grad=wA.requires_grad)
    lin_b.weight = torch.nn.Parameter(new_b, requires_grad=wB.requires_grad)
    lin_a.out_features = lin_b.in_features = r * len(keep)


def _register_checkpoint_hooks(wrapper, adapter: str, keep: list[int],
                               n_experts: int, r: int, dim_in: int, dim_out: int) -> None:
    """Make a sparse adapter save and load as a full-width one.

    Sparse allocation is a memory decision, not a format change: the file on disk
    stays exactly what the pipeline wrote before (full [r·n_experts, …] tensors
    with zeros where an expert is frozen), so `tools/serve_checkpoint.py`, which
    rebuilds a fresh PEFT tree and calls `load_state_dict` into it, needs no
    knowledge of any of this — and adapters trained before this change still load.

    The expansion runs on CPU: the only caller that wants these tensors is a save,
    which copies to CPU anyway, and `save_merged_model` builds a state dict it
    immediately discards the LoRA keys from — neither should cost GPU memory.
    """
    idx_a = _full_index("A", keep, n_experts, r)
    idx_b = _full_index("B", keep, n_experts, r)

    def _expand(_module, state_dict, prefix, *_args):
        for name, idx, shape, dim in (
            (f"{prefix}lora_A.{adapter}.weight", idx_a, (r * n_experts, dim_out), 0),
            (f"{prefix}lora_B.{adapter}.weight", idx_b, (dim_in, r * n_experts), 1),
        ):
            compact = state_dict.get(name)
            if compact is None or compact.shape[dim] == shape[dim]:
                continue
            full = torch.zeros(shape, dtype=compact.dtype, device="cpu")
            full.index_copy_(dim, idx, compact.detach().to("cpu"))
            state_dict[name] = full
        # torch requires a state_dict post-hook to return None; the dict above is
        # edited in place.

    def _compress(_module, state_dict, prefix, *_args):
        for name, idx, dim in ((f"{prefix}lora_A.{adapter}.weight", idx_a, 0),
                               (f"{prefix}lora_B.{adapter}.weight", idx_b, 1)):
            full = state_dict.get(name)
            if full is None or full.shape[dim] != r * n_experts:
                continue          # already compact (an in-process round trip)
            compact = full.index_select(dim, idx.to(full.device))
            # Anything outside the kept slots must be zero, or the checkpoint was
            # trained against a DIFFERENT expert profile and silently dropping
            # those blocks would change the model. Say so rather than lose them.
            # Compared as ABSOLUTE mass: a signed comparison of the two sums can
            # cancel to zero across the dropped blocks and report a clean load for
            # a checkpoint that is losing half its experts.
            kept_mass = compact.abs().sum().to(torch.float64)
            dropped = full.abs().sum().to(torch.float64) - kept_mass
            if float(dropped) > max(1e-3, 1e-6 * float(kept_mass)):
                logger.warning(
                    "%s carries non-zero LoRA weight for experts this run does not "
                    "adapt — the checkpoint was trained under a different expert "
                    "profile (EXPERT_PROFILE_PATH / PROFILE_EXPERT_MASS_THRESHOLD). "
                    "Those experts' contribution is being dropped.", name,
                )
            state_dict[name] = compact

    wrapper.register_state_dict_post_hook(_expand)
    wrapper.register_load_state_dict_pre_hook(_compress)


def _bind_merge(wrapper, entry: "_ExpertAdapter") -> None:
    """Replace PEFT's fused merge/unmerge with a per-expert accumulation.

    PEFT's `merge()` calls `get_delta_weight`, which builds the whole
    [n_experts, in, out] delta — 4.25 GiB for one 120B `gate_up_proj`, and the
    reason a merge needs a 6-card block today. Accumulating expert by expert costs
    one [in, out] slice (33 MiB) instead, and it is also the only version that
    works once the factors are sparse, since a compact factor cannot be reshaped
    to the base parameter's expert count.

    Bookkeeping (`merged_adapters`) mirrors PEFT's so `self.merged`, the forward's
    merged branch, and `unmerge_adapter()` all keep behaving normally.
    """

    def merge(self, safe_merge: bool = False, adapter_names=None) -> None:
        from peft.tuners.tuners_utils import check_adapters_to_merge

        names = check_adapters_to_merge(self, adapter_names)
        if not names:
            return
        param = self.get_param()
        for name in names:
            if name != entry.adapter:                # a second adapter we don't own
                raise RuntimeError(
                    f"expert_lora cannot merge adapter {name!r}; only "
                    f"{entry.adapter!r} was installed."
                )
            with torch.no_grad():
                for expert, slot in enumerate(entry.slots):
                    if slot < 0:
                        continue
                    delta = entry.delta(slot).to(param.dtype)
                    if safe_merge and not torch.isfinite(delta).all():
                        raise ValueError(
                            f"NaNs in the merged weights for expert {expert} of "
                            f"adapter {name}; the adapter seems to be broken."
                        )
                    param.data[expert] += delta
            self.merged_adapters.append(name)

    def unmerge(self) -> None:
        if not self.merged:
            return
        param = self.get_param()
        while self.merged_adapters:
            name = self.merged_adapters.pop()
            if name != entry.adapter:
                continue
            with torch.no_grad():
                for expert, slot in enumerate(entry.slots):
                    if slot < 0:
                        continue
                    param.data[expert] -= entry.delta(slot).to(param.dtype)

    wrapper.merge = types.MethodType(merge, wrapper)
    wrapper.unmerge = types.MethodType(unmerge, wrapper)
    # The parametrization is what materialised the fused delta; with the adapter
    # applied inside the expert loop there is nothing to register, so the context
    # becomes a no-op. Everything else about the wrapper (state dict, config,
    # disable_adapters, merged) is untouched.
    wrapper._activate_lora = types.MethodType(
        lambda self, active_adapters: contextlib.nullcontext(), wrapper
    )


def _patch_expert_forward() -> None:
    """Teach GptOssExperts.forward to apply the adapter inside its expert loop.

    Forked from transformers' GptOssExperts.forward (4.57) rather than wrapped,
    because the whole point is to reach INSIDE the per-expert iteration: the
    adapter has to be applied to the ~64 tokens routed to expert `e`, using
    expert `e`'s two rank-r factors, at the point where the base weight slice is
    used. There is no seam in the upstream function to hook that onto.

    Modules with no adapter installed (the profiler's base model, a non-LoRA run)
    fall through to the original implementation, so this patch is inert unless
    `install()` attached something.
    """
    global _FORWARD_PATCHED
    if _FORWARD_PATCHED:
        return
    from transformers.models.gpt_oss import modeling_gpt_oss as gpt_oss

    original = gpt_oss.GptOssExperts.forward

    def forward(self, hidden_states, router_indices=None, routing_weights=None):
        adapters = getattr(self, "_expert_lora", None)
        live = [a for a in (adapters or {}).values() if a.active]
        if not live:
            # No adapter to apply (or it is merged/disabled): upstream's own
            # implementation, including its faster dense inference branch.
            return original(self, hidden_states, router_indices=router_indices,
                            routing_weights=routing_weights)

        gate_up_lora = adapters.get("gate_up_proj")
        down_lora = adapters.get("down_proj")
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        num_experts = routing_weights.shape[1]
        next_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                router_indices, num_classes=num_experts + 1
            ).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit[:]:
            # Upstream keeps `expert_idx` as a 0-d CUDA tensor. Reading it once as a
            # Python int costs the same host sync its own `== num_experts` test
            # already pays, and buys two things back: the adapter's slot lookup
            # becomes a list index instead of a second sync, and `self.gate_up_proj
            # [expert]` becomes basic indexing — a VIEW — where indexing with a
            # tensor is advanced indexing and copies the whole 33 MiB expert slice.
            expert = int(expert_idx[0])
            if expert == num_experts:                   # the masking class
                continue
            with torch.no_grad():
                _, token_idx = torch.where(expert_mask[expert])
            current_state = hidden_states[token_idx]
            gate_up = current_state @ self.gate_up_proj[expert] \
                + self.gate_up_proj_bias[expert]
            if gate_up_lora is not None and gate_up_lora.active:
                slot = gate_up_lora.slots[expert]
                if slot >= 0:
                    gate_up = gate_up + gate_up_lora.apply(slot, current_state)
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)
            gated_output = (up + 1) * glu
            out = gated_output @ self.down_proj[expert] \
                + self.down_proj_bias[expert]
            if down_lora is not None and down_lora.active:
                slot = down_lora.slots[expert]
                if slot >= 0:
                    out = out + down_lora.apply(slot, gated_output)
            out = out * routing_weights[token_idx, expert, None]
            next_states.index_add_(0, token_idx, out.to(hidden_states.dtype))
        return next_states.view(batch_size, -1, self.hidden_size)

    gpt_oss.GptOssExperts.forward = forward
    _FORWARD_PATCHED = True
    logger.info("gpt-oss expert forward patched: LoRA applies per expert in "
                "activation space (no fused delta materialised).")


def install(model, adapter_name: str, selected_by_layer: Optional[dict] = None) -> Optional[dict]:
    """Convert a PEFT expert-LoRA model onto the activation-space path.

    Call once, immediately after `get_peft_model`. Returns a summary dict
    ({"layers", "kept_min", "kept_max", "kept_mean", "n_experts", "params"}), or
    None if this model is not a gpt-oss expert-LoRA tree or PEFT's layout is not
    the one `_describe` expects — in which case NOTHING has been changed and the
    caller should keep the stock path (and its grad masks).

    `selected_by_layer` is the profiled per-layer expert selection
    (probing_script._selected_experts_from_profile). When given, only those
    experts get parameters at all; when None every expert is adapted, which still
    avoids the fused delta but keeps the full optimizer state.
    """
    try:
        import peft
        from peft.tuners.lora.layer import ParamWrapper
    except ImportError:
        return None

    wrappers: list[tuple[str, object]] = [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, ParamWrapper)
        and getattr(module, "parameter_name", None) in _EXPERT_PARAMS
    ]
    if not wrappers:
        return None

    entries: list[tuple[object, _ExpertAdapter]] = []
    kept_per_layer: dict[int, int] = {}
    n_experts_seen = 0
    for name, wrapper in wrappers:
        described = _describe(wrapper, adapter_name)
        if described is None:
            logger.warning(
                "PEFT's expert-LoRA layout for %s is not the one pipeline."
                "expert_lora expects (peft %s); keeping the stock fused-delta "
                "path, which needs ~18 GiB per layer on the 120B.",
                name, getattr(peft, "__version__", "?"),
            )
            return None
        n_experts, r, dim_in, dim_out = described
        n_experts_seen = n_experts
        layer_match = re.search(r"layers\.(\d+)\.", name)
        keep = None
        if selected_by_layer is not None and layer_match is not None:
            keep = selected_by_layer.get(int(layer_match.group(1)))
        keep = sorted(keep) if keep else list(range(n_experts))
        if len(keep) < n_experts:
            _sparsify(wrapper, adapter_name, keep, n_experts, r)
            _register_checkpoint_hooks(wrapper, adapter_name, keep, n_experts, r,
                                       dim_in, dim_out)
        slots = [-1] * n_experts
        for slot, expert in enumerate(keep):
            slots[expert] = slot
        entry = _ExpertAdapter(
            wrapper=wrapper, adapter=adapter_name,
            slots=slots,
            n_slots=len(keep), n_experts=n_experts, rank=r,
        )
        entries.append((wrapper, entry))
        if layer_match is not None:
            kept_per_layer[int(layer_match.group(1))] = len(keep)

    # Only once every wrapper validated: an all-or-nothing switch, so a partial
    # conversion can never leave some layers applying the delta twice.
    _patch_expert_forward()
    for wrapper, entry in entries:
        experts = wrapper.get_base_layer()
        store = getattr(experts, "_expert_lora", None)
        if store is None:
            store = {}
            experts._expert_lora = store
        store[wrapper.parameter_name] = entry
        _bind_merge(wrapper, entry)

    counts = sorted(kept_per_layer.values()) or [n_experts_seen]
    trainable = sum(p.numel() for _, w in entries for p in
                    (w.wrapper.lora_A[adapter_name].weight,
                     w.wrapper.lora_B[adapter_name].weight))
    logger.info(
        "Expert LoRA on the activation-space path: %d wrapper(s) over %d layer(s), "
        "%d–%d of %d experts adapted per layer (mean %.1f), %s expert LoRA "
        "parameter(s). No fused delta is materialised; merging accumulates per "
        "expert.",
        len(entries), len(kept_per_layer) or 1, counts[0], counts[-1],
        n_experts_seen, sum(counts) / len(counts), f"{trainable:,}",
    )
    return {
        "layers": len(kept_per_layer),
        "kept_min": counts[0],
        "kept_max": counts[-1],
        "kept_mean": sum(counts) / len(counts),
        "n_experts": n_experts_seen,
        "params": trainable,
    }


def is_enabled() -> bool:
    """Whether the activation-space path should be used (escape hatch: env)."""
    return config.EXPERT_LORA_ACTIVATION_SPACE


def merge_fidelity(model, sample: int = 4) -> Optional[tuple[float, float]]:
    """How much of the expert delta survives being merged into the bf16 base.

    Merging writes `W + ΔW` back into a bf16 parameter, and bf16 carries 8 mantissa
    bits — one ulp is ~0.4% of |W|. A trained expert delta is far below that: across
    this repo's own expert-LoRA checkpoints it runs 0.001-0.02% of the weight scale
    after one epoch and 0.06-0.3% after eight, so most elements never move the
    weight at all while a few jump a whole ulp.

    Returns `(surviving, noise)`, both against what the merge ACTUALLY applies:

        surviving = <ΔW_effective, ΔW> / <ΔW, ΔW>     (1.0 = merged exactly)
        noise     = ‖ΔW_effective − ΔW‖ / ‖ΔW‖

    The projection is the honest quantity, and the reason this function does not
    simply compare magnitudes: a magnitude comparison counts the quantization
    residual as lost signal and reports catastrophe (88%!) even where the update
    lands largely intact. Measured on real checkpoints, `surviving` is 0.17 at
    epoch 1 of one run and 0.92 by epoch 4 of a longer one, with `noise` around
    0.45-0.9 — the update is coarsely quantized, not uniformly shrunk, and it is
    the small early deltas that barely register.

    Worth logging because the merged checkpoint IS what vLLM serves at every epoch
    boundary, so this is the share of the epoch's expert training that reaches the
    student generating the next epoch's rollouts — invisible otherwise, since both
    the adapter and the checkpoint look perfectly healthy. It is a property of
    merging into bf16, not of this module: the same quantization applied before,
    and applied to PEFT's forward as well, which folded the delta into bf16 weights
    on EVERY forward and so trained against a student that had already discarded it.

    Samples `sample` wrappers rather than all of them — the ratio tracks the weight
    and delta scales, which barely vary across layers, and a full sweep would cost a
    dense delta per expert parameter. Returns None when there is nothing to measure.
    """
    entries = []
    for module in model.modules():
        store = getattr(module, "_expert_lora", None)
        if store:
            entries.extend(store.values())
    if not entries:
        return None
    step = max(1, len(entries) // max(1, sample))
    dot = sq = residual = 0.0
    with torch.no_grad():
        for entry in entries[::step][:sample]:
            param = entry.wrapper.get_param()
            for expert, slot in enumerate(entry.slots):
                if slot < 0:
                    continue
                delta = entry.delta(slot).float()
                base = param.data[expert].float()
                # What the merge applies: the bf16 round trip of base + delta.
                effective = (base + delta.to(param.dtype)).float() - base
                dot += float((effective * delta).sum())
                sq += float((delta * delta).sum())
                residual += float(((effective - delta) ** 2).sum())
                break                      # one expert per wrapper is enough
    if sq <= 0:
        return None
    return dot / sq, (residual / sq) ** 0.5
