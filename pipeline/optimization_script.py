"""
optimization_script.py — backpropagates the KL loss and updates model weights.

The optimizer (AdamW) is initialised lazily on first call and persisted across
training steps so that momentum/variance estimates accumulate correctly. A
LambdaLR scheduler applies a short cosine LR warmup over the first
NUM_WARMUP_STEPS optimizer steps, multiplied by a cosine DECAY to
LR_DECAY_FLOOR·LEARNING_RATE run on EPOCH progress (see _lr_lambda).
AdamW's beta2 defaults to 0.95 rather than 0.999 so the second-moment estimate
adapts within this run's small (tens-of-steps) optimizer budget.

Why the decay is keyed to epochs and not steps: the total optimizer-step count is
not knowable up front (it depends on how many items are answered incorrectly per
epoch), which is why this schedule used to be warmup-then-constant. But
NUM_EPOCHS *is* known, so the anneal runs against that horizon instead — the run
then ends on small steps rather than on its largest ones. Callers drive it with
set_epoch() at each epoch boundary; a caller that never calls set_epoch simply
trains at the epoch-1 (full) LR, i.e. the old behaviour.

Gradients are accumulated over GRAD_ACCUM_STEPS micro-batches before each
optimizer step. A micro-batch is one run_optimization call (a single item, the
full-vocab path), one run_optimization_batch call (up to TRAIN_BATCH_SIZE items
sharing one batched forward, the top-k path), or — in the student-rollout path —
one problem's N rollouts backpropped via accumulate_loss(scale=1/N) and sealed by
a single commit_micro_batch() call. So GRAD_ACCUM_STEPS always counts problems
and the effective batch is TRAIN_BATCH_SIZE × GRAD_ACCUM_STEPS items
(= GRAD_ACCUM_STEPS when TRAIN_BATCH_SIZE is 1). Each micro-batch loss is scaled
by 1/GRAD_ACCUM_STEPS so micro-batches are weighted equally. Callers must flush_gradients() before saving
the adapter so a partial accumulation window is not silently dropped; the flush
step is proportionally smaller, which is the standard treatment of a remainder.

Gradient clipping (max_norm=1.0) is applied before each step to prevent
exploding gradients on long reasoning sequences.

Weight-sync note:
    After each call to run_optimization(), the HuggingFace model weights diverge
    from the vLLM server's weights. run_pipeline.py resyncs at each epoch boundary
    via one of two paths (pipeline.vllm_server):
      • expert LoRA (config.LORA_TARGET_EXPERTS) — gpt-oss's fused-MoE experts have
        no per-expert modules for vLLM to attach an adapter to, so save_merged_model()
        writes a clean full checkpoint with the LoRA baked in and vLLM is restarted
        on it;
      • non-expert LoRA — the saved adapter (save_adapter) is hot-loaded into the
        live vLLM server, no merge or restart needed.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union

import torch

from . import config
from .utils import logger
from .probing_script import get_model_and_tokenizer  # shared singleton

_optimizer: Optional[torch.optim.Optimizer] = None
_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
_accum_count: int = 0
# 1-based epoch driving the LR decay factor (see _lr_lambda / set_epoch). Held
# here rather than passed through every call site because LambdaLR's lambda only
# receives the step index. Not part of the saved scheduler state — a resumed run
# sets it from its own epoch loop before the first step.
_current_epoch: int = 1


def set_epoch(epoch: int) -> None:
    """Point the LR decay at a (1-based) epoch; call once per epoch, before training.

    Only affects the decay factor. The warmup factor stays keyed to the optimizer
    step index, so restarting an epoch does not re-ramp the LR.

    The live param-group LRs are refreshed here rather than left to the next
    scheduler.step(): LambdaLR only re-evaluates its lambda when it steps, so
    without this the first optimizer step of each epoch would still run at the
    previous epoch's LR — which, at ~4 steps per epoch, is a quarter of the run.
    """
    global _current_epoch
    _current_epoch = max(1, int(epoch))
    if _optimizer is not None and _scheduler is not None:
        factor = _lr_lambda(_scheduler.last_epoch)
        for group, base_lr in zip(_optimizer.param_groups, _scheduler.base_lrs):
            group["lr"] = base_lr * factor
        _scheduler._last_lr = [g["lr"] for g in _optimizer.param_groups]


def current_lr() -> float:
    """The LR the next optimizer step will use (0.0 before the optimizer exists)."""
    if _optimizer is None:
        return 0.0
    return float(_optimizer.param_groups[0]["lr"])

def save_adapter(out_dir: Union[str, Path]) -> str:
    """Save the trained LoRA adapter to `out_dir` and return its path."""
    model, _ = get_model_and_tokenizer()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    logger.info("LoRA adapter saved to %s", out)
    return str(out)


def save_optimizer_state(out_dir: Union[str, Path]) -> None:
    """Persist the AdamW + LR-scheduler state alongside the adapter checkpoint.

    A resumed run must restore not just the LoRA weights but the optimizer's
    moment estimates (exp_avg / exp_avg_sq) and the scheduler's step count —
    otherwise AdamW restarts cold and the cosine warmup (_warmup_lr_lambda)
    re-ramps from zero mid-run. Written to CPU so it reloads regardless of the
    device_map="auto" shard placement. No-op if training never initialised the
    optimizer (nothing to save).
    """
    if _optimizer is None:
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "optimizer": _optimizer.state_dict(),
            "scheduler": _scheduler.state_dict() if _scheduler is not None else None,
        },
        out / "optim_state.pt",
    )
    logger.info("Optimizer/scheduler state saved to %s", out / "optim_state.pt")


def load_optimizer_state(in_dir: Union[str, Path]) -> None:
    """Restore AdamW + scheduler state onto the freshly-built optimizer.

    Caller ordering is load-bearing: the LoRA adapter weights must already be
    loaded and get_optimizer() must have bound AdamW to those params before this
    call, so the state_dict maps back by param index. torch.load(map_location=
    "cpu") + load_state_dict moves each moment tensor onto its param's device, so
    this is correct under the sharded device_map. No-op if the checkpoint has no
    optim_state.pt (e.g. an older checkpoint that predates this feature).
    """
    path = Path(in_dir) / "optim_state.pt"
    if not path.exists():
        logger.warning("No optim_state.pt in %s; optimizer will start cold.", in_dir)
        return
    optimizer = get_optimizer()
    ckpt = torch.load(path, map_location="cpu")
    optimizer.load_state_dict(ckpt["optimizer"])
    if _scheduler is not None and ckpt.get("scheduler") is not None:
        _scheduler.load_state_dict(ckpt["scheduler"])
    logger.info("Optimizer/scheduler state restored from %s", path)


def load_adapter_weights(in_dir: Union[str, Path]) -> None:
    """Overwrite the live PEFT adapter's weights with those saved in `in_dir`.

    Used on resume: get_model_and_tokenizer() has already built the base model
    with a fresh (randomly-initialised) LoRA adapter via the usual construction
    path; this replaces those weights in place with the checkpointed ones,
    leaving requires_grad and the module wiring untouched so training (and the
    optimizer binding) continue as normal.
    """
    from peft import load_peft_weights, set_peft_model_state_dict

    model, _ = get_model_and_tokenizer()
    state_dict = load_peft_weights(str(in_dir))
    result = set_peft_model_state_dict(model, state_dict)
    unexpected = getattr(result, "unexpected_keys", None)
    if unexpected:
        logger.warning("Adapter load from %s had unexpected keys: %s", in_dir, unexpected)
    logger.info("LoRA adapter weights loaded from %s", in_dir)


def save_merged_model(out_dir: Union[str, Path], model=None, tokenizer=None) -> str:
    """Save a clean full checkpoint with the LoRA merged into the base weights.

    Used to refresh the vLLM-served weights at each epoch boundary. gpt-oss packs
    its MoE experts as fused 3-D nn.Parameters with no per-expert modules, so vLLM
    has nothing to attach an expert LoRA adapter to — the trained weights have to
    be baked into a full checkpoint and served directly (see vllm_server.restart).

    The merge is done in place with PEFT's reversible merge_adapter()/
    unmerge_adapter() so training continues on the same adapter afterward. A clean
    state dict is recovered from the (otherwise PEFT-mangled) merged module tree by
    dropping the lora_* tensors and stripping PEFT's `base_layer` wrapper infixes,
    yielding the stock gpt-oss keys vLLM expects. The bf16 merge/unmerge round-trip
    leaves negligible drift in the base weights across epochs.

    `model`/`tokenizer` default to the RL pipeline's shared probing singleton; the
    SFT flow passes its own Trainer model so it can merge mid-training for
    end-of-epoch validation without touching that singleton.
    """
    import re
    from safetensors.torch import save_file

    if model is None or tokenizer is None:
        model, tokenizer = get_model_and_tokenizer()

    # A model with no adapter to merge (the SFT flow's non-LoRA runs) already IS
    # the checkpoint, so write the live weights straight out.
    if not getattr(model, "peft_config", None) or not hasattr(model, "merge_adapter"):
        return save_full_model(out_dir, model, tokenizer)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # How much of the epoch's expert training survives the merge. Writing W + ΔW
    # back into a bf16 parameter quantizes to ~0.4% of |W| per element, and a
    # trained expert delta is well under that — and this checkpoint is exactly what
    # vLLM then serves, so whatever does not survive here never reaches the student
    # that generates the next epoch's rollouts. See expert_lora.merge_fidelity.
    from . import expert_lora
    fidelity = expert_lora.merge_fidelity(model)
    if fidelity is not None:
        surviving, noise = fidelity
        log = logger.warning if surviving < 0.5 else logger.info
        log(
            "Merged checkpoint carries %.0f%% of the expert LoRA delta (quantization "
            "noise %.2f× the delta): bf16's rounding step is ~0.4%% of |W| and this "
            "epoch's delta is smaller. Raise LORA_ALPHA or LORA_EXPERT_RANK if it "
            "stays low — the served student is what the next epoch trains against.",
            100 * surviving, noise,
        )

    model.merge_adapter()
    try:
        clean: dict[str, torch.Tensor] = {}
        for key, tensor in model.get_base_model().state_dict().items():
            if ".lora_A." in key or ".lora_B." in key:
                continue
            # Strip PEFT's wrapper infixes (`.base_layer`, doubled for the
            # twice-wrapped experts module) to recover the stock gpt-oss key.
            clean_key = re.sub(r"(\.base_layer)+", "", key)
            clean[clean_key] = tensor.detach().to("cpu").clone()
    finally:
        model.unmerge_adapter()

    save_file(clean, str(out / "model.safetensors"))
    base = model.get_base_model()
    base.config.save_pretrained(str(out))
    # gpt-oss keeps its harmony stop tokens (<|return|>, <|call|>, <|endoftext|>)
    # in generation_config.json, NOT config.json — without it vLLM falls back to
    # config.json's lone eos_token_id, omits <|call|> (200012), and never stops at
    # a tool-call boundary. The model then over-generates and vLLM's harmony parser
    # dies ("expecting start token 200006"). Write it so the merged checkpoint stops
    # exactly like the base model.
    if getattr(base, "generation_config", None) is not None:
        base.generation_config.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    logger.info("Merged full checkpoint saved to %s (%d tensors).", out, len(clean))
    return str(out)


def save_full_model(out_dir: Union[str, Path], model=None, tokenizer=None) -> str:
    """Write the full (non-PEFT) student weights + config + tokenizer to `out_dir`.

    save_merged_model's fallback for a model that carries no adapter to merge (the
    SFT flow's non-LoRA runs): the live model IS the checkpoint. save_pretrained
    shards the state dict across safetensors files under the sharded device_map,
    and the written directory is a stock HF checkpoint vLLM can serve directly
    (see vllm_server.restart).
    """
    if model is None or tokenizer is None:
        model, tokenizer = get_model_and_tokenizer()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    logger.info("Full checkpoint saved to %s", out)
    return str(out)


def _warmup_factor(step: int) -> float:
    """Cosine ramp from ~0 to 1.0 over the first W optimizer steps, then 1.0.

    `step` is the LambdaLR step index (0-based, incremented after each
    optimizer.step()). Using (step + 1) avoids a wasted zero-LR first update.
    """
    warmup = config.NUM_WARMUP_STEPS
    if warmup <= 0:
        return 1.0
    t = min(step + 1, warmup)
    return 0.5 * (1.0 - math.cos(math.pi * t / warmup))


def _decay_factor(epoch: int) -> float:
    """Cosine decay from 1.0 at epoch 1 down toward LR_DECAY_FLOOR at the last epoch.

    Runs on epoch progress (epoch−1)/NUM_EPOCHS rather than on step progress,
    because the total step count is not known up front — see the module docstring.
    The last epoch therefore lands slightly above the floor rather than exactly on
    it (at 10 epochs: ~0.12 of peak), which is the intended shape: the floor is a
    minimum step size, not a target to converge onto.
    """
    floor = config.LR_DECAY_FLOOR
    if floor >= 1.0 or config.NUM_EPOCHS <= 1:
        return 1.0
    progress = min(1.0, max(0.0, (epoch - 1) / config.NUM_EPOCHS))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _lr_lambda(step: int) -> float:
    """LambdaLR multiplier: step-indexed warmup × epoch-indexed cosine decay."""
    return _warmup_factor(step) * _decay_factor(_current_epoch)


def get_optimizer() -> torch.optim.Optimizer:
    """Lazily create and cache the AdamW optimiser + LR scheduler."""
    global _optimizer, _scheduler
    if _optimizer is None:
        model, _ = get_model_and_tokenizer()
        trainable = [p for p in model.parameters() if p.requires_grad]
        adamw_kwargs = dict(
            lr=config.LEARNING_RATE,
            betas=(config.ADAM_BETA1, config.ADAM_BETA2),
            weight_decay=config.WEIGHT_DECAY,
            eps=1e-8,
        )
        # Fused rather than the default `foreach` path. Same update rule and the
        # same fp32 arithmetic, but foreach stacks the whole parameter list into
        # temporaries on each step — for the 120B's expert LoRA that is a couple of
        # GiB of transient allocation per step, on cards already holding ~55 GiB of
        # weights. Fused does it in one kernel with no intermediates, grouping by
        # device so the sharded device_map is fine. Falls back if this torch/device
        # combination refuses (fused requires floating CUDA params).
        try:
            _optimizer = torch.optim.AdamW(trainable, fused=True, **adamw_kwargs)
        except (RuntimeError, ValueError) as exc:
            logger.info("Fused AdamW unavailable (%s); using the default path.", exc)
            _optimizer = torch.optim.AdamW(trainable, **adamw_kwargs)
        _scheduler = torch.optim.lr_scheduler.LambdaLR(_optimizer, _lr_lambda)
        logger.info(
            "AdamW optimizer initialised (lr=%.2e, betas=(%.3f, %.3f), wd=%.3g); "
            "cosine warmup over %d step(s), then cosine decay to %.0f%% of peak "
            "across %d epoch(s).",
            config.LEARNING_RATE, config.ADAM_BETA1, config.ADAM_BETA2,
            config.WEIGHT_DECAY, config.NUM_WARMUP_STEPS,
            config.LR_DECAY_FLOOR * 100, config.NUM_EPOCHS,
        )
    return _optimizer


def _step() -> float:
    """Clip accumulated gradients, apply the optimizer step, and reset."""
    global _accum_count
    optimizer = get_optimizer()

    # The optimizer's param groups already hold exactly the trainable params, so
    # reuse them rather than re-filtering all of the model's (120B) parameters by
    # requires_grad on every step.
    trainable = [p for group in optimizer.param_groups for p in group["params"]]
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)

    # Read before scheduler.step(), so the logged LR is the one this update used
    # rather than the one the next update will use.
    step_lr = optimizer.param_groups[0]["lr"]
    optimizer.step()
    optimizer.zero_grad()
    if _scheduler is not None:
        _scheduler.step()
    n_accumulated = _accum_count
    _accum_count = 0

    logger.debug(
        "Optimisation step over %d accumulated micro-batch(es): |grad|=%.4f, lr=%.2e",
        n_accumulated, grad_norm, step_lr,
    )
    return float(grad_norm)


def discard_accumulated(reason: str) -> None:
    """Drop every gradient accumulated so far and reset the window.

    Called when a backward pass RAISES partway through. A half-applied backward
    leaves the grad buffer holding a fraction of one micro-batch's gradient — with
    no way to tell which parameters got their contribution and which didn't — and
    the next _step() would clip and apply that, silently taking an optimizer step
    on a gradient that corresponds to no actual loss. There is no partial undo, so
    the whole window goes: up to GRAD_ACCUM_STEPS-1 micro-batches of work is lost,
    which is the cheap side of the trade against corrupting the update.

    Safe to call before the optimizer exists (nothing has been accumulated yet).
    """
    global _accum_count
    if _optimizer is not None:
        _optimizer.zero_grad(set_to_none=True)
    else:
        model, _ = get_model_and_tokenizer()
        for p in model.parameters():
            p.grad = None
    logger.warning(
        "Discarding %d accumulated micro-batch(es) after %s: a partial backward "
        "leaves an unusable gradient buffer, so the accumulation window is reset.",
        _accum_count, reason,
    )
    _accum_count = 0


def accumulate_loss(kl_loss: torch.Tensor, scale: float = 1.0) -> float:
    """Backprop one loss into the shared grad buffer WITHOUT advancing the window.

    The gradient is scaled by ``scale / GRAD_ACCUM_STEPS``. ``scale`` lets a
    caller split one micro-batch across several backward passes that should
    average rather than sum — e.g. the student-rollout path backprops each of a
    problem's N rollouts with ``scale=1/N`` so the N gradients mean into a single
    per-problem micro-batch. The accumulation counter is untouched here; the
    caller must call commit_micro_batch() once the micro-batch is complete.

    Args:
        kl_loss: Scalar loss tensor with a gradient function.
        scale:   Extra weighting applied before the 1/GRAD_ACCUM_STEPS divide.

    Returns:
        The unscaled loss value (kl_loss.item()).
    """
    if not kl_loss.requires_grad and kl_loss.grad_fn is None:
        raise ValueError(
            "kl_loss has no gradient. Ensure logits_without_hint were computed "
            "outside torch.no_grad() in probing_script."
        )
    try:
        (kl_loss * (scale / config.GRAD_ACCUM_STEPS)).backward()
    except Exception:
        discard_accumulated("a failed backward pass")
        raise
    return kl_loss.item()


def commit_micro_batch() -> Optional[float]:
    """Count one completed micro-batch; step if the window is full.

    Returns the clipped grad norm if an optimizer step was taken this call, else
    None (gradients still accumulating).
    """
    global _accum_count
    _accum_count += 1
    grad_norm: Optional[float] = None
    if _accum_count >= config.GRAD_ACCUM_STEPS:
        grad_norm = _step()
    return grad_norm


def run_optimization(kl_loss: torch.Tensor) -> tuple[float, Optional[float]]:
    """Accumulate one item's KL gradient as a micro-batch; step every GRAD_ACCUM_STEPS.

    The single-item micro-batch path: one call is one micro-batch, weighted
    1/GRAD_ACCUM_STEPS. For the multi-backward (student-rollout) case, use
    accumulate_loss() per rollout followed by one commit_micro_batch() per problem.

    Args:
        kl_loss: Scalar tensor returned by loss_script.run_loss().
                 Must have requires_grad=True / a gradient function.

    Returns:
        (loss_val, grad_norm) — loss_val is the unscaled per-item loss;
        grad_norm is the clipped norm if an optimizer step was taken this
        call, else None (gradients still accumulating).
    """
    loss_val = accumulate_loss(kl_loss)
    grad_norm = commit_micro_batch()

    logger.debug(
        "Accumulated gradient %d/%d: loss=%.6f",
        _accum_count or config.GRAD_ACCUM_STEPS, config.GRAD_ACCUM_STEPS, loss_val,
    )
    return loss_val, grad_norm


def run_optimization_batch(
    kl_losses: list[torch.Tensor],
) -> tuple[float, Optional[float]]:
    """Backprop a real minibatch of per-item KL losses in ONE backward.

    The multi-item analogue of run_optimization: the B item losses are averaged
    into a single micro-batch loss, divided by GRAD_ACCUM_STEPS, and a single
    .backward() walks the shared autograd graph for all B items at once. This
    call counts as ONE accumulation micro-batch (not B), so the effective batch
    is TRAIN_BATCH_SIZE × GRAD_ACCUM_STEPS — the conventional meaning, and a
    superset of the per-item path (where B=1 reduces this to run_optimization's
    1/GRAD_ACCUM_STEPS-per-item weighting). Each item is weighted
    1/(B × GRAD_ACCUM_STEPS).

    Args:
        kl_losses: Non-empty list of scalar loss tensors, each with a grad fn.

    Returns:
        (mean_loss, grad_norm) — mean_loss is the per-item loss averaged over the
        batch; grad_norm is the clipped norm if this call filled the accumulation
        window (GRAD_ACCUM_STEPS micro-batches) and stepped, else None.
    """
    global _accum_count
    if not kl_losses:
        raise ValueError("run_optimization_batch called with an empty loss list.")
    for kl_loss in kl_losses:
        if not kl_loss.requires_grad and kl_loss.grad_fn is None:
            raise ValueError(
                "A kl_loss in the batch has no gradient. Ensure student logits were "
                "computed outside torch.no_grad() in probing_script."
            )

    # Mean over the batch's items, then scale by 1/GRAD_ACCUM_STEPS so that
    # summing GRAD_ACCUM_STEPS micro-batches yields a mean over the full
    # effective batch (TRAIN_BATCH_SIZE × GRAD_ACCUM_STEPS items).
    batch_loss = torch.stack([kl.reshape(()) for kl in kl_losses]).mean()
    try:
        (batch_loss / config.GRAD_ACCUM_STEPS).backward()
    except Exception:
        # An OOM or a device error part-way through the batch's shared graph has
        # written some parameters' gradients and not others; see discard_accumulated.
        discard_accumulated("a failed batched backward pass")
        raise
    _accum_count += 1
    mean_loss = float(batch_loss.item())

    grad_norm: Optional[float] = None
    if _accum_count >= config.GRAD_ACCUM_STEPS:
        grad_norm = _step()

    logger.debug(
        "Accumulated micro-batch of %d item(s) -> %d/%d: mean loss=%.6f",
        len(kl_losses), _accum_count or config.GRAD_ACCUM_STEPS,
        config.GRAD_ACCUM_STEPS, mean_loss,
    )
    return mean_loss, grad_norm


def flush_gradients() -> Optional[float]:
    """Apply any partially accumulated gradients (e.g. at an epoch boundary).

    Returns the clipped grad norm, or None if nothing was pending.
    """
    if _accum_count == 0:
        return None
    logger.info("Flushing %d accumulated gradient(s) before checkpoint.", _accum_count)
    return _step()
