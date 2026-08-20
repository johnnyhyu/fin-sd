#!/usr/bin/env python3
"""opsd training loop — forward-KL context distillation over student rollouts.

Each epoch, over the training problems (served on vLLM):
  1. Run inference on the problem (vLLM).
  2. Grade the answer (OpenRouter). Correct → skip (no update).
  3. Wrong → sample student rollout(s) from the bare problem, and for each pull
     the student toward a teacher conditioned on the problem's reference solution
     via forward KL(teacher ‖ student) over the rollout tokens.
  4. Accumulate/step (AdamW), then resync vLLM at the epoch boundary and grade
     the held-out validation split.

Reuses the main pipeline's HF singleton, optimizer, vLLM lifecycle, inference and
eval; only the objective (opsd.probing + loss_script.run_loss) is opsd-specific.

    python -m opsd.train                                   # GPUs auto-placed
    CUDA_VISIBLE_DEVICES=4,5,6,7 python -m opsd.train  # …or pin them yourself
"""
import os

# Match run_pipeline.py: configure the CUDA allocator before torch is imported
# (pulled in transitively below). expandable_segments reduces the reserved-vs-
# allocated gap between the wide rollout-sampling step and the batch-1 grad
# forwards. setdefault so an explicit override still wins; vllm_server strips it
# from the vLLM subprocess env (vLLM breaks under expandable_segments).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import math
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import wandb

from pipeline import config as pc
from pipeline import (
    eval_script,
    inference_script,
    loss_script,
    optimization_script,
    vllm_server,
)
from pipeline.utils import configure_train_gpus, logger, set_global_seed, setup_logging

from . import config, probing

_CHECKPOINT_BASE = Path(os.getenv("OPSD_CHECKPOINT_DIR", config.OUTPUT_DIR))
RUN_ID = os.getenv("OPSD_RUN_ID") or os.getenv("RUN_ID") \
    or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
CHECKPOINT_DIR = _CHECKPOINT_BASE / RUN_ID
DIVIDER = "─" * 70


# ── Data ─────────────────────────────────────────────────────────────────────

def load_references(path: str) -> list[dict]:
    """Load the reference JSONL, sorted by problem `number` for a stable split.

    Sorting decouples the train/val split from the (concurrent, nondeterministic)
    order opsd.generate_data wrote records in, so the split is reproducible.
    """
    p = Path(path)
    if not p.exists():
        sys.exit(
            f"Reference set not found: {p}. Run `python -m opsd.generate_data` first."
        )
    items: list[dict] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("reference_solution") and rec.get("problem"):
            items.append(rec)
    items.sort(key=lambda r: (r.get("number") is None, r.get("number")))
    return items


def _split(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """Fixed random.Random(42) train/val split (independent of config.SEED).

    Mirrors run_pipeline.py: the split defines *what* val accuracy measures, so it
    stays fixed while SEED is swept. TRAIN_VAL_SPLIT is the train fraction.
    """
    rng = random.Random(42)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    n_train = math.ceil(len(items) * config.TRAIN_VAL_SPLIT)
    train = [items[i] for i in idx[:n_train]]
    val = [items[i] for i in idx[n_train:]]
    return train, val


# ── Per-item training ────────────────────────────────────────────────────────

def _train_item(epoch: int, idx: int, total: int, item: dict) -> str:
    """Process one training problem; return 'skipped' | 'trained' | 'error'."""
    label = f"Epoch {epoch} | Item {idx}/{total} (#{item.get('number')})"
    problem = str(item["problem"])
    ground_truth = str(item.get("answer", ""))
    reference_solution = str(item.get("reference_solution", ""))

    # Step 1-2: inference + grade. Correct → no update.
    try:
        inf = inference_script.run_inference(problem)
    except Exception as exc:
        logger.error("%s: inference failed — %s. Skipping.", label, exc)
        return "error"
    if eval_script.run_eval(inf["generated_answer"], ground_truth):
        logger.info("%s: correct — skipping.", label)
        return "skipped"

    # Step 3: sample student rollout(s) and forward-KL toward the teacher.
    seed = None if config.SEED < 0 else config.SEED + epoch * 1_000_003 + idx
    try:
        sampled = probing.sample_student_rollouts(problem, seed=seed)
    except Exception as exc:
        logger.error("%s: rollout sampling failed — %s. Skipping.", label, exc)
        return "error"

    completions = sampled["completions"]
    n_rollouts = len(completions)
    try:
        rollout_losses = []
        for r_idx, completion_ids in enumerate(completions, 1):
            r = probing.rollout_logits(problem, reference_solution, sampled, completion_ids)
            n_tokens = len(r["completion_tokens"])
            # Forward KL(teacher ‖ student) on the student-sampled tokens.
            kl_loss = loss_script.run_loss(r["logits_with_hint"], r["logits_without_hint"])
            # One problem = one micro-batch; its N rollouts average via scale=1/N.
            loss_val = optimization_script.accumulate_loss(kl_loss, scale=1.0 / n_rollouts)
            rollout_losses.append(loss_val)
            del r, kl_loss
            logger.info("%s: rollout %d/%d forward-KL = %.6f over %d tokens.",
                        label, r_idx, n_rollouts, loss_val, n_tokens)

        grad_norm = optimization_script.commit_micro_batch()
        mean_loss = sum(rollout_losses) / len(rollout_losses)
        if grad_norm is not None:
            logger.info("%s: problem mean forward-KL = %.6f over %d rollouts. "
                        "Weights updated (|grad|=%.4f).", label, mean_loss, n_rollouts, grad_norm)
        metrics = {"loss": mean_loss, "n_rollouts": n_rollouts, "epoch": epoch}
        if grad_norm is not None:
            metrics["grad_norm"] = grad_norm
        if config.USE_WANDB:
            wandb.log(metrics)
    except Exception as exc:
        logger.error("%s: optimization failed — %s. Skipping.", label, exc, exc_info=True)
        return "error"
    return "trained"


def _validate_item(epoch: int, idx: int, n_val: int, item: dict) -> str:
    """One validation item (inference + eval): 'correct' | 'wrong' | 'error'."""
    try:
        inf = inference_script.run_inference(str(item["problem"]))
        ok = eval_script.run_eval(inf["generated_answer"], str(item.get("answer", "")))
        return "correct" if ok else "wrong"
    except Exception as exc:
        logger.warning("Epoch %d | Val %d/%d failed — %s", epoch, idx, n_val, exc)
        return "error"


# ── vLLM resync (mirrors run_pipeline.py's epoch boundary) ───────────────────

def _resync_vllm(epoch: int, adapter_dir: str, prev_merged_dir: Optional[str],
                 prev_adapter_name: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Refresh the served weights with this epoch's training.

    Expert LoRA (config.LORA_TARGET_EXPERTS): merge into a full checkpoint and
    restart vLLM on it (fused-MoE experts can't take a hot-swapped adapter — see
    memory on the expert-LoRA merge+restart). Otherwise hot-load the adapter.

    The merged dir is a complete, standalone checkpoint (weights + config +
    generation_config + tokenizer), so any epoch it survives for can be re-served
    later with `tools/serve_checkpoint.py --source opsd --epoch <N>`. Once vLLM is
    up on this epoch's copy the previous one is dead weight and is dropped, so
    disk doesn't grow by a full model every epoch — unless KEEP_EPOCH_MERGED.
    """
    if config.LORA_TARGET_EXPERTS:
        merged_dir = optimization_script.save_merged_model(CHECKPOINT_DIR / f"epoch_{epoch}_merged")
        vllm_server.restart(merged_dir)
        if prev_merged_dir is not None and not config.KEEP_EPOCH_MERGED:
            shutil.rmtree(prev_merged_dir, ignore_errors=True)
        return merged_dir, None

    adapter_name = f"epoch_{epoch}"
    vllm_server.load_adapter(adapter_name, adapter_dir)
    inference_script.set_active_model(adapter_name)
    if prev_adapter_name is not None:
        vllm_server.unload_adapter(prev_adapter_name)
    return None, adapter_name


# ── Main ─────────────────────────────────────────────────────────────────────

def train(on_checkpoint=lambda: None) -> None:
    setup_logging()
    # Place + reserve this run's GPUs before anything initialises CUDA — the same
    # autoselect launch.py gives run_pipeline.py. Without it the HF student's
    # device_map="auto" spreads across every visible card INCLUDING the two the vLLM
    # server below is about to claim, and the load OOMs against a server holding 90%
    # of their VRAM (the `CUDA_VISIBLE_DEVICES=2,…,7` prefix in this module's usage
    # line was the only thing standing between a bare `python -m opsd.train` and
    # that). An explicit pin still wins; it is just reserved so siblings skip it.
    pc.HF_MAX_MEMORY = configure_train_gpus(
        pc.HF_MODEL_PATH,
        needs_vllm=True,                 # opsd always serves the student on vLLM
        max_memory_env="HF_MAX_MEMORY",  # where a per-card cap pin would come from
        full_finetune=False,             # opsd trains through the pipeline's LoRA path
    )
    set_global_seed(config.SEED)
    logger.info("opsd run %s: checkpoints → %s", RUN_ID, CHECKPOINT_DIR)

    items = load_references(config.REFERENCE_PATH)
    if not items:
        sys.exit(f"No usable references in {config.REFERENCE_PATH}.")
    train_items, val_items = _split(items)
    total, n_val = len(train_items), len(val_items)
    logger.info("Split: %d train / %d val from %d references (%s).",
                total, n_val, len(items), config.REFERENCE_PATH)

    # Bring up the pipeline-owned vLLM server on the base student before anything
    # talks to it (epoch boundaries then refresh it with the trained weights).
    prev_merged_dir: Optional[str] = None
    prev_adapter_name: Optional[str] = None
    # Not a bare atexit hook: SIGTERM (`kill`, a scheduler) and SIGHUP (closing the
    # terminal) terminate the process without running atexit at all, orphaning the
    # server with every GPU it holds and the port still bound. install_shutdown_handlers
    # converts those into a clean exit that does run stop() (and registers the same
    # atexit hook), exactly as run_pipeline.py does.
    vllm_server.install_shutdown_handlers()
    vllm_server.start(pc.VLLM_MODEL)

    if config.USE_WANDB:
        wandb.init(
            entity=config.WANDB_ENTITY or None,
            project=config.WANDB_PROJECT,
            name=config.WANDB_RUN_NAME or None,
            id=RUN_ID, resume="allow",
            config={
                "objective": "opsd-forward-kl-context-distill",
                "model": pc.VLLM_MODEL,
                "hf_model_path": pc.HF_MODEL_PATH,
                "hint_model": config.HINT_MODEL,
                "learning_rate": config.LEARNING_RATE,
                "grad_accum_steps": config.GRAD_ACCUM_STEPS,
                "num_epochs": config.NUM_EPOCHS,
                "num_rollouts": config.NUM_ROLLOUTS,
                "max_rollout_tokens": config.MAX_ROLLOUT_TOKENS,
                "rollout_temperature": config.ROLLOUT_TEMPERATURE,
                "rollout_top_p": config.ROLLOUT_TOP_P,
                "lora_target_experts": config.LORA_TARGET_EXPERTS,
                "train_val_split": config.TRAIN_VAL_SPLIT,
                "total_items": len(items),
            },
        )

    counts = {"skipped": 0, "trained": 0, "error": 0}
    for epoch in range(1, config.NUM_EPOCHS + 1):
        logger.info(DIVIDER)
        logger.info("Epoch %d/%d", epoch, config.NUM_EPOCHS)
        counts = {"skipped": 0, "trained": 0, "error": 0}
        # Drive the shared LR schedule's epoch-indexed decay (the total optimizer
        # step count is unknown up front; NUM_EPOCHS is). Without this the run
        # would train at the epoch-1 LR throughout.
        optimization_script.set_epoch(epoch)

        # Serial: each item does an inline HF forward+backward, so processing must
        # not overlap on the GPU (same constraint as the pipeline's rollout path).
        for idx, item in enumerate(train_items, 1):
            counts[_train_item(epoch, idx, total, item)] += 1

        train_acc = counts["skipped"] / total if total else 0.0
        logger.info(DIVIDER)
        logger.info("Epoch %d/%d done. Correct/skipped=%d | Trained=%d | Errors=%d | "
                    "Train acc=%.4f", epoch, config.NUM_EPOCHS, counts["skipped"],
                    counts["trained"], counts["error"], train_acc)

        # ── Checkpoint + resync BEFORE validation so val (and next epoch) see the
        # weights just trained this epoch. ──────────────────────────────────────
        if probing.is_loaded():
            optimization_script.flush_gradients()
            epoch_dir = CHECKPOINT_DIR / f"epoch_{epoch}"
            adapter_dir = optimization_script.save_adapter(epoch_dir)
            optimization_script.save_optimizer_state(epoch_dir)
            prev_merged_dir, prev_adapter_name = _resync_vllm(
                epoch, adapter_dir, prev_merged_dir, prev_adapter_name
            )
            (epoch_dir / "DONE").write_text(RUN_ID)
            on_checkpoint()
        else:
            logger.info("Epoch %d: no training occurred; skipping checkpoint.", epoch)

        # ── Validation on the freshly-served weights ─────────────────────────
        val_acc = 0.0
        val_correct = val_errors = 0
        if n_val > 0:
            logger.info("Epoch %d: validating on %d items …", epoch, n_val)
            width = max(1, config.MAX_CONCURRENT_ITEMS)
            with ThreadPoolExecutor(max_workers=width) as ex:
                for outcome in ex.map(lambda p: _validate_item(epoch, p[0], n_val, p[1]),
                                      enumerate(val_items, 1)):
                    val_correct += int(outcome == "correct")
                    val_errors += int(outcome == "error")
            val_acc = val_correct / n_val
            logger.info("Epoch %d validation: %d/%d correct | Errors=%d | Val acc=%.4f",
                        epoch, val_correct, n_val, val_errors, val_acc)

        if config.USE_WANDB:
            wandb.log({
                "epoch": epoch,
                "epoch_skipped": counts["skipped"],
                "epoch_trained": counts["trained"],
                "epoch_errors": counts["error"],
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "val_correct": val_correct,
            })

    logger.info(DIVIDER)
    if probing.is_loaded():
        final_dir = optimization_script.save_adapter(CHECKPOINT_DIR / "final")
        final_merged = optimization_script.save_merged_model(CHECKPOINT_DIR / "final_merged")
        # final_merged supersedes the last epoch's merged copy (same weights); drop it.
        if prev_merged_dir is not None and not config.KEEP_EPOCH_MERGED:
            shutil.rmtree(prev_merged_dir, ignore_errors=True)
        logger.info("Final adapter → %s (merged → %s)", final_dir, final_merged)
        on_checkpoint()
        if config.USE_WANDB:
            wandb.summary.update({"final_adapter": final_dir, "final_merged": final_merged})
    else:
        logger.info("No training occurred in any epoch; no adapter to save.")

    if config.USE_WANDB:
        wandb.finish()
    logger.info("opsd training complete (%d epochs).", config.NUM_EPOCHS)


def main() -> None:
    train()


if __name__ == "__main__":
    main()
