"""Standard SFT training.

Fine-tunes the student on the teacher-distilled (problem → solution) pairs with
Hugging Face `Trainer` and a plain next-token loss over the full sequence. LoRA
is on by default (see `sft.config`).

Checkpoints follow the main pipeline's layout, under `SFT_OUTPUT_DIR/<RUN_ID>/`:

    epoch_<N>/          this epoch's adapter (or, without LoRA, the full model)
    epoch_<N>_merged/   the same weights merged into a standalone full checkpoint
    final/              the trained adapter (or full model) at end of training
    final_merged/       merged copy of the final weights (SFT_MERGE_AFTER_TRAIN=1)

A new pair is written every epoch, and each `epoch_<N>_merged` is a complete
checkpoint vLLM can serve as-is — so any epoch of any run can be brought up for
benchmarking exactly like a pipeline epoch::

    python tools/serve_checkpoint.py --source sft --run-id 20260730-1200 --epoch 2

Older merged copies are kept by default (`SFT_KEEP_EPOCH_MERGED=0` prunes them
like the pipeline does, leaving only the newest).

A fraction of the distilled records (`SFT_VAL_SPLIT`) is held out and never
trained on. At each epoch boundary those problems are answered by the LIVE
student — `model.generate` on the in-memory weights — and graded for answer
accuracy (see `sft.validate`, which reuses the serving prompt, harmony stop
tokens, parse and grader so the number still means what a served checkpoint
would score). No vLLM is involved, so validation costs no extra GPU. With
`--serve` (or `SFT_SERVE_AFTER_TRAIN=1`) a vLLM server is left up on the final
checkpoint once training finishes; that is now the only thing that starts one.

    python -m sft.train                 # train on sft/data/distill.jsonl
    python -m sft.train --data foo.jsonl --output-dir runs/exp1
    python -m sft.train --serve         # leave vLLM serving the final checkpoint
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path
from typing import Callable, Optional

from pipeline import config as pc
from pipeline.utils import logger, set_global_seed, setup_logging

from . import config
from .dataset import SFTCollator, SFTDataset, load_records, split_records
from .model import build_model_and_tokenizer
from .validate import validate


def train(data_path: str | None = None, output_dir: str | None = None,
          serve: bool = False, on_checkpoint: Callable[[], None] = lambda: None) -> str:
    """Run the SFT loop; return the final adapter (or full-model) checkpoint dir.

    `on_checkpoint` is called after each epoch's checkpoint — and the final one —
    is written. It defaults to a no-op for local runs; `sft/modal_app.py` passes
    the checkpoints Volume's `.commit` so per-epoch artifacts become durable
    mid-run rather than only at container exit (mirrors `run_pipeline.main`).
    """
    from transformers import Trainer, TrainingArguments

    data_path = data_path or config.DISTILL_PATH
    output_dir = output_dir or config.OUTPUT_DIR
    # Per-run subdir, exactly like run_pipeline's CHECKPOINT_DIR: --output-dir (or
    # SFT_OUTPUT_DIR) is the BASE holding one dir per run, and everything this run
    # writes — Trainer's own step checkpoints included — lands inside it.
    run_dir = Path(output_dir) / config.RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("SFT run %s: checkpoints → %s", config.RUN_ID, run_dir)

    # Seed Python/NumPy/torch before the model is built: a fresh LoRA's A matrices
    # are drawn at attach time, which is *before* `Trainer` calls `set_seed` on
    # config.SEED, so without this the initialization is nondeterministic. Uses the
    # pipeline's SEED so an SFT student starts from the same init as a pipeline run
    # (pc.SEED < 0 disables seeding entirely).
    set_global_seed(pc.SEED)

    model, tokenizer = build_model_and_tokenizer()
    train_records, val_records = split_records(
        load_records(data_path), config.VAL_SPLIT, config.SEED, config.VAL_MAX_EXAMPLES
    )
    dataset = SFTDataset(train_records, tokenizer, config.MAX_SEQ_LEN, config.MASK_PROMPT)
    collator = SFTCollator(pad_token_id=tokenizer.pad_token_id)
    if val_records:
        logger.info(
            "Holding out %d record(s) for end-of-epoch in-process validation.",
            len(val_records),
        )

    if config.USE_WANDB:
        # Set regardless of whether a run NAME was configured: without it the
        # Trainer's wandb integration logs to its own default project ("huggingface")
        # instead of this repo's.
        os.environ.setdefault("WANDB_PROJECT", config.WANDB_PROJECT)

    # Every exit path must tear the serving vLLM down — including the
    # SIGTERM/SIGHUP that run neither `finally` blocks nor atexit hooks. Without
    # this an interrupted run orphans a server that keeps a whole GPU and the port
    # (which is what left port 8000 bound and killed the 20260730-234617 run).
    # Only --serve starts one now; validation decodes in-process.
    if serve:
        from pipeline import vllm_server

        vllm_server.install_shutdown_handlers()

    save_strategy = "steps" if config.SAVE_STEPS > 0 else "epoch"
    args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type=config.LR_SCHEDULER,
        warmup_ratio=config.WARMUP_RATIO,
        weight_decay=config.WEIGHT_DECAY,
        logging_steps=config.LOGGING_STEPS,
        save_strategy=save_strategy,
        save_steps=config.SAVE_STEPS or 500,
        bf16=config.HF_DTYPE == "bfloat16",
        fp16=config.HF_DTYPE == "float16",
        gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
        # gpt-oss is MoE: bf16 router matmuls dispatch a slightly different
        # token count per expert on the checkpoint recompute, which trips
        # non-reentrant checkpointing's tensor-metadata consistency check.
        # Reentrant skips that check (matching the main RL/probing pipeline).
        gradient_checkpointing_kwargs={"use_reentrant": True},
        seed=config.SEED,
        # Our dataset yields {"input_ids": [...], "labels": [...]}; keep it verbatim
        # (Trainer would otherwise drop columns its model signature doesn't name).
        remove_unused_columns=False,
        report_to=["wandb"] if config.USE_WANDB else [],
        run_name=config.WANDB_RUN_NAME or None,
        optim="adamw_torch",
    )

    # Always attached: it writes this run's per-epoch checkpoint (and, when there
    # is a held-out split, serves and grades it). `epoch_cb.last_merged` is the
    # newest merged checkpoint once training returns.
    epoch_cb = _EpochCheckpointCallback(model, tokenizer, run_dir, val_records, on_checkpoint)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[epoch_cb],
    )

    logger.info(
        "Starting SFT: %d examples, %d epochs, effective batch = %d (bs=%d × grad_accum=%d).",
        len(dataset), config.NUM_EPOCHS,
        config.BATCH_SIZE * config.GRAD_ACCUM_STEPS, config.BATCH_SIZE, config.GRAD_ACCUM_STEPS,
    )
    trainer.train()

    adapter_dir = str(run_dir / "final")
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("Saved %s to %s.", "adapter" if config.USE_LORA else "model", adapter_dir)

    if config.USE_LORA and config.MERGE_AFTER_TRAIN:
        final_merged = _merge(model, tokenizer, run_dir / "final_merged")
        # final_merged holds the same weights as the last epoch's merged copy,
        # which is therefore superseded (the pipeline drops it here too).
        _prune_superseded(epoch_cb.last_merged)
        epoch_cb.last_merged = final_merged
    on_checkpoint()

    if serve:
        _serve(_servable_checkpoint(model, tokenizer, run_dir, adapter_dir,
                                    epoch_cb.last_merged))

    return adapter_dir


def _merge(model, tokenizer, merged_dir: Path) -> str:
    """Merge the live LoRA weights into a standalone full checkpoint at `merged_dir`.

    Delegates to the pipeline's own merge (optimization_script.save_merged_model)
    rather than PEFT's merge_and_unload, for two reasons: it is REVERSIBLE
    (merge_adapter/unmerge_adapter), so it can run mid-training at an epoch
    boundary and leave `Trainer` training on the same adapter afterwards; and it
    writes generation_config.json, which gpt-oss keeps its harmony stop tokens in
    and without which vLLM over-generates past tool-call boundaries (it is also
    what tools/serve_checkpoint.py checks before serving a merged dir as-is).
    Without LoRA there is nothing to merge and it writes the live weights straight
    out (save_full_model).
    """
    from pipeline import optimization_script

    logger.info("Merging weights into a full checkpoint → %s …", merged_dir)
    return optimization_script.save_merged_model(merged_dir, model, tokenizer)


def _prune_superseded(merged_dir: Optional[str]) -> None:
    """Drop a merged checkpoint that a newer one supersedes, unless we keep them all.

    Mirrors run_pipeline's epoch boundary, which deletes the previous epoch's
    merged dir so disk doesn't grow by a full model every epoch. Here it is
    skipped by default (KEEP_EPOCH_MERGED=1): an SFT adapter can't be re-merged
    after the fact as reliably as a pipeline one — serve_checkpoint rebuilds the
    PEFT tree from `pipeline.config`, whose expert rank need not match this run's
    — so every epoch keeps its own servable copy.
    """
    if merged_dir is None or config.KEEP_EPOCH_MERGED:
        return
    shutil.rmtree(merged_dir, ignore_errors=True)
    logger.info("Removed superseded merged checkpoint %s (SFT_KEEP_EPOCH_MERGED=0).",
                merged_dir)


def _servable_checkpoint(model, tokenizer, run_dir: Path, adapter_dir: str,
                         last_merged: Optional[str]) -> str:
    """Resolve a full, standalone checkpoint dir that vLLM can serve.

    vLLM serves a base-model path; there is no adapter to hot-load in the SFT
    flow, so a LoRA run needs its weights merged into a full checkpoint first.
    Prefer the newest merged checkpoint this run wrote (`final_merged`, or the
    last epoch's — which holds the same final weights); a non-LoRA `final` is
    itself a full model; otherwise merge the adapter now.
    """
    if last_merged and Path(last_merged).exists():
        return last_merged
    if not config.USE_LORA:
        return adapter_dir
    return _merge(model, tokenizer, run_dir / "final_merged")


def _serve(checkpoint_dir: str) -> None:
    """Spin up the pipeline-owned vLLM server on `checkpoint_dir` and block.

    Reuses pipeline.vllm_server (its own process group, on config.VLLM_GPUS which
    are kept disjoint from the HF training GPUs). Blocks until interrupted so the
    server stays up, then tears it down cleanly on Ctrl-C.
    """
    from pipeline import config as pc
    from pipeline import vllm_server

    logger.info("Spinning up vLLM server on final checkpoint %s …", checkpoint_dir)
    # restart() (not start()) so this is robust to a per-epoch validation server
    # still holding the port from the last epoch.
    vllm_server.restart(checkpoint_dir)
    logger.info(
        "vLLM server ready at %s (served model name: %s). Press Ctrl-C to stop.",
        pc.VLLM_BASE_URL, pc.VLLM_MODEL,
    )
    try:
        # Poll is_alive() rather than sleeping blind (what tools/serve_checkpoint.py
        # does): an engine that dies under us — OOM, a killed TP worker — would
        # otherwise leave this parked for an hour at a time holding the GPU
        # reservation, with nothing served.
        while vllm_server.is_alive():
            time.sleep(5)
        logger.error("vLLM server exited on its own; shutting down.")
    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down vLLM server.")
    finally:
        vllm_server.stop()


def _save_epoch_checkpoint(model, tokenizer, run_dir: Path, epoch: int) -> tuple[Path, str]:
    """Write this epoch's checkpoint; return (epoch_dir, servable merged dir).

    Mirrors run_pipeline's epoch boundary so the two layouts are interchangeable
    to tools/serve_checkpoint.py:

      * LoRA — `epoch_<N>/` holds the adapter (+ tokenizer) and
        `epoch_<N>_merged/` the same weights merged into a standalone full
        checkpoint. The merge is reversible, so training continues unaffected.
      * no LoRA — the live weights ARE the checkpoint, so `epoch_<N>/` is itself a
        stock HF checkpoint and doubles as the servable dir (no `_merged` twin).

    The `DONE` sentinel is written LAST, so a dir without it was interrupted
    mid-write and should not be trusted (same contract as the pipeline's).
    """
    epoch_dir = run_dir / f"epoch_{epoch}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    if config.USE_LORA:
        model.save_pretrained(str(epoch_dir))       # adapter weights + adapter_config
        tokenizer.save_pretrained(str(epoch_dir))
        merged = _merge(model, tokenizer, run_dir / f"epoch_{epoch}_merged")
    else:
        merged = _merge(model, tokenizer, epoch_dir)
    (epoch_dir / "DONE").write_text(config.RUN_ID)
    logger.info("Epoch %d checkpoint: %s (servable: %s).", epoch, epoch_dir, merged)
    return epoch_dir, merged


def _EpochCheckpointCallback(model, tokenizer, run_dir: Path, val_records: list[dict],
                             on_checkpoint: Callable[[], None]):
    """Build the TrainerCallback that checkpoints (and optionally validates) each epoch.

    Every epoch boundary writes a fresh `epoch_<N>` / `epoch_<N>_merged` pair, then
    — when a held-out split exists — grades the live weights on the held-out set
    (`sft.validate`, in-process; the merged copy is written for later serving, not
    for validation). The previous epoch's merged copy is pruned only AFTER the new
    one is written (and only when SFT_KEEP_EPOCH_MERGED=0), so a failure never
    leaves the run without a servable checkpoint.

    Built by a factory rather than declared at module level so `transformers` is
    only imported when training actually runs. `.last_merged` is read back by
    `train()` for the final merge / `--serve`.
    """
    from transformers import TrainerCallback

    class _Cb(TrainerCallback):
        last_merged: Optional[str] = None

        def on_epoch_end(self, args, state, control, **kwargs):
            epoch = int(round(state.epoch or 0))
            prev_merged = self.last_merged
            _, merged = _save_epoch_checkpoint(model, tokenizer, run_dir, epoch)
            self.last_merged = merged
            if val_records:
                # After the checkpoint, so a validation failure still leaves this
                # epoch on disk. The merge above is reversible and already undone,
                # so these are the same weights it just wrote.
                acc = validate(model, tokenizer, val_records, epoch)
                if config.USE_WANDB:
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.log({"val/accuracy": acc}, step=state.global_step)
                    except Exception as exc:
                        logger.warning("[val] wandb log failed: %s", exc)
            _prune_superseded(prev_merged)
            # Persist this epoch's checkpoint to durable storage (no-op locally; on
            # Modal this commits the checkpoints Volume).
            on_checkpoint()

    return _Cb()


def main() -> None:
    ap = argparse.ArgumentParser(description="Standard SFT on teacher-distilled data.")
    ap.add_argument("--data", default=config.DISTILL_PATH)
    ap.add_argument("--output-dir", default=config.OUTPUT_DIR,
                    help="Base dir holding the per-run checkpoint dirs; this run "
                         f"writes <output-dir>/{config.RUN_ID}/ (set SFT_RUN_ID to pin it).")
    ap.add_argument("--serve", action="store_true",
                    help="Spin up a vLLM server on the final checkpoint after training.")
    args = ap.parse_args()

    # USE_MODAL=1 dispatches training to a Modal GPU container instead of local
    # GPUs (mirrors the root launch.py). This is train-only (skip the generate
    # stage), reusing the distilled data already on the checkpoints Volume.
    # Config — including --data / --output-dir equivalents (SFT_DISTILL_PATH /
    # SFT_OUTPUT_DIR) — must travel via .env, which is what gets forwarded.
    from pipeline import config as pc
    if pc.USE_MODAL:
        if args.serve:
            logger.warning(
                "--serve is ignored on the Modal path: the detached container is "
                "ephemeral, so there is no persistent local vLLM endpoint to hold open."
            )
        from ._modal_dispatch import dispatch
        raise SystemExit(dispatch(skip_generate=True, skip_train=False))

    setup_logging()
    serve = args.serve or config.SERVE_AFTER_TRAIN
    # Place + reserve this run's GPUs before train() loads the student and
    # initialises CUDA. Only --serve needs a vLLM card; without it the 20B path
    # reserves just the two cards the student shards over.
    from .gpus import configure_training_gpus
    configure_training_gpus(needs_vllm=serve)
    train(args.data, args.output_dir, serve=serve)


if __name__ == "__main__":
    main()
