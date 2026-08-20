#!/usr/bin/env python3
"""Serve one expert-LoRA epoch checkpoint on vLLM (for benchmarking).

The expert-path companion to ``serve_epoch.py``. gpt-oss packs its MoE experts
as fused 3-D params, so an expert-LoRA adapter (``LORA_TARGET_EXPERTS=1``) has no
per-expert module for vLLM to hot-load onto — ``serve_epoch.py`` refuses those and
points here. The pipeline serves them instead by merging the adapter into a full
checkpoint and restarting vLLM on it (``run_pipeline.main``'s epoch boundary).

This does exactly that, in one command:

    1. resolve — ``epoch_<N>`` under the run dir (``--run-id``, or the latest run);
    2. merge   — replay the pipeline's merge (``load_adapter_weights`` +
                 ``save_merged_model``) into ``epoch_<N>_merged``, skipped if that
                 dir already exists (``--force`` re-does it). If the resolved dir
                 is already a merged/full checkpoint, it's served as-is. Runs in a
                 SUBPROCESS: the merge and the server share cards, and only process
                 exit reliably returns the merge's VRAM (see
                 ``_merge_out_of_process``);
    3. serve   — ``vllm_server.start`` launches ``vllm serve`` on ``config.VLLM_GPUS``
                 (disjoint from any HF process) under served name ``config.VLLM_MODEL``,
                 then blocks until Ctrl-C.

Run with the SAME config as the training run — in particular
``LORA_TARGET_EXPERTS=1`` — so the merge rebuilds the matching PEFT tree. The
merge loads the BF16 base model in a subprocess, so every byte of its VRAM is back
before vLLM starts.

The SFT (`sft/`) and opsd (`opsd/`) trainers write the same
``<base>/<run-id>/epoch_<N>[_merged]`` layout, so ``--source`` is all it takes to
serve one of their epochs instead of a pipeline one — the flag only switches which
base dir holds the per-run dirs. SFT runs write their own ``epoch_<N>_merged``
every epoch, which is served as-is; that matters because the merge in step 2
rebuilds the PEFT tree from ``pipeline.config`` (LORA_RANK / LORA_EXPERT_RANK),
whose expert rank need not match an SFT run's, so re-merging a bare SFT adapter is
best-effort. Keep ``SFT_KEEP_EPOCH_MERGED=1`` (the default) and there is nothing
to re-merge.

Examples::

    # By epoch number, latest run under CHECKPOINT_DIR:
    python tools/serve_checkpoint.py --epoch 4

    # Pin the run:
    python tools/serve_checkpoint.py --run-id 20260717-1330 --epoch 4

    # An SFT (or opsd) run's epoch, from that trainer's own checkpoint base:
    python tools/serve_checkpoint.py --source sft --run-id 20260730-1200 --epoch 2

    # By explicit adapter (or already-merged) dir:
    python tools/serve_checkpoint.py --adapter checkpoints/20260717-1330/epoch_4

    # Re-merge even if epoch_4_merged already exists:
    python tools/serve_checkpoint.py --epoch 4 --force

    # Merge + resolve only, don't serve:
    python tools/serve_checkpoint.py --epoch 4 --no-serve
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# This tool lives in tools/; put the repo root on sys.path so `pipeline`
# imports resolve when run as `python tools/serve_checkpoint.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import config
from pipeline import utils
from pipeline.utils import logger, setup_logging

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Which trainer's checkpoint base --source selects. Each entry is the env vars that
# override it (first one set wins) and the repo-relative default, kept in step with
# run_pipeline._CHECKPOINT_BASE, sft.config.OUTPUT_DIR and opsd.train._CHECKPOINT_BASE.
# Read from the environment rather than by importing those modules: `opsd.config`
# writes back onto pipeline.config at import (optimizer knobs), which has no place
# in a serving tool.
_SOURCE_BASES: dict[str, tuple[tuple[str, ...], Path]] = {
    "pipeline": (("CHECKPOINT_DIR",), _REPO_ROOT / "checkpoints"),
    "sft": (("SFT_OUTPUT_DIR",), _REPO_ROOT / "sft" / "checkpoints"),
    "opsd": (("OPSD_CHECKPOINT_DIR", "OPSD_OUTPUT_DIR"), _REPO_ROOT / "opsd" / "checkpoints"),
}

# Set on the merge subprocess's environment by _merge_out_of_process. That child
# inherits the cards the parent already reserved and made visible, so it must NOT
# redo the reservation: reading a fresh config it would see config.VLLM_GPUS back at
# its default, reserve the wrong cards, and warn about the ones its own parent
# legitimately holds.
_MERGE_CHILD_ENV = "SERVE_CHECKPOINT_MERGE_CHILD"


def _pin_gpus() -> None:
    """Reserve + confine this instance's GPUs so concurrent serves don't collide.

    Delegates to utils.configure_serve_gpus: when GPUs aren't pinned it auto-selects
    idle cards (the same reserved idle-GPU finder launch.py uses) instead of the
    fixed VLLM_GPUS default, and pins CUDA_VISIBLE_DEVICES so the merge's
    device_map="auto" stays on the serving cards rather than spreading across every
    visible GPU. Reusing the serving cards for the merge is safe because the merge
    runs in a subprocess that exits before vLLM starts (_merge_out_of_process) —
    in-process it left GiB behind. Must run before anything initialises CUDA.

    The reservation is sized by the MERGE, not by the serving tensor-parallel degree
    (utils.hf_merge_gpu_count): the merge child loads the full BF16 base onto the
    cards reserved here, and while the 20B (~39 GiB) fits on the two cards
    VLLM_TENSOR_PARALLEL defaults to,
    the 120B (~240 GiB) does not and used to OOM at weight load — long before vLLM
    was ever reached. vLLM still serves on VLLM_TENSOR_PARALLEL of those cards.
    """
    utils.configure_serve_gpus(set_visible=True, min_gpus=utils.hf_merge_gpu_count())


def _source_base(source: str) -> Path:
    """Base dir holding `source`'s per-run checkpoint dirs (env override, else default)."""
    env_names, default = _SOURCE_BASES[source]
    for name in env_names:
        value = os.getenv(name)
        if value:
            return Path(value)
    return default


def _resolve_run_dir(base: Path, run_id: str | None) -> Path:
    """Return the run dir under `base` for `--run-id`, or the most recent one."""
    if run_id:
        run_dir = base / run_id
        if not run_dir.is_dir():
            sys.exit(f"Run dir does not exist: {run_dir}")
        return run_dir

    runs = sorted(p for p in base.glob("*") if p.is_dir())
    if not runs:
        sys.exit(f"No run dirs under {base}; nothing to serve.")
    # RUN_IDs default to UTC timestamps, so the lexically-largest is the newest.
    run_dir = runs[-1]
    if len(runs) > 1:
        logger.info(
            "No --run-id; using most recent of %d runs: %s (others: %s)",
            len(runs), run_dir.name, ", ".join(r.name for r in runs[:-1]),
        )
    return run_dir


def _resolve_checkpoint(run_dir: Path, epoch: int | None, adapter: str | None) -> Path:
    """Return the checkpoint dir for `adapter` (a direct path) or `--epoch`."""
    if adapter:
        path = Path(adapter)
        if not path.is_dir():
            sys.exit(f"Checkpoint dir does not exist: {path}")
        return path

    ckpt = run_dir / f"epoch_{epoch}"
    if ckpt.is_dir():
        logger.info("Epoch %d → %s", epoch, ckpt)
        return ckpt

    avail = ", ".join(
        sorted(p.name for p in run_dir.glob("epoch_*") if not p.name.endswith("_merged"))
    ) or "none"
    sys.exit(f"No epoch_{epoch} under {run_dir}. Available epoch dirs: {avail}")


def _is_merged(d: Path) -> bool:
    """A merged/full checkpoint has model weights + config; an adapter dir has neither.

    generation_config.json is required too: gpt-oss's harmony stop tokens live there
    (not in config.json), and a checkpoint missing it makes vLLM over-generate past
    tool-call boundaries until its harmony parser crashes. Stale merged dirs from
    before that file was written are thus treated as incomplete and re-merged.
    """
    return (d / "config.json").exists() and (d / "generation_config.json").exists() \
        and any(d.glob("*.safetensors")) and not (d / "adapter_config.json").exists()


def _merge(adapter_dir: Path, out_dir: Path, force: bool) -> Path:
    """Rebuild the merged full checkpoint from a saved adapter, then free the HF model.

    Replays the pipeline's own epoch-boundary merge: materialise the base model
    with a fresh expert-LoRA adapter, overwrite it with the saved weights
    (optimization_script.load_adapter_weights), then merge + write a clean
    checkpoint (optimization_script.save_merged_model). Idempotent — reuses an
    existing merged dir unless `force`.

    Normally reached only inside the merge subprocess (--merge-in-process), whose
    exit is what actually returns the VRAM. The drop-and-empty_cache below is a
    best-effort courtesy for that case and for a direct in-process invocation; it
    does NOT fully free the cards (measured: 18.4 GiB left reserved on the second
    card), which is precisely why the subprocess exists.
    """
    if out_dir.exists() and not force:
        logger.info("Reusing existing merged checkpoint at %s (use --force to re-merge).", out_dir)
        return out_dir
    # The merge rebuilds the PEFT tree on config.HF_MODEL_PATH, so an adapter from a
    # differently-sized run cannot be loaded onto it. Flag it before we spend the
    # minutes it takes to materialise a base model that can't accept the weights.
    utils.check_adapter_matches_base(adapter_dir, config.HF_MODEL_PATH, "HF_MODEL_PATH")
    # Same idea one level down: the rebuilt PEFT tree has to have the adapter's
    # ranks, which an sft/opsd run need not share (see the module docstring).
    utils.check_adapter_matches_lora_config(adapter_dir)

    # Serialise the merge across concurrent serve_checkpoint runs on this box: two
    # processes writing the same epoch_<N>_merged dir at once would interleave
    # save_merged_model's writes and corrupt the checkpoint. The lock is keyed to
    # the resolved output path (utils.file_lock hashes names too long to be a
    # filename); a second process blocks here, then re-checks below and reuses the
    # finished dir instead of re-merging.
    lock_name = "merge_" + "".join(c if c.isalnum() else "_" for c in str(out_dir.resolve()))
    with utils.file_lock(lock_name):
        if out_dir.exists() and not force:
            logger.info("Merged checkpoint at %s was produced by a concurrent run; reusing it.",
                        out_dir)
            return out_dir

        import gc
        import torch
        from pipeline import probing_script, optimization_script

        logger.info("Merging %s → %s …", adapter_dir, out_dir)
        probing_script.get_model_and_tokenizer()            # base + fresh expert LoRA
        optimization_script.load_adapter_weights(adapter_dir)    # overwrite with trained weights
        merged = optimization_script.save_merged_model(out_dir)  # merge + write clean checkpoint

        # Drop the merge model so its VRAM is reclaimed before vLLM starts.
        probing_script._model = None
        probing_script._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return Path(merged)


def _merge_out_of_process(adapter_dir: Path, out_dir: Path, force: bool) -> Path:
    """Do the merge in a CHILD process, so its VRAM is gone before vLLM starts.

    The merge and the server deliberately share the same cards (see _pin_gpus),
    which only works if the merge's ~20 GiB/card is genuinely back by the time
    `vllm serve` measures free memory. Doing it in-process does not achieve that.
    Measured on this box (torch 2.11, 20B BF16 merge across two cards): after
    dropping the model, gc.collect() and torch.cuda.empty_cache() — what _merge
    does — the first card came back to its 0.5 GiB context, but the second was left
    with 18.4 GiB still RESERVED against 0.01 GiB allocated. A handful of MiB that
    outlive the model (CUDA/cuBLAS workspaces and friends) sit scattered through the
    big segments the merge allocated, and the caching allocator can only return a
    segment that is entirely free — so per-device empty_cache, torch.cuda.synchronize
    and repeated gc passes all left those 18.4 GiB exactly where they were.

    vLLM then died at init with "Free memory on device cuda:1 (58.75/79.25 GiB) on
    startup is less than desired GPU memory utilization (0.9, 71.33 GiB)" — after
    the merge had already burned minutes, and looking for all the world like the
    previous server had leaked its memory.

    Process exit is the one teardown that returns the allocator's cache and the CUDA
    context unconditionally, so the merge gets its own process — the same reason
    launch._ensure_expert_profile runs the expert profiler out of process. The parent
    then never initialises CUDA at all and hands vLLM untouched cards.

    Re-invokes this same tool with --merge-in-process --no-serve so the merge itself
    (and its cross-process lock) stays in exactly one place. stdout/stderr are
    inherited, so the operator sees the merge's own progress and errors inline.
    """
    if out_dir.exists() and _is_merged(out_dir) and not force:
        logger.info("Reusing existing merged checkpoint at %s (use --force to re-merge).",
                    out_dir)
        return out_dir
    # An out_dir that exists but is INCOMPLETE has to be redone rather than reused:
    # a merge interrupted mid-write (Ctrl-C during the minutes it takes) or one from
    # before generation_config.json was written leaves a dir that exists and is
    # unservable — _is_merged spells out why. Existence alone is also all the child's
    # own reuse check looks at, so force it past that too.
    redo = force or out_dir.exists()
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--adapter", str(adapter_dir.resolve()),
        "--out", str(out_dir),
        "--merge-in-process", "--no-serve",
    ]
    if redo:
        cmd.append("--force")
        if not force:
            logger.warning(
                "%s exists but is not a complete merged checkpoint (interrupted "
                "merge, or one predating generation_config.json); re-merging it.",
                out_dir,
            )
    logger.info("Merging in a subprocess (its VRAM is then fully released before "
                "vLLM starts): %s", " ".join(cmd))
    result = subprocess.run(cmd, env=dict(os.environ, **{_MERGE_CHILD_ENV: "1"}))
    if result.returncode != 0:
        sys.exit(f"Merge subprocess failed (exit {result.returncode}); see its output "
                 f"above. Nothing was served.")
    # The child reports success by exit code; confirm it actually produced a
    # checkpoint we can serve rather than trusting that and failing later inside vLLM.
    if not _is_merged(out_dir):
        sys.exit(f"Merge subprocess exited 0 but {out_dir} is not a complete merged "
                 f"checkpoint (missing weights/config/generation_config).")
    return out_dir


def serve(merged_dir: Path) -> None:
    """Serve `merged_dir` directly, blocking until Ctrl-C.

    Mirrors the pipeline's expert-LoRA epoch boundary (run_pipeline.main): restart
    vLLM on the merged full checkpoint under served name config.VLLM_MODEL. No
    adapter to hot-load — the trained weights are baked in.
    """
    from pipeline import vllm_server

    # Tear the server down on every exit path — including the SIGTERM/SIGHUP that
    # would otherwise kill this process without running the finally below, orphaning
    # the vLLM session with all its VRAM still allocated and the port still bound.
    vllm_server.install_shutdown_handlers()
    # One try/finally around the whole lifecycle so an abnormal exit (start()
    # raising, or the engine dying later) still runs stop() and killpg's the vLLM
    # session — otherwise it orphans the engine / leaves a zombie API front-end on
    # the port and keeps the GPU reservation held until this process is killed.
    try:
        logger.info("Starting vLLM on %s (GPUs %s) …", merged_dir, config.VLLM_GPUS)
        vllm_server.start(str(merged_dir))
        logger.info(
            "vLLM ready at %s (served model name: %s). Press Ctrl-C to stop.",
            config.VLLM_BASE_URL, config.VLLM_MODEL,
        )
        # Poll instead of signal.pause() so a dead engine ends the wait rather than
        # pausing forever with the GPUs reserved (pause() wouldn't wake on the
        # child's death).
        while vllm_server.is_alive():
            time.sleep(5)
        logger.error("vLLM server exited on its own; shutting down.")
    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down vLLM server.")
    finally:
        vllm_server.stop()


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--epoch", type=int, help="Epoch number to serve (epoch_<N> under the run dir).")
    src.add_argument("--adapter", help="Explicit epoch adapter (or already-merged) dir to serve.")
    ap.add_argument("--source", choices=sorted(_SOURCE_BASES), default="pipeline",
                    help="Which trainer's checkpoints to serve; picks the base dir "
                         "holding its per-run dirs (pipeline: $CHECKPOINT_DIR or "
                         "./checkpoints; sft: $SFT_OUTPUT_DIR or ./sft/checkpoints; "
                         "opsd: $OPSD_CHECKPOINT_DIR/$OPSD_OUTPUT_DIR or "
                         "./opsd/checkpoints). Overridden by --checkpoint-dir.")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="Base dir holding the per-run checkpoint dirs (default: the "
                         "--source base).")
    ap.add_argument("--run-id", default=None,
                    help="RUN_ID subdir under --checkpoint-dir (default: most recent run).")
    ap.add_argument("--out", default=None,
                    help="Merged-checkpoint output dir (default: <checkpoint>_merged).")
    ap.add_argument("--force", action="store_true",
                    help="Re-merge even if the merged dir already exists.")
    ap.add_argument("--no-serve", action="store_true",
                    help="Merge + resolve only; don't start vLLM.")
    ap.add_argument("--merge-in-process", action="store_true",
                    help="Merge in THIS process instead of a subprocess. The subprocess "
                         "is what guarantees the merge's VRAM is back before vLLM starts "
                         "(see _merge_out_of_process), so this is how that subprocess "
                         "runs itself — and an escape hatch if one can't be spawned.")
    args = ap.parse_args()
    # Reserve + confine this instance's GPUs before anything touches CUDA (the merge
    # imports torch), so a concurrent serve on other cards can't collide. Skipped in
    # the merge child, which inherits the parent's reservation and its pin.
    if os.environ.get(_MERGE_CHILD_ENV) != "1":
        _pin_gpus()

    if args.adapter:
        ckpt_dir = _resolve_checkpoint(Path("."), None, args.adapter)
    else:
        base = Path(args.checkpoint_dir) if args.checkpoint_dir else _source_base(args.source)
        run_dir = _resolve_run_dir(base, args.run_id)
        ckpt_dir = _resolve_checkpoint(run_dir, args.epoch, None)

    # Already a full/merged checkpoint → serve as-is; otherwise merge the adapter.
    if _is_merged(ckpt_dir):
        merged_dir = ckpt_dir
    else:
        out_dir = Path(args.out) if args.out else ckpt_dir.parent / f"{ckpt_dir.name}_merged"
        if args.merge_in_process:
            # Merging and then serving from the SAME process is exactly the
            # configuration that strands GiB on the second card (see
            # _merge_out_of_process). The merge child always pairs this flag with
            # --no-serve, so only a deliberate operator override lands here — say
            # why the launch that follows may fail rather than letting it look
            # like a vLLM bug.
            if not args.no_serve:
                logger.warning(
                    "--merge-in-process without --no-serve: this process will hold "
                    "the merge's VRAM (the allocator does not return all of it), so "
                    "vLLM may abort with 'Free memory on device cuda:N ... less than "
                    "desired GPU memory utilization'. Drop the flag to merge in a "
                    "subprocess instead."
                )
            merged_dir = _merge(ckpt_dir, out_dir, args.force)
        else:
            merged_dir = _merge_out_of_process(ckpt_dir, out_dir, args.force)

    if args.no_serve:
        logger.info("Merge/resolve complete (--no-serve). Merged checkpoint: %s", merged_dir)
        return
    serve(merged_dir)


if __name__ == "__main__":
    main()
