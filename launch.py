#!/usr/bin/env python3
"""Single entrypoint that runs the pipeline locally or on Modal.

Flip one env var (USE_MODAL in .env, or `USE_MODAL=1 python launch.py`):

    USE_MODAL=0  → run run_pipeline.main() on local GPUs (today's behaviour)
    USE_MODAL=1  → dispatch to Modal via `modal run modal_app.py`
                   (GPU type from MODAL_GPU, default A100-80GB:8)

See modal_app.py for the Modal prerequisites (token + secrets).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from pipeline import config

ROOT = Path(__file__).parent

# The 20B BF16 LoRA run fits one H100 for serving + one for training (see the
# VRAM budget: ~13 GiB for vLLM, ~50-65 GiB for the HF trainer), so instead of
# the 6-GPU trainer block the 120B model needs, auto-place it on the next two
# idle cards on an 8xH100 box: first -> vLLM, second -> trainer.
#
# Keyed on the model FAMILY (utils.model_family), not on one exact repo id: the
# same 20B weights are spelled "unsloth/gpt-oss-20b-BF16", "unsloth/gpt-oss-20b",
# "openai/gpt-oss-20b", or a local checkout path, and an equality check sent every
# spelling but the blessed one down the 120B branch — reserving a 6-GPU trainer
# block and serving with tensor-parallel 2 for a model that needs one card per
# role, on a box where those cards are usually already spoken for.
_GPU_AUTOSELECT_FAMILY = "gpt-oss-20b"


def _autoselect_20b_gpus() -> None:
    """Pin a 20B BF16 run to the next two idle GPUs: first -> vLLM, second -> HF.

    Only used when HF_MODEL_PATH names a _GPU_AUTOSELECT_FAMILY model, and only
    when the operator has not already pinned CUDA_VISIBLE_DEVICES (an explicit
    override always wins). Picks two cards with <4 GiB in use via nvidia-smi, then:
      * vLLM  -> the first card  (config.VLLM_GPUS, tensor-parallel 1)
      * HF    -> the second card (this process's CUDA_VISIBLE_DEVICES)
    keeping the two DISJOINT as the pipeline requires. Pins CUDA_DEVICE_ORDER to
    PCI_BUS_ID first so CUDA ordinals match the nvidia-smi indices we selected.
    Must run before run_pipeline (or the profiler subprocess) touches CUDA.
    """
    from pipeline.utils import reserve_open_gpus

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # reserve_ (not select_) so two launch.py started at once can't both grab the
    # same "idle" cards — a freshly-selected card shows no VRAM until its model
    # loads, so a plain nvidia-smi scan races. The reservation locks are held for
    # this process's lifetime, covering the whole run.
    vllm_gpu, train_gpu = reserve_open_gpus(2, max_used_gib=4.0)

    # vLLM subprocess reads these from config at launch (pipeline/vllm_server.py);
    # one card means tensor-parallel 1. Mirror to os.environ so any subprocess
    # that re-imports config (e.g. the profiler) sees the same placement.
    config.VLLM_GPUS = os.environ["VLLM_GPUS"] = str(vllm_gpu)
    config.VLLM_TENSOR_PARALLEL = 1
    os.environ["VLLM_TENSOR_PARALLEL"] = "1"

    # The HF trainer inherits this process's CUDA_VISIBLE_DEVICES; give it only the
    # second card. No memory cap goes with it: the card was just verified idle, so
    # the ceiling the trainer loads under is that card's own free VRAM minus the
    # reserve a step needs, read off the driver at load time (see
    # pipeline.utils.resolve_max_memory). This used to pin a flat "72", which was
    # an 80 GB card's number written down by hand.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(train_gpu)

    print(f"20B BF16 auto-placement: vLLM -> GPU {vllm_gpu}, trainer -> GPU "
          f"{train_gpu} (next two idle cards, <4 GiB used).", flush=True)


def _ensure_expert_profile() -> None:
    """Build the expert-activation profile cache before a local run if it's stale.

    The expert-LoRA path (LORA_TARGET_EXPERTS + PROFILE_EXPERT_ACTIVATIONS)
    reads config.EXPERT_PROFILE_PATH to pick which experts to adapt,
    and that cache is keyed to the model it was profiled on (its "model" field is
    the config.HF_MODEL_PATH at write time). Re/build it here when it is missing or
    was written against a different HF_MODEL_PATH, so a bare `python launch.py`
    never trains against a stale/mismatched profile. Runs tools/profile_experts.py
    as a subprocess so the base model it loads is released (GPU freed) before
    run_pipeline loads its own copy.

    Skipped entirely when the profile would not be consumed (either profiling flag
    off) — building it then would be wasted GPU time.
    """
    if not config.LORA_TARGET_EXPERTS or not config.PROFILE_EXPERT_ACTIVATIONS:
        return

    path = Path(config.EXPERT_PROFILE_PATH)
    reason = None
    if not path.exists():
        reason = f"no profile cache at {path}"
    else:
        try:
            cached_model = json.loads(path.read_text()).get("model")
        except (json.JSONDecodeError, OSError) as exc:
            reason = f"profile cache {path} unreadable ({exc})"
        else:
            if cached_model != config.HF_MODEL_PATH:
                reason = (f"profile cache {path} was built for {cached_model!r}, "
                          f"but HF_MODEL_PATH is {config.HF_MODEL_PATH!r}")
    if reason is None:
        print(f"Expert profile cache up to date for {config.HF_MODEL_PATH} "
              f"({path}); skipping profiling.", flush=True)
        return

    print(f"Building expert profile cache ({reason}) …", flush=True)
    cmd = [
        sys.executable, str(ROOT / "tools" / "profile_experts.py"),
        "--control", "none",                       # single-corpus cache-writer mode
        "--data", str(ROOT / "data" / "trainingset.json"),
    ]
    subprocess.run(cmd, check=True, env=os.environ.copy())


def main() -> int:
    if not config.USE_MODAL:
        # Local: run in-process, identical to `python run_pipeline.py` — except
        # we pin the HF trainer's GPUs here so a bare `python launch.py` is safe
        # without a CUDA_VISIBLE_DEVICES prefix. Two placement paths, both keeping
        # the trainer DISJOINT from the vLLM GPUs the pipeline requires:
        #   * 20B BF16 (the default model): auto-place on the next two idle cards
        #     via _autoselect_20b_gpus() — first -> vLLM, second -> trainer. A 20B
        #     LoRA run only needs one card per role, so we don't reserve a block.
        #   * anything else (e.g. the 120B): fall back to the fixed 4-7 trainer
        #     block, leaving VLLM_GPUS (default 0,1,2,3) for serving — four cards
        #     each way, which is what the merged 120B checkpoint vLLM restarts on at
        #     each epoch boundary needs to be served at all (see config.VLLM_GPUS).
        # Either way an explicit CUDA_VISIBLE_DEVICES always wins (the auto path is
        # gated on it being unset; the fallback uses setdefault). config import
        # above already merged .env into os.environ. Must run before run_pipeline
        # imports torch and initialises CUDA.
        from pipeline.utils import model_family
        if model_family(config.HF_MODEL_PATH) == _GPU_AUTOSELECT_FAMILY \
                and "CUDA_VISIBLE_DEVICES" not in os.environ:
            _autoselect_20b_gpus()
        else:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4,5,6,7")
            # Reserve the fixed vLLM + trainer blocks so a concurrent autoselecting
            # run (20B launch.py or a serve tool) won't pick these same cards.
            from pipeline.utils import reserve_gpus
            _fixed = {int(g) for spec in (config.VLLM_GPUS,
                                          os.environ["CUDA_VISIBLE_DEVICES"])
                      for g in spec.split(",") if g.strip() != ""}
            reserve_gpus(sorted(_fixed))
        # Refresh the per-expert activation profile if it's missing or was built
        # for a different HF_MODEL_PATH, before run_pipeline loads the student. Runs
        # after CUDA_VISIBLE_DEVICES is pinned so the profiler uses the HF GPUs.
        _ensure_expert_profile()
        from run_pipeline import main as run_main

        run_main()
        return 0

    # Modal: hand off to the Modal CLI. `--detach` decouples the run from this
    # client connection so a local network blip / laptop sleep can't tear down a
    # multi-hour training job (see modal_app.py's 24h timeout). Logs then live in
    # the Modal dashboard (and W&B) rather than streaming to this terminal.
    #
    # Belt-and-suspenders on macOS: `caffeinate -i -s` holds a power assertion for
    # the lifetime of the wrapped `modal run` so an idle laptop won't sleep while
    # the client is still attached. Detach already protects the *remote* job, but
    # this also (a) closes the small startup window before the app is fully
    # registered detached, and (b) keeps the live log stream alive instead of
    # dying on every sleep. It does NOT prevent lid-close sleep on battery.
    cmd = ["modal", "run", "--detach", str(ROOT / "modal_app.py")]
    if sys.platform == "darwin":
        cmd = ["caffeinate", "-i", "-s", *cmd]
    print(f"USE_MODAL=1 → launching on Modal ({config.MODAL_GPU}) …", flush=True)
    return subprocess.call(cmd, env=os.environ.copy())


if __name__ == "__main__":
    sys.exit(main())
