"""GPU placement for local SFT runs — the `launch.py` autoselect, for `sft`.

`sft.run` / `sft.train` used to take whatever GPUs happened to be visible: the HF
student loaded with `device_map="auto"` across every card on the box, while the
per-epoch validation server started on the fixed `config.VLLM_GPUS` default
("0,1") — cards the trainer had usually already spread onto. Nothing reserved
anything, so two concurrent runs (or an SFT run next to a `launch.py` one) landed
on the same silicon and OOMed each other.

The placement itself lives in `pipeline.utils.configure_train_gpus` (shared with
`opsd`, and built on the same reserved idle-GPU finder launch.py uses), so all
three flows are safe to run side by side. This module only adapts it to the SFT
config: SFT names its student and its per-GPU memory cap with `SFT_`-prefixed
knobs, and `sft.model` reads the cap off `sft.config`, so the resolved value is
mirrored back there.
"""
from __future__ import annotations

import os

from pipeline.utils import configure_train_gpus

from . import config


def configure_training_gpus(needs_vllm: bool = False) -> None:
    """Pin and reserve the GPUs this SFT run will train (and optionally serve) on.

    `needs_vllm` is whether the run will actually start a vLLM server, which for
    SFT now means `--serve` (or SFT_SERVE_AFTER_TRAIN) and nothing else — hence
    the False default. Per-epoch validation used to force it on, because it merged
    the adapter and served the checkpoint; `sft.validate` decodes from the live
    in-process weights instead, so a validating run needs no server and no third
    card. When it is False the 20B path reserves just the two training cards the
    student shards over.

    Must be called before anything initialises CUDA (the HF student load). Raises
    RuntimeError (from reserve_open_gpus) if not enough idle cards exist — the
    generate stage is API-only and resumable, so callers reserve here, right before
    training, rather than holding cards through a long distillation.
    """
    # configure_train_gpus picks the card BLOCK; the per-card ceilings are derived
    # from those cards' live VRAM when sft.model loads the student, so what comes
    # back here is "" unless the operator pinned caps of their own.
    config.HF_MAX_MEMORY = configure_train_gpus(
        config.STUDENT_MODEL,
        needs_vllm=needs_vllm,
        # sft.model reads the cap from sft.config (SFT_HF_MAX_MEMORY), not the
        # pipeline's HF_MAX_MEMORY, so that is the name to resolve an operator pin
        # against. Usually neither is set and the ceilings are derived per card.
        max_memory_env="SFT_HF_MAX_MEMORY",
        full_finetune=not config.USE_LORA,
    )
    os.environ["SFT_HF_MAX_MEMORY"] = config.HF_MAX_MEMORY
