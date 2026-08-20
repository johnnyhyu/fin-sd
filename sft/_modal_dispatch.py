"""Dispatch the SFT flow to Modal when USE_MODAL=1 (mirrors the root launch.py).

`sft.run` / `sft.train` call `dispatch(...)` instead of running locally. It shells
out to `modal run sft/modal_app.py`, forwarding the stage flags. Kept import-light
(no `modal` import) so the local, non-Modal path never needs Modal installed.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_APP = Path(__file__).resolve().parent / "modal_app.py"


def dispatch(
    *, skip_generate: bool = False, skip_train: bool = False, limit: int | None = None
) -> int:
    """Launch the SFT run on Modal and return the `modal run` exit code."""
    from pipeline import config

    # `--detach` decouples the run from this client so a laptop sleep / network
    # blip can't tear down a multi-hour job (the app spawns and returns fast).
    cmd = ["modal", "run", "--detach", str(_APP)]
    if skip_generate:
        cmd.append("--skip-generate")
    if skip_train:
        cmd.append("--skip-train")
    if limit is not None:
        cmd += ["--limit", str(limit)]

    # macOS: hold a power assertion for the (brief) lifetime of `modal run` so an
    # idle laptop won't sleep before the detached app is registered.
    if sys.platform == "darwin":
        cmd = ["caffeinate", "-i", "-s", *cmd]

    print(f"USE_MODAL=1 → launching SFT on Modal ({config.MODAL_GPU}) …", flush=True)
    return subprocess.call(cmd, env=os.environ.copy())
