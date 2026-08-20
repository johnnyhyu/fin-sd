"""Shared utilities: logging and exponential-backoff HTTP/OpenRouter calls."""
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Optional

import requests

from . import config

try:
    import fcntl  # POSIX only (Linux/macOS); the pipeline never runs on Windows.
except ImportError:  # pragma: no cover - defensive; cross-process locks degrade to no-ops.
    fcntl = None

logger = logging.getLogger("pipeline")

RETRY_STATUSES = {429, 500, 502, 503, 504}

# Per-backend concurrency caps. When items are processed concurrently (see
# run_pipeline), every HTTP request funnels through post_with_retry, so gating
# here by label bounds how many requests hit each server at once — vLLM's
# KV-cache budget and OpenRouter's rate limits — without the call sites needing
# to know about threading. The slot is held across backoff sleeps too, so a
# struggling server is not piled onto while a request retries.
_VLLM_SEM = threading.BoundedSemaphore(config.VLLM_MAX_CONCURRENCY)
_OPENROUTER_SEM = threading.BoundedSemaphore(config.OPENROUTER_MAX_CONCURRENCY)

# Serialises appends to the JSONL debug logs (reasoning_log, state-finder log)
# so concurrent multi-KB writes from worker threads cannot interleave.
FILE_LOG_LOCK = threading.Lock()


# ── Cooperative shutdown ─────────────────────────────────────────────────────
# Set by vllm_server's signal handlers; read by every blocking wait a worker
# thread can be parked in.
#
# A signal handler only ever runs on the MAIN thread, and the exception it raises
# (KeyboardInterrupt for Ctrl-C, SystemExit for SIGTERM) surfaces at the next
# bytecode boundary — which, for a main thread blocked in Future.result() inside
# a ThreadPoolExecutor window, does not come until every in-flight worker has
# finished. Measured on this box: SIGINT delivered T+64.1s, teardown at T+81.2s;
# SIGTERM delivered 19:10:30, teardown 19:10:47. Both windows ran to COMPLETION,
# ~17s of a Ctrl-C that looked like a hang with the server still holding its VRAM
# — and with real items (300s request timeout, MAX_RETRIES backoff) that stretches
# to many minutes.
#
# The worker threads are where the waiting actually happens, so they are what has
# to notice. This flag is the cross-thread channel the raised exception cannot be:
# workers poll it and bail out instead of starting or retrying more work.
SHUTDOWN = threading.Event()


class ShutdownRequested(RuntimeError):
    """Raised in a worker thread once a shutdown signal has been received.

    Distinguishable from a genuine backend failure so callers can tell an
    operator-requested abort from a run that actually broke.
    """


def request_shutdown() -> None:
    """Signal every worker thread to stop as soon as it can. Idempotent."""
    SHUTDOWN.set()


def shutdown_requested() -> bool:
    """True once a SIGINT/SIGTERM/SIGHUP has been seen by the signal handlers."""
    return SHUTDOWN.is_set()


def interruptible_sleep(seconds: float) -> None:
    """Sleep, but wake immediately if a shutdown is requested — then raise.

    Retry backoff is the single longest thing a worker sits in during teardown
    (RETRY_BASE_DELAY doubles up to ~32s at MAX_RETRIES=5, and the slot is held
    across it), so a plain time.sleep here is what makes a Ctrl-C feel wedged.
    """
    if SHUTDOWN.wait(seconds):
        raise ShutdownRequested("shutdown requested during retry backoff")


def map_interruptible(fn, items, *, max_workers: int, poll: float = 0.5) -> list:
    """`list(ThreadPoolExecutor.map(fn, items))`, but abortable within `poll` seconds.

    Same contract as the ex.map it replaces: results in SUBMISSION order (batch
    composition and gradient-accumulation boundaries depend on that), and the
    first failure in that order propagates.

    What it adds is the ability to stop. Two things made the plain version
    unabortable, and both are fixed here:

      * The main thread parked in Future.result() with no timeout. A signal
        handler runs immediately, but the KeyboardInterrupt/SystemExit it raises
        only surfaces at the next bytecode boundary — which never comes while the
        thread is blocked. Waiting with a timeout gives the interpreter that
        boundary every `poll` seconds, so a pending signal lands promptly, and we
        re-check SHUTDOWN ourselves for the same reason.
      * `with ThreadPoolExecutor(...)` tears down as shutdown(wait=True) even when
        the block is leaving on a Ctrl-C, re-blocking for as long as the slowest
        in-flight request takes. On the abort path we cancel what has not started
        and do NOT wait for what has.

    Measured before this: SIGINT at T+64.1s, teardown at T+81.2s, and the window
    still ran to completion — 17s of apparent hang with vLLM holding all its VRAM,
    proportional in a real run to the 300s request timeout and MAX_RETRIES backoff.

    The abandoned worker threads are each one already-issued HTTP request from
    returning (requests offers no cancellation) and the server is about to be
    stopped anyway, so nothing is left half-written that a completed join would
    have finished.
    """
    ex = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = [ex.submit(fn, item) for item in items]
        pending = set(futures)
        while pending:
            _, pending = futures_wait(pending, timeout=poll,
                                      return_when=FIRST_COMPLETED)
            if SHUTDOWN.is_set():
                raise ShutdownRequested(
                    f"aborted with {len(pending)} item(s) still in flight"
                )
        # Submission order, and the first failure in it — exactly ex.map.
        results = [f.result() for f in futures]
    except BaseException:
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown(wait=True)
    return results


def _concurrency_gate(label: str):
    """Return the backend semaphore guarding `label`, or a no-op context.

    vLLM calls use labels starting with "vLLM" (e.g. "vLLM", "vLLM-probe");
    OpenRouter calls use "OpenRouter". Anything else (e.g. the one-off LoRA
    sync) is ungated.
    """
    if label.startswith("vLLM"):
        return _VLLM_SEM
    if label == "OpenRouter":
        return _OPENROUTER_SEM
    return nullcontext()


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Cross-process resource locks (GPUs, vLLM ports, checkpoint merges) ────────
# Multiple pipeline processes can share one box (launch.py + the serve tools, or
# several runs on disjoint GPUs). They coordinate through flock'd files in a shared
# directory so they never grab the same physical GPU or vLLM port. A lock is held
# for the LIFETIME of the process that acquired it — the fd stays open, and the OS
# drops the flock when the process exits — so a card/port stays reserved for as
# long as its owner is alive, even before its VRAM registers in nvidia-smi.
_LOCK_DIR = Path(os.getenv("PIPELINE_LOCK_DIR", "/tmp/pipeline_locks"))
_held_locks: dict[str, object] = {}
_lock_guard = threading.Lock()
# Filenames are capped at 255 bytes on ext4/APFS. Lock names derived from a path
# (serve_checkpoint's per-output-dir merge lock) blow past that on a deep
# checkpoint tree, and open() then raises ENAMETOOLONG — the merge dies instead of
# serialising. Longer names collapse to a stable hash so the lock still works.
_MAX_LOCK_NAME = 200


def _lock_path(name: str) -> Path:
    """Path of the lock file backing `name`, hashing names too long to be filenames.

    The hash is of the FULL name, so two distinct long names never collide onto one
    lock (which would serialise unrelated operations) and the same name always maps
    to the same file across processes.
    """
    if len(name) > _MAX_LOCK_NAME:
        digest = hashlib.sha256(name.encode()).hexdigest()[:32]
        name = f"{name[:_MAX_LOCK_NAME]}_{digest}"
    return _LOCK_DIR / f"{name}.lock"


def _acquire_named_lock(name: str) -> bool:
    """Grab a process-lifetime, cross-process exclusive lock `name`; True if acquired.

    Non-blocking: a name another live pipeline process already holds returns False
    immediately. The open file object is stashed in `_held_locks` so the flock is
    held until this process exits. Without fcntl (non-POSIX) this degrades to an
    in-process-only guard.
    """
    with _lock_guard:
        if name in _held_locks:
            return True
        _LOCK_DIR.mkdir(parents=True, exist_ok=True)
        f = open(_lock_path(name), "w")
        if fcntl is not None:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                f.close()
                return False
        _held_locks[name] = f
        return True


def _release_named_lock(name: str) -> None:
    """Release a lock previously taken with `_acquire_named_lock` (no-op if unheld)."""
    with _lock_guard:
        f = _held_locks.pop(name, None)
    if f is None:
        return
    try:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    finally:
        f.close()


@contextmanager
def file_lock(name: str):
    """Blocking, cross-process exclusive lock named `name`, released on exit.

    Serialises an operation across concurrent pipeline processes on one box — e.g.
    two serve_checkpoint runs merging the same epoch into the same output dir, which
    would otherwise interleave writes and corrupt the checkpoint. Unlike the
    resource locks above this blocks until the lock is free rather than failing.
    """
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    f = open(_lock_path(name), "w")
    try:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()


def reserve_port(port: int) -> bool:
    """Reserve `port` for this process across pipeline instances; True if acquired.

    A sibling pipeline process that already claimed the port returns False so the
    caller can move to another one. Held for the process lifetime (see the lock
    registry above); pair with an actual bind check for foreign, non-pipeline
    servers on the port.
    """
    return _acquire_named_lock(f"port_{port}")


def release_port(port: int) -> None:
    """Release a port reserved with `reserve_port`."""
    _release_named_lock(f"port_{port}")


def _nvidia_smi_csv(query_flag: str) -> str:
    """Run one `nvidia-smi` CSV query and return its raw body (no header, no units).

    `query_flag` is the whole flag, e.g. "--query-gpu=index,memory.used" or
    "--query-compute-apps=gpu_bus_id,pid". Raises RuntimeError if nvidia-smi is
    unavailable, so callers can decide whether that is fatal.
    """
    try:
        return subprocess.check_output(
            ["nvidia-smi", query_flag, "--format=csv,noheader,nounits"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"could not query GPUs via nvidia-smi: {exc}") from exc


def gpu_memory_gib(indices: Optional[list[int]] = None) -> dict[int, tuple[float, float]]:
    """{physical index: (free_gib, total_gib)}, straight from the driver.

    The same view every other process on the box has, so it answers the question a
    launching server actually cares about — "is the VRAM I am about to ask for free
    *right now*" — including memory held by processes this pipeline knows nothing
    about, and memory a dying one has not returned yet. `indices` filters the result
    (indices absent from nvidia-smi's output are omitted rather than guessed at);
    None reports every physical GPU, which is what a caller sizing a layout wants
    before it has picked any cards.

    Reads nvidia-smi rather than torch so it can run BEFORE CUDA is initialised —
    the point at which the GPU block is still being chosen — and so it sees
    physical indices regardless of CUDA_VISIBLE_DEVICES.
    """
    out = _nvidia_smi_csv("--query-gpu=index,memory.free,memory.total")
    wanted = None if indices is None else set(indices)
    mem: dict[int, tuple[float, float]] = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        idx = int(parts[0])
        if wanted is None or idx in wanted:
            mem[idx] = (float(parts[1]) / 1024, float(parts[2]) / 1024)
    return mem


# ── Per-GPU placement ceilings for device_map="auto" ─────────────────────────
# A sharded HF load sizes its placement from a per-card ceiling, and that ceiling
# has to leave room for everything the WEIGHTS don't cover: one MoE layer's
# activations, the expert-LoRA factors PEFT materialises per layer, the fp32 logits
# on the lm_head card, AdamW's moments. Accelerate leaves none of it back on a
# multi-card load — get_balanced_memory sizes the split from the model and lets the
# last card keep whatever the driver reports free — so stating the ceiling is ours.
#
# It is stated as a RESERVE against live VRAM (`free - HF_MIN_HEADROOM_GIB` per
# card), not as a hand-tuned absolute cap. Two reasons, both of which the old
# "62,62,62,62" default got wrong the moment the box changed:
#   * the reserve is a measured, model-side quantity (~3 GiB/card for the 120B at
#     8192 tokens, doubled for margin) and it is the SAME quantity log_vram_budget
#     audits once the weights land — so the loader holds back exactly what the
#     post-load check demands, instead of the two agreeing by coincidence on an
#     80 GB card and disagreeing everywhere else;
#   * the capacity half comes from the driver, so the caps track the hardware:
#     a 40 GB card next to an 80 GB one, a card a neighbouring process is already
#     half-holding, or a wider/narrower trainer block than the cap string had
#     entries for.
def resolve_max_memory(
    spec: str = "",
    *,
    model_path: Optional[str] = None,
    reserve_gib: Optional[float] = None,
    label: str = "HF_MAX_MEMORY",
) -> Optional[dict[int, str]]:
    """The `max_memory` dict for a `device_map="auto"` load — derived, or `spec` if pinned.

    `spec` is the operator's override: comma-separated GiB ceilings, one per visible
    card by position ("62,62,62,62"), honored as written. Entries beyond the visible
    GPU count are dropped with a warning. Empty (the default) derives the ceilings
    from the driver — each visible card's live free VRAM minus `reserve_gib`
    (default config.HF_MIN_HEADROOM_GIB), the per-card headroom one training step
    needs on top of its shard of the weights.

    A card holding less free VRAM than the reserve is EXCLUDED from the result
    rather than given a ceiling it cannot honor, so accelerate places nothing there.
    `model_path` then buys the failure that should follow: pass it and the total
    derived capacity is checked against a 2-byte-per-param estimate of the weights
    (bf16_weight_gib) BEFORE the load, because accelerate does not refuse a block
    too narrow to hold the model — infer_auto_device_map quietly maps the overflow
    to `disk`, and from_pretrained then complains about a missing `offload_folder`,
    which names neither the card that was excluded nor the GiB that were missing.
    Skip it for a load whose dtype is under 2 bytes per param (nothing here trains
    one: MXFP4 has no backward kernels), where the estimate would overshoot.

    Returns None when CUDA is unavailable (nothing to cap, and HF ignores max_memory
    anyway). Raises RuntimeError when CUDA is up but the visible cards cannot hold
    the model, naming the processes holding the VRAM.

    Only meaningful for the "auto"/"balanced" device maps: HF ignores max_memory
    when it isn't computing the placement itself.
    """
    import torch

    caps = [tok.strip() for tok in (spec or "").split(",") if tok.strip()]
    n_visible = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if caps:
        if n_visible and len(caps) > n_visible:
            logger.warning(
                "%s pins %d cap(s) but only %d GPU(s) are visible; ignoring the "
                "extra entries.", label, len(caps), n_visible,
            )
            caps = caps[:n_visible]
        pinned = {i: f"{cap}GiB" for i, cap in enumerate(caps)}
        logger.info("Placement caps pinned by %s: %s.", label, pinned)
        return pinned

    if not n_visible:
        return None

    reserve = config.HF_MIN_HEADROOM_GIB if reserve_gib is None else reserve_gib
    caps_gib: dict[int, int] = {}
    rows, starved = [], []
    for i in range(n_visible):
        free, total = (x / 1024 ** 3 for x in torch.cuda.mem_get_info(i))
        # Floor to whole GiB: accelerate parses the string form, and a fractional
        # ceiling only ever buys back memory the reserve just set aside.
        cap = int(free - reserve)
        if cap <= 0:
            starved.append((i, free, total))
            continue
        caps_gib[i] = cap
        rows.append(f"cuda:{i} {free:.1f}/{total:.1f} GiB free -> {cap}GiB")

    for i, free, total in starved:
        logger.warning(
            "cuda:%d has only %.1f of %.1f GiB free, less than the %.1f GiB a "
            "training step needs per card; excluding it from the placement. %s",
            i, free, total, reserve,
            "; ".join(gpu_compute_apps([_physical_gpu_index(i)])) or
            "(no compute apps reported — memory may be held by a dying process)",
        )
    if not caps_gib:
        raise RuntimeError(
            f"none of the {n_visible} visible GPU(s) has {reserve:.1f} GiB free, so "
            "there is nowhere to place the model. Free the cards (or widen "
            f"CUDA_VISIBLE_DEVICES), or lower HF_MIN_HEADROOM_GIB if the tight fit "
            "is deliberate."
        )
    logger.info("Placement caps derived from live VRAM (reserving %.1f GiB/card "
                "for the step) — %s.", reserve, "; ".join(rows))

    capacity = sum(caps_gib.values())
    weights = bf16_weight_gib(model_path) if model_path else None
    if weights and capacity < weights:
        raise RuntimeError(
            f"{model_path} needs ~{weights:.0f} GiB of BF16 weights but the "
            f"{len(caps_gib)} usable card(s) offer {capacity} GiB after reserving "
            f"{reserve:.1f} GiB each for the training step"
            + (f" ({len(starved)} card(s) excluded as too full)" if starved else "")
            + ". Widen CUDA_VISIBLE_DEVICES, free the excluded cards, or pin "
            f"{label} yourself if the estimate is wrong for this checkpoint."
        )
    return {i: f"{cap}GiB" for i, cap in caps_gib.items()}


def _physical_gpu_index(ordinal: int) -> int:
    """Map a CUDA ordinal back to the physical index nvidia-smi reports.

    Only needed to make a VRAM complaint actionable: torch counts from 0 within
    CUDA_VISIBLE_DEVICES, while the pids holding a card are reported against its
    physical index, so "cuda:0 is full" and "pid 123 is on GPU 4" are the same card
    under two numbers. Falls back to the ordinal when the pin is absent or names
    UUIDs rather than indices.
    """
    visible = [g.strip() for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
               if g.strip()]
    if ordinal < len(visible) and visible[ordinal].isdigit():
        return int(visible[ordinal])
    return ordinal


def log_vram_budget(what: str, min_headroom_gib: float) -> bool:
    """Log what the loaded model left on each visible card, and flag a tight fit.

    The training step's transient cost is bounded and knowable — one MoE layer's
    activations plus its expert-LoRA factors, about 3 GiB per card on the 120B once
    pipeline/expert_lora.py and pipeline/attention.py are in play — so whether a
    layout will survive is decidable the moment the weights are placed, not twenty
    minutes later in the middle of an epoch. That is the difference this reports:
    an OOM here reads as "card 6 has 1.2 GiB free after the weights", which names
    the fix (widen the block, or raise the reserve resolve_max_memory holds back),
    where the same failure during a backward pass reads as a CUDA allocation error
    inside a checkpoint recompute.

    `min_headroom_gib` is the same per-card reserve resolve_max_memory subtracted
    from live VRAM when it sized the placement, so on a derived layout this check
    re-measures that promise against where the weights actually landed (accelerate
    balances by model size, so a card can still come out tighter than its ceiling).

    Reports per card: what THIS process has allocated (torch's own accounting) and
    what the driver still has free (which also catches a neighbour on the card).
    Returns False and warns when any card is below `min_headroom_gib`; never raises
    — a tight fit is sometimes deliberate, and the run should be the operator's
    call rather than ours.
    """
    import torch

    if not torch.cuda.is_available():
        return True
    rows, tight = [], []
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
        free, total = (x / 1024 ** 3 for x in torch.cuda.mem_get_info(i))
        rows.append(f"cuda:{i} {allocated:.1f} GiB allocated ({reserved:.1f} "
                    f"reserved), {free:.1f}/{total:.1f} GiB free")
        if free < min_headroom_gib:
            tight.append((i, free))
    logger.info("VRAM after %s — %s.", what, "; ".join(rows))
    if tight:
        logger.warning(
            "Only %s left after %s, below the %.1f GiB per-card headroom a training "
            "step needs (one layer's activations + expert-LoRA factors). Widen the "
            "trainer block (CUDA_VISIBLE_DEVICES), or expect an OOM mid-epoch on the "
            "longest sequence of the run.",
            "; ".join(f"cuda:{i} {free:.1f} GiB" for i, free in tight),
            what, min_headroom_gib,
        )
        return False
    return True


def vram_peak_report(reset: bool = True) -> str:
    """Per-card peak allocated/reserved since the last reset, as one log-ready line.

    Peak — not current — because the number that decides whether a run survives is
    the high-water mark inside a backward pass, which no snapshot taken between
    steps can see. Logged per epoch so a run that is quietly creeping toward the
    ceiling (a longer probe, a wider batch) is visible before it OOMs rather than
    after, and so the headroom a future layout change can spend is a measured
    quantity rather than a guess.
    """
    import torch

    if not torch.cuda.is_available():
        return "no CUDA devices"
    parts = []
    for i in range(torch.cuda.device_count()):
        peak = torch.cuda.max_memory_allocated(i) / 1024 ** 3
        peak_reserved = torch.cuda.max_memory_reserved(i) / 1024 ** 3
        total = torch.cuda.mem_get_info(i)[1] / 1024 ** 3
        parts.append(f"cuda:{i} {peak:.1f} GiB peak ({peak_reserved:.1f} reserved) "
                     f"of {total:.0f}")
        if reset:
            torch.cuda.reset_peak_memory_stats(i)
    return "; ".join(parts)


def gpu_compute_apps(indices: list[int]) -> list[str]:
    """Human-readable rows for the processes holding memory on `indices`.

    Turns "your GPU is full" into something an operator can act on — which pid to
    look at or kill. nvidia-smi reports compute apps against a PCI bus id rather
    than a device index, so the bus ids are mapped back through --query-gpu.
    Best-effort: returns [] if either query fails, since this only ever decorates
    a message that is already being raised or logged.
    """
    try:
        bus_out = _nvidia_smi_csv("--query-gpu=index,gpu_bus_id")
        app_out = _nvidia_smi_csv(
            "--query-compute-apps=gpu_bus_id,pid,process_name,used_gpu_memory"
        )
    except RuntimeError:
        return []
    bus_to_idx: dict[str, int] = {}
    for line in bus_out.strip().splitlines():
        idx_str, _, bus = line.partition(",")
        bus_to_idx[bus.strip().lower()] = int(idx_str.strip())
    wanted = set(indices)
    rows: list[str] = []
    for line in app_out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        idx = bus_to_idx.get(parts[0].lower())
        if idx is None or idx not in wanted:
            continue
        rows.append(f"GPU {idx}: pid {parts[1]} ({parts[2]}) holding {parts[3]} MiB")
    return rows


def _idle_gpu_indices(max_used_gib: float) -> list[int]:
    """Physical GPU indices (nvidia-smi order) holding < `max_used_gib` GiB of VRAM.

    Queries `nvidia-smi`, which enumerates every physical GPU regardless of
    CUDA_VISIBLE_DEVICES. Raises RuntimeError if nvidia-smi is unavailable.
    """
    out = _nvidia_smi_csv("--query-gpu=index,memory.used")
    threshold_mib = max_used_gib * 1024
    idle: list[int] = []
    for line in out.strip().splitlines():
        idx_str, _, used_str = line.partition(",")
        if float(used_str.strip()) < threshold_mib:
            idle.append(int(idx_str.strip()))
    return idle


def select_open_gpus(n: int, max_used_gib: float = 4.0) -> list[int]:
    """Return the indices of the first `n` idle GPUs, in nvidia-smi order.

    "Idle" means the card is holding less than `max_used_gib` GiB of VRAM, so a
    fresh run lands on cards nobody else is using. Returns the lowest-numbered
    devices that pass the threshold. Does NOT reserve them — for concurrent
    instances use `reserve_open_gpus`, which additionally locks each card so two
    processes can't both select it before their VRAM registers.

    Call this BEFORE CUDA is initialised and pair it with CUDA_DEVICE_ORDER=
    PCI_BUS_ID so the returned nvidia-smi indices match the CUDA ordinals the
    launched processes will see. Raises RuntimeError if fewer than `n` idle GPUs
    exist.
    """
    open_gpus = _idle_gpu_indices(max_used_gib)
    if len(open_gpus) < n:
        raise RuntimeError(
            f"need {n} GPU(s) with <{max_used_gib:g} GiB used, but only found "
            f"{len(open_gpus)}: {open_gpus}"
        )
    return open_gpus[:n]


def reserve_open_gpus(n: int, max_used_gib: float = 4.0) -> list[int]:
    """Atomically select and RESERVE `n` idle GPUs across concurrent instances.

    Like `select_open_gpus`, but each returned card is held under a cross-process
    lock for this process's lifetime, so two pipeline processes started at once
    (two launch.py runs, or launch.py + a serve tool) can never both land on the
    same physical GPU. The used-memory heuristic alone races: a card just selected
    by another process shows no VRAM until its model finishes loading, so both would
    otherwise pick it. Skips any idle card a sibling already reserved.

    Call BEFORE CUDA is initialised, paired with CUDA_DEVICE_ORDER=PCI_BUS_ID.
    Raises RuntimeError if fewer than `n` unreserved idle GPUs exist.
    """
    reserved: list[int] = []
    for idx in _idle_gpu_indices(max_used_gib):
        if _acquire_named_lock(f"gpu_{idx}"):
            reserved.append(idx)
            if len(reserved) == n:
                return reserved
    # Not enough free-and-unreserved cards — release the ones we did grab so a
    # retry (or another process) can use them, then fail loudly.
    for idx in reserved:
        _release_named_lock(f"gpu_{idx}")
    raise RuntimeError(
        f"need {n} idle GPU(s) with <{max_used_gib:g} GiB used and not already "
        f"reserved by another pipeline process, but only found {len(reserved)}."
    )


def reserve_gpus(indices: list[int]) -> None:
    """Best-effort reserve explicitly-chosen GPUs so autoselecting siblings skip them.

    Used by the serve tools and launch.py's fixed-placement path, which take their
    cards from config (VLLM_GPUS / CUDA_VISIBLE_DEVICES) rather than autoselecting.
    Acquires the same process-lifetime locks `reserve_open_gpus` uses so a
    concurrent autoselecting run won't grab these cards. A card already held by
    another pipeline process is WARNED about (the operator pinned it explicitly, so
    we don't refuse) but still used.
    """
    for idx in indices:
        if not _acquire_named_lock(f"gpu_{idx}"):
            logger.warning(
                "GPU %d is already reserved by another pipeline process; using it "
                "anyway (it was pinned explicitly via VLLM_GPUS/CUDA_VISIBLE_DEVICES). "
                "Expect contention if both processes load models onto it.", idx,
            )


# ── Model-size routing ───────────────────────────────────────────────────────
# The 20B and 120B runs need very different GPU layouts (one card per role vs a
# multi-card block), and the size is only ever encoded in the model path. Every
# spelling of the same weights — "unsloth/gpt-oss-20b", "unsloth/gpt-oss-20b-BF16",
# "openai/gpt-oss-20b", a local "/models/gpt-oss-20b-BF16" checkout — must route
# identically, so callers key on the family below rather than on an exact string
# equality against one blessed repo id (which silently sent every other spelling
# down the 120B path).
_MODEL_FAMILY_RE = re.compile(r"(gpt-oss)[-_]?(\d+)\s*b", re.IGNORECASE)


def model_family(path: str) -> Optional[str]:
    """Normalised '<family>-<size>b' tag for a model path, or None if unrecognised.

    Matches on the basename-ish shape shared by HF repo ids and local checkout
    paths, ignoring precision/format suffixes (-BF16, -MXFP4) and casing, so all
    spellings of one model collapse to a single tag ("gpt-oss-20b").
    """
    m = _MODEL_FAMILY_RE.search(path or "")
    return f"{m.group(1).lower()}-{m.group(2)}b" if m else None


def model_billions(path: str) -> Optional[float]:
    """Parameter count in billions, from the size the model path spells, or None.

    The same digits model_family normalises ("…-120b" -> 120.0), read separately so
    a caller can size a layout — how many cards this model's weights need — rather
    than only route on the tag. Nominal, not exact (gpt-oss-120b is 116.8B params),
    which is the safe direction for a capacity estimate: it rounds up.
    """
    m = _MODEL_FAMILY_RE.search(path or "")
    return float(m.group(2)) if m else None


def bf16_weight_gib(path: str) -> Optional[float]:
    """GiB a BF16 copy of `path`'s weights occupies, from its nominal size, or None.

    Two bytes per parameter — the trainer and the merge both load BF16 (MXFP4, the
    gpt-oss release format, has no backward kernels), so this is the number that
    decides how many cards a load needs. 20B -> ~37 GiB, 120B -> ~224 GiB against
    the ~218 GiB the 120B actually measures.
    """
    billions = model_billions(path)
    return billions * 2e9 / 1024 ** 3 if billions else None


def same_model_family(a: str, b: str) -> bool:
    """True if `a` and `b` name the same model size, or either is unrecognised.

    Unrecognised paths return True (permissive): a custom/renamed checkout is not
    evidence of a mismatch, and refusing on it would block legitimate runs. Only a
    confident, recognised disagreement (20b vs 120b) is reported as a mismatch.
    """
    fam_a, fam_b = model_family(a), model_family(b)
    return fam_a is None or fam_b is None or fam_a == fam_b


def adapter_base_model(adapter_dir) -> Optional[str]:
    """The base model an adapter was trained against, per its adapter_config.json.

    PEFT records this at save time, which makes it the only trustworthy record of
    what a checkpoint on disk actually belongs to. Returns None for a directory that
    is not a PEFT adapter, or whose config is missing/unreadable.
    """
    cfg = Path(adapter_dir) / "adapter_config.json"
    try:
        return json.loads(cfg.read_text()).get("base_model_name_or_path")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def check_adapter_matches_base(adapter_dir, base_model: str, what: str) -> None:
    """Warn loudly when an adapter was trained on a different-size model than `base_model`.

    Serving an epoch checkpoint routes two independently-configured paths at the
    same weights, and nothing else cross-checks them: VLLM_MODEL/HF_MODEL_PATH come
    from the environment, while the adapter comes from a --epoch/--adapter argument.
    Point either one at the wrong run — the classic case being a 120B-trained
    adapter loaded onto the default 20B base, since VLLM_MODEL keeps its 20B default
    unless explicitly overridden — and the shapes disagree; vLLM's error names
    tensor dimensions, not the misconfiguration that caused them. Differing
    precision/format spellings of ONE model (BF16 base vs MXFP4 release) are the
    normal, intended case and pass quietly (see same_model_family).
    """
    trained_on = adapter_base_model(adapter_dir)
    if trained_on is None or same_model_family(trained_on, base_model):
        return
    logger.warning(
        "Adapter %s was trained on %r (%s), but %s is %r (%s) — these are different "
        "model sizes. Loading it will fail on mismatched tensor shapes, or serve "
        "nonsense if it does not. Set %s to the model this run trained against.",
        adapter_dir, trained_on, model_family(trained_on), what, base_model,
        model_family(base_model), what,
    )


def check_adapter_matches_lora_config(adapter_dir) -> None:
    """Warn when an adapter's LoRA shape can't be rebuilt from the current config.

    Re-merging an adapter (tools/serve_checkpoint) materialises a FRESH PEFT tree
    from `config` and loads the saved weights into it, so every rank in
    adapter_config.json has to match what this config would build. It need not:
    `sft` sizes its expert LoRA as LORA_RANK // num_experts (2 for the 20B) while
    the pipeline uses LORA_EXPERT_RANK (8 by default), and a run may simply have
    been trained with a different LORA_RANK. The load then dies deep inside
    torch's `load_state_dict` on "size mismatch for …lora_A…", which names tensor
    dimensions rather than the two config values that disagree. Say it up front —
    as a warning, not a refusal, since only the ranks that are actually adapted
    have to line up.
    """
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    try:
        saved = json.loads(cfg_path.read_text())
    except (OSError, json.JSONDecodeError, AttributeError):
        return
    mismatches = []
    if isinstance(saved.get("r"), int) and saved["r"] != config.LORA_RANK:
        mismatches.append(f"rank r={saved['r']} vs LORA_RANK={config.LORA_RANK}")
    expert_ranks = {v for v in (saved.get("rank_pattern") or {}).values() if isinstance(v, int)}
    if expert_ranks and config.LORA_EXPERT_RANK is not None \
            and expert_ranks != {config.LORA_EXPERT_RANK}:
        mismatches.append(
            f"per-expert rank {sorted(expert_ranks)} vs "
            f"LORA_EXPERT_RANK={config.LORA_EXPERT_RANK}"
        )
    saved_experts = bool(saved.get("target_parameters"))
    if saved_experts != bool(config.LORA_TARGET_EXPERTS):
        mismatches.append(
            f"expert targeting {'on' if saved_experts else 'off'} vs "
            f"LORA_TARGET_EXPERTS={int(bool(config.LORA_TARGET_EXPERTS))}"
        )
    if mismatches:
        logger.warning(
            "Adapter %s was trained with a different LoRA shape than this config "
            "builds (%s). Re-merging it will fail on mismatched tensor shapes — set "
            "those knobs to the values the run trained with, or serve that run's own "
            "pre-merged epoch_<N>_merged (SFT_KEEP_EPOCH_MERGED=1 keeps them).",
            adapter_dir, "; ".join(mismatches),
        )


def hf_merge_gpu_count() -> int:
    """How many GPUs an in-process BF16 load of config.HF_MODEL_PATH needs.

    serve_checkpoint materialises the full BF16 base to merge an adapter into it,
    and that load is sized by the MODEL, not by the serving tensor-parallel degree:
    the 20B BF16 base (~37 GiB) fits one 80 GB card, the 120B (~224 GiB) needs four,
    and reserving only VLLM_TENSOR_PARALLEL cards (2) OOMs the 120B merge at weight
    load, long before vLLM is reached.

    Computed as `ceil(weights / usable per-card VRAM)` from the model's nominal size
    and the driver's own report of the cards on this box — so it stays right on a box
    whose cards aren't 80 GB, and doesn't have to be re-derived from the width of a
    cap string (which it used to count the commas of, and which is now empty by
    default). Per-card usable VRAM is total minus HF_MIN_HEADROOM_GIB, matching what
    resolve_max_memory will actually hand the load; TOTAL rather than free because
    this runs before the cards are reserved, and the point of reserving is that they
    will be idle by the time the merge loads. An explicit HF_MAX_MEMORY still wins:
    the operator has then stated the block width themselves.
    """
    caps = [tok for tok in (config.HF_MAX_MEMORY or "").split(",") if tok.strip()]
    if caps:
        return len(caps)

    weights_gib = bf16_weight_gib(config.HF_MODEL_PATH)
    if weights_gib is None:
        logger.warning(
            "Cannot size the merge block for %s (unrecognised model size); assuming "
            "the %d-card trainer block.", config.HF_MODEL_PATH, _FIXED_TRAIN_GPU_COUNT,
        )
        return _FIXED_TRAIN_GPU_COUNT
    try:
        totals = [total for _, total in gpu_memory_gib().values()]
    except RuntimeError as exc:                       # no driver to ask
        logger.warning("Cannot query GPU capacity (%s); assuming the %d-card trainer "
                       "block for the merge.", exc, _FIXED_TRAIN_GPU_COUNT)
        return _FIXED_TRAIN_GPU_COUNT
    per_card = min(totals, default=0.0) - config.HF_MIN_HEADROOM_GIB
    if per_card <= 0:
        return _FIXED_TRAIN_GPU_COUNT
    return max(1, math.ceil(weights_gib / per_card))


def configure_serve_gpus(set_visible: bool = True, min_gpus: int = 1) -> list[int]:
    """Reserve the GPUs a serve tool will use, auto-selecting idle cards if unpinned.

    The serving tools (serve_epoch / serve_checkpoint) otherwise fall back to the
    fixed config.VLLM_GPUS default ("0,1"), so two of them on one box silently
    collide on the same cards. When the operator has NOT pinned GPUs (neither
    CUDA_VISIBLE_DEVICES nor VLLM_GPUS is in the environment), auto-select
    config.VLLM_TENSOR_PARALLEL idle cards with the same reserved idle-GPU finder
    launch.py uses (reserve_open_gpus), point config.VLLM_GPUS at them, and — when
    `set_visible` — pin CUDA_VISIBLE_DEVICES so an in-process HF load (the merge in
    serve_checkpoint) is confined to those cards rather than spreading across every
    visible GPU. When GPUs ARE pinned, honor the pin and just reserve those cards so
    a concurrent autoselecting run skips them.

    `min_gpus` is the number of cards the CALLER's in-process work needs, which can
    exceed the serving degree: serve_checkpoint's merge materialises the full BF16
    base (see hf_merge_gpu_count), and for the 120B that needs a multi-card block
    while vLLM itself serves the merged checkpoint on VLLM_TENSOR_PARALLEL cards.
    We therefore reserve max(tensor-parallel, min_gpus) cards and make ALL of them
    visible to this process, but point config.VLLM_GPUS at only the first
    tensor-parallel of them — the merge runs in a subprocess whose exit returns its
    VRAM before vLLM starts, so the two uses can overlap on the same cards. Under a pin, too few pinned cards is
    warned about rather than refused (the operator chose them explicitly).

    Returns the reserved physical GPU indices. Must run before CUDA is initialised;
    pairs the selection with CUDA_DEVICE_ORDER=PCI_BUS_ID so nvidia-smi indices
    match the CUDA ordinals the launched processes see.
    """
    tp = max(1, config.VLLM_TENSOR_PARALLEL)
    if "CUDA_VISIBLE_DEVICES" not in os.environ and "VLLM_GPUS" not in os.environ:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        n = max(tp, min_gpus)
        gpus = reserve_open_gpus(n, max_used_gib=4.0)
        gpu_csv = ",".join(str(g) for g in gpus)
        # vLLM gets exactly `tp` cards so --tensor-parallel-size and the card list
        # it is confined to always agree (vllm_server._serve_command derives the
        # degree from this list); any extra cards exist only for the merge.
        serve_csv = ",".join(str(g) for g in gpus[:tp])
        config.VLLM_GPUS = os.environ["VLLM_GPUS"] = serve_csv
        if set_visible:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_csv
        logger.info(
            "Auto-selected %d idle GPU(s): %s (vLLM will serve on %s).",
            n, gpu_csv, serve_csv,
        )
        return gpus

    # A pin is present. vLLM always runs on config.VLLM_GPUS (vllm_server.start pins
    # its subprocess to it), so that's the card list to reserve — but when the
    # operator pinned ONLY CUDA_VISIBLE_DEVICES, config.VLLM_GPUS still holds its
    # default ("0,1"), which is NOT what they asked for. Honor an explicit VLLM_GPUS
    # if given; otherwise take the cards from CUDA_VISIBLE_DEVICES and align
    # config.VLLM_GPUS to them so the reservation, the merge's visible devices, and
    # the vLLM server all target the operator's chosen cards.
    if "VLLM_GPUS" not in os.environ:
        config.VLLM_GPUS = os.environ["CUDA_VISIBLE_DEVICES"]
    gpus = [int(g) for g in config.VLLM_GPUS.split(",") if g.strip() != ""]
    if gpus:
        reserve_gpus(gpus)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if set_visible:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", config.VLLM_GPUS)
    # The pinned block may be narrower than the caller's in-process load needs (the
    # 120B merge on a 2-card pin OOMs at weight load). Say so up front rather than
    # letting it surface as an opaque CUDA OOM twenty minutes into loading weights.
    visible = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
    if set_visible and min_gpus > max(len(visible), 1):
        logger.warning(
            "This step needs ~%d GPU(s) for its in-process model load but only %d "
            "are pinned (%s); expect a CUDA OOM at weight load. Widen the pin, or "
            "unset CUDA_VISIBLE_DEVICES/VLLM_GPUS to auto-select idle cards.",
            min_gpus, len(visible), ",".join(visible) or "none",
        )
    return gpus


# Training cards the 20B auto-placement takes — always two, whether or not the run
# also serves. The student SHARDS over both (device_map="auto" + get_balanced_memory
# splits the ~39 GiB bf16 base ~20/20), which is the point: the per-card activation
# headroom is what lets a long example train, and the sharded student is a supported
# configuration end to end (see sft.model._keep_flex_attention_fused, which keeps
# FlexAttention on its fused kernel across shards).
_AUTOSELECT_TRAIN_GPUS = 2
# Trainer block for the non-20B path, mirroring launch.py's fallback. Four cards,
# leaving the other four for vLLM — the merged 120B checkpoint it restarts on at
# each epoch boundary is ~218 GiB and does not fit on fewer (see config.VLLM_GPUS).
_FIXED_TRAIN_GPUS = "4,5,6,7"
_FIXED_TRAIN_GPU_COUNT = len([g for g in _FIXED_TRAIN_GPUS.split(",") if g.strip()])
# Only this family gets the auto-placement; keyed on the family rather than a repo
# id for the reason launch.py's _GPU_AUTOSELECT_FAMILY spells out.
_AUTOSELECT_FAMILY = "gpt-oss-20b"


def configure_train_gpus(
    model_path: str,
    *,
    needs_vllm: bool = True,
    max_memory_env: str = "HF_MAX_MEMORY",
    full_finetune: bool = False,
) -> str:
    """Pin and reserve the GPUs a trainer will train (and optionally serve) on.

    The training-side counterpart to configure_serve_gpus, and the same placement
    launch.py gives the RL pipeline — factored out here so `sft` and `opsd` get it
    too instead of taking whatever GPUs happen to be visible (an HF student loaded
    with device_map="auto" across every card on the box lands on the very cards the
    vLLM server is using, and two concurrent runs OOM each other):

      * 20B (the default student): auto-place on the next idle cards — TWO for the
        trainer, which shards the student across both, plus one more for vLLM when
        the run serves, kept DISJOINT from the training pair.
      * anything else (e.g. the 120B): the fixed 4-7 trainer block, leaving
        VLLM_GPUS ("0,1,2,3") for serving — four cards each way, which is what the
        merged 120B checkpoint needs to be served at all (see config.VLLM_GPUS).

    An explicit CUDA_VISIBLE_DEVICES always wins; we then only reserve the pinned
    cards so an autoselecting sibling skips them.

    This function chooses the card BLOCK only, never the per-card memory ceilings:
    those are derived from the chosen cards' live VRAM when the model loads
    (resolve_max_memory), so a placement is sized by the hardware it landed on
    rather than by a cap string written for one box.

    `needs_vllm` is whether this run will actually start a server (rollouts, or
    serve-after-train). When it won't — an SFT run, whose held-out validation
    decodes from the live HF model rather than a served checkpoint — the 20B path
    takes the two training cards only, instead of holding a third idle one.

    Must be called before anything initialises CUDA, and pairs its selection with
    CUDA_DEVICE_ORDER=PCI_BUS_ID so the nvidia-smi indices it reserves match the
    CUDA ordinals the trainer and the vLLM subprocess see. Returns the HF
    max-memory spec the caller should train under — an operator's pin, passed
    through unchanged, or "" for "derive it from the cards at load time" — so a
    caller with its own config mirror can update it.
    """
    current_max_memory = os.environ.get(max_memory_env) or config.HF_MAX_MEMORY

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        _reserve_pinned_train_gpus(needs_vllm, "pinned")
        return current_max_memory

    if model_family(model_path) != _AUTOSELECT_FAMILY:
        os.environ["CUDA_VISIBLE_DEVICES"] = _FIXED_TRAIN_GPUS
        # Reserve both blocks so a concurrent autoselecting run (a 20B trainer or a
        # serve tool) won't pick these same cards.
        _reserve_pinned_train_gpus(needs_vllm, "fixed-layout")
        return current_max_memory

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # reserve_ (not select_) so two runs started at once can't both grab the same
    # "idle" card — a freshly-selected card shows no VRAM until its model loads, so a
    # plain nvidia-smi scan races. The locks are held for this process's lifetime.
    gpus = reserve_open_gpus(_AUTOSELECT_TRAIN_GPUS + (1 if needs_vllm else 0),
                             max_used_gib=4.0)
    if needs_vllm:
        vllm_gpu, *train_gpus = gpus
        # The vLLM subprocess reads these from config at launch (see
        # pipeline/vllm_server.py); one card means tensor-parallel 1. Mirrored to
        # os.environ so a child that re-imports pipeline.config agrees.
        config.VLLM_GPUS = os.environ["VLLM_GPUS"] = str(vllm_gpu)
        config.VLLM_TENSOR_PARALLEL = 1
        os.environ["VLLM_TENSOR_PARALLEL"] = "1"
    else:
        train_gpus = gpus

    # The student loads with device_map="auto", so it spreads over whatever is
    # visible; give it exactly the training pair. No cap string goes with it — both
    # cards were just verified idle, so the ceilings resolve_max_memory reads off
    # them at load time are the whole card minus the step's reserve.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in train_gpus)

    if full_finetune:
        logger.warning(
            "Full fine-tuning on %d training GPUs will OOM for 20B (it needs "
            "~168 GB); pin a wider CUDA_VISIBLE_DEVICES instead.",
            len(train_gpus),
        )
    logger.info(
        "20B auto-placement: trainer -> GPUs %s (sharded)%s — next idle cards, "
        "<4 GiB used.",
        ",".join(str(g) for g in train_gpus),
        f", vLLM -> GPU {gpus[0]}" if needs_vllm else "",
    )
    return current_max_memory


def _reserve_pinned_train_gpus(needs_vllm: bool, source: str) -> None:
    """Reserve the cards named by CUDA_VISIBLE_DEVICES (and VLLM_GPUS), as-is.

    Used both for an operator's explicit pin and for the fixed non-20B block —
    neither autoselects, so all this does is take the locks an autoselecting sibling
    checks. `source` only labels the log line.
    """
    specs = [os.environ["CUDA_VISIBLE_DEVICES"]]
    if needs_vllm:
        specs.append(config.VLLM_GPUS)
    # A pin may legally name UUIDs ("GPU-abc…") rather than indices; those can't be
    # matched against nvidia-smi indices, so reserve the numeric ones and leave the
    # rest unlocked rather than failing the run over an unparseable token.
    pinned = sorted({int(g) for spec in specs
                     for g in (t.strip() for t in spec.split(",")) if g.isdigit()})
    reserve_gpus(pinned)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    logger.info(
        "Using %s GPUs: trainer -> %s%s.",
        source, os.environ["CUDA_VISIBLE_DEVICES"],
        f", vLLM -> {config.VLLM_GPUS}" if needs_vllm else "",
    )


def parse_vllm_token_id(token: str) -> int:
    """Parse a vLLM "token_id:NNN" logprob token string into its integer id.

    With return_tokens_as_token_ids the server reports each token as
    "token_id:1234" instead of its decoded text, which is the only unambiguous
    way to align server-side tokens with the HF vocabulary. A plain decoded string
    here means the server ignored the flag (too old, or not enabled), so fail
    loudly rather than guess an id from text.
    """
    if isinstance(token, str) and token.startswith("token_id:"):
        try:
            return int(token.split(":", 1)[1])
        except ValueError:
            pass
    raise RuntimeError(
        f"vLLM logprob token {token!r} is not in 'token_id:N' form. Token-id "
        "alignment needs return_tokens_as_token_ids; ensure the vLLM server "
        "supports it (recent version) and accepts the request flag."
    )


def set_global_seed(seed: int) -> None:
    """Seed Python/NumPy/torch RNGs for reproducibility (no-op when seed < 0).

    Called once at run start (run_pipeline.main). This covers host-side
    randomness — Python's `random`, NumPy, and torch (including the currently
    unseeded fresh-LoRA init) — while the vLLM server seed (--seed) and the
    per-request `seed` on vLLM sampling payloads cover the served-model side.
    Deliberately does NOT enable torch.use_deterministic_algorithms: it can raise
    on unsupported CUDA kernels and slows training; opt in at the call site if
    you need bit-level determinism over speed.
    """
    if seed < 0:
        logger.info("Seeding disabled (SEED=%d); run is nondeterministic.", seed)
        return
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Global seed set to %d (Python/NumPy/torch).", seed)


def post_with_retry(
    url: str,
    *,
    headers: dict,
    payload: dict,
    timeout: int,
    label: str,
) -> requests.Response:
    """POST with exponential backoff on transient failures only.

    Retries network errors and RETRY_STATUSES (429/5xx). Any other HTTP error
    (400, 401, 403, …) is raised immediately — retrying those just burns time.

    The matching backend concurrency cap (see _concurrency_gate) is held for the
    whole call, including retry backoff, so concurrent callers never exceed the
    server's configured limit.

    Aborts with ShutdownRequested as soon as a shutdown signal is seen, rather
    than starting another attempt or sitting out a backoff — see SHUTDOWN. A
    request already on the wire still has to return (requests offers no
    cancellation), which is why the caller must also stop WAITING on these
    workers; the two together are what make teardown prompt.
    """
    with _concurrency_gate(label):
        for attempt in range(config.MAX_RETRIES + 1):
            # Checked before every attempt, not just the first: a worker that
            # grabbed its concurrency slot before the signal must not spend the
            # next few minutes retrying against a server we are tearing down.
            if SHUTDOWN.is_set():
                raise ShutdownRequested(
                    f"{label} aborted: shutdown requested before attempt "
                    f"{attempt + 1}/{config.MAX_RETRIES + 1}"
                )
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            except requests.RequestException as exc:
                if attempt < config.MAX_RETRIES:
                    delay = config.RETRY_BASE_DELAY * (2**attempt)
                    logger.warning(
                        "%s network error (attempt %d/%d): %s — retrying in %.1fs",
                        label, attempt + 1, config.MAX_RETRIES + 1, exc, delay,
                    )
                    interruptible_sleep(delay)
                    continue
                raise RuntimeError(
                    f"{label} failed after {config.MAX_RETRIES + 1} attempts: {exc}"
                ) from exc

            if resp.status_code in RETRY_STATUSES and attempt < config.MAX_RETRIES:
                delay = config.RETRY_BASE_DELAY * (2**attempt)
                logger.warning(
                    "%s HTTP %d (attempt %d/%d) — retrying in %.1fs",
                    label, resp.status_code, attempt + 1, config.MAX_RETRIES + 1, delay,
                )
                interruptible_sleep(delay)
                continue

            if not resp.ok:
                raise RuntimeError(
                    f"{label} HTTP {resp.status_code}: {resp.text[:500]}"
                )
            return resp

    raise RuntimeError(f"{label} exhausted {config.MAX_RETRIES + 1} attempts.")


def parse_chat_message(data: dict, *, label: str, allow_truncated: bool = False) -> dict:
    """Validate an OpenAI-style chat completion body and return its message parts.

    Returns {"content": str | None, "reasoning": str}. Reasoning models
    (gpt-oss et al.) emit their chain-of-thought on a separate channel that
    providers return as message.reasoning_content (message.reasoning on some
    versions), leaving message.content with little beyond the final answer — so
    a content-only read silently drops the reasoning.

    Both OpenRouter and vLLM can return HTTP 200 with an error body, an empty
    message, or a length-truncated completion — all of which would otherwise
    surface later as confusing KeyError/AttributeError crashes.

    `allow_truncated` keeps a finish_reason='length' response instead of raising.
    The judges/hint models (openrouter_call) want the default: a cut-off grade or
    a half-written JSON hint is unusable. The training rollout wants the opposite —
    a student that ran past its budget is the item most worth training on, and its
    reasoning is intact even when no final answer was reached (see
    inference_script.run_inference).
    """
    if "error" in data:
        raise RuntimeError(f"{label} returned an error body: {str(data['error'])[:500]}")
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"{label} response missing choices: {str(data)[:500]}") from exc

    if choice.get("finish_reason") == "length" and not allow_truncated:
        raise RuntimeError(
            f"{label} completion was truncated at the max_tokens limit "
            f"(finish_reason='length'); refusing to use a cut-off response."
        )

    message = choice.get("message") or {}
    content = message.get("content")
    reasoning = (message.get("reasoning_content") or message.get("reasoning") or "").strip()
    if not content and not reasoning:
        raise RuntimeError(f"{label} returned an empty message: {str(choice)[:500]}")
    return {"content": content, "reasoning": reasoning}


def parse_chat_completion(data: dict, *, label: str) -> str:
    """Validate an OpenAI-style chat completion body and return the content.

    Content-only view of parse_chat_message; raises if the message carried no
    content (even when it carried reasoning).
    """
    content = parse_chat_message(data, label=label)["content"]
    if content is None:
        raise RuntimeError(f"{label} returned null content: {str(data)[:500]}")
    return content


def openrouter_call(
    messages: list[dict],
    model: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    """Synchronous OpenRouter API call with exponential-backoff retry.

    Returns the assistant message content string.
    Raises RuntimeError on exhausted retries, non-retryable HTTP errors,
    error bodies, null content, or length-truncated completions.
    """
    if not config.OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")

    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = post_with_retry(
        config.OPENROUTER_URL, headers=headers, payload=payload,
        timeout=120, label="OpenRouter",
    )
    return parse_chat_completion(resp.json(), label="OpenRouter")


def openrouter_call_with_reasoning(
    messages: list[dict],
    model: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: int = 300,
) -> dict:
    """openrouter_call, but keeping the native reasoning channel.

    Returns {"content": str, "reasoning": str} — for reasoning-model teachers
    the chain-of-thought arrives on message.reasoning(_content) rather than in
    the content, and callers that need the full trace (SFT distillation) must
    see both halves. Same retry/validation semantics as openrouter_call.
    """
    if not config.OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")

    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = post_with_retry(
        config.OPENROUTER_URL, headers=headers, payload=payload,
        timeout=timeout, label="OpenRouter",
    )
    message = parse_chat_message(resp.json(), label="OpenRouter")
    return {"content": message["content"] or "", "reasoning": message["reasoning"]}


def vllm_call(
    messages: list[dict],
    model: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    """Synchronous vLLM chat-completion call with exponential-backoff retry.

    Mirrors openrouter_call but targets the self-hosted, OpenAI-compatible vLLM
    endpoint (config.VLLM_BASE_URL) and is gated by the vLLM concurrency cap.
    Returns the assistant message content string; raises RuntimeError on
    exhausted retries, non-retryable HTTP errors, error bodies, null content, or
    length-truncated completions.
    """
    url = f"{config.VLLM_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.VLLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = post_with_retry(
        url, headers=headers, payload=payload, timeout=300, label="vLLM",
    )
    return parse_chat_completion(resp.json(), label="vLLM")
