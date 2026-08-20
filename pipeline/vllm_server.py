"""Pipeline-owned vLLM lifecycle: start, wait-until-ready, restart, stop.

Two weight-refresh strategies, keyed on config.LORA_TARGET_EXPERTS:

  • Expert LoRA (LORA_TARGET_EXPERTS=1): gpt-oss packs its MoE experts as fused 3-D
    parameters with no per-expert modules, so a trained expert LoRA cannot be
    hot-swapped into a running vLLM server (vLLM has nothing to attach the adapter
    to). The pipeline serves the base model for the first epoch's rollouts, then at
    each epoch boundary restarts the server on a freshly merged full checkpoint
    (optimization_script.save_merged_model) via restart().

  • Non-expert LoRA (LORA_TARGET_EXPERTS=0): the adapter targets ordinary nn.Linear
    modules (attention/MLP projections) that vLLM can attach a LoRA to, so the
    server is launched with --enable-lora and the runtime-update endpoints. At each
    epoch boundary the saved adapter is hot-loaded with load_adapter() — no restart,
    no full-checkpoint merge. Requests then target the adapter's served name.

The base model is always launched with `--served-model-name config.VLLM_MODEL`. In
the merge+restart path the model name in request payloads stays constant as the
underlying weights swap; in the hot-load path callers switch to the adapter name
(inference_script.set_active_model) once it is loaded.
"""
import atexit
import contextlib
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from typing import Optional
from urllib.parse import urlparse, urlunparse

import requests

from . import config
from . import endpoint
from . import utils
from .utils import logger

_proc: Optional[subprocess.Popen] = None
# Process-group id of the running server, recorded at launch. Kept separately from
# _proc because os.getpgid(pid) stops working the moment the launcher is reaped —
# and the case that matters most (the launcher died on its own, leaving engine and
# worker processes alive on the GPUs) is exactly when it has already been reaped by
# a poll(). Holding the pgid means stop() can still drain those survivors.
_pgid: Optional[int] = None
# Set once, on this instance's first start(), when the configured port was already
# taken and we moved to a free one; keeps the port stable across restart()s.
_port_assigned = False


def _host_port() -> tuple[str, str]:
    """Derive the bind host/port for `vllm serve` from config.VLLM_BASE_URL."""
    parsed = urlparse(config.VLLM_BASE_URL)
    return parsed.hostname or "127.0.0.1", str(parsed.port or 8000)


def _with_port(url: str, port: int) -> str:
    """Return `url` with its port replaced by `port` (host/scheme/path preserved)."""
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    return urlunparse(parsed._replace(netloc=f"{host}:{port}"))


def _port_bindable(host: str, port: int) -> bool:
    """True if nothing is currently bound to (host, port) — i.e. we could bind it.

    Catches foreign (non-pipeline) servers occupying the port that our per-instance
    port reservation (utils.reserve_port) wouldn't know about.

    SO_REUSEADDR mirrors what vLLM's own uvicorn listener sets, so this answers the
    question that actually matters: "would `vllm serve` bind here?" Without it, the
    TIME_WAIT connections a just-stopped server leaves behind make the port look
    occupied for ~60s — which would strand wait_down() until it timed out, and drift
    each restart onto a new port even though vLLM would have bound the old one fine.
    A live LISTENing socket still blocks the bind (that needs SO_REUSEPORT), so real
    occupancy is still detected.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _assign_instance_port() -> None:
    """Pin this instance to a free vLLM port, rewriting config.VLLM_BASE_URL if needed.

    Multiple pipeline processes on one box default to the SAME VLLM_BASE_URL port,
    so without this the second `vllm serve` dies on an address-in-use bind — or,
    worse, the content-blind readiness probe (is_ready) latches onto the sibling's
    server and the run silently trains against the wrong weights. Starting from the
    configured port, pick the first port that is BOTH claimable across pipeline
    instances (utils.reserve_port) AND actually free to bind (no foreign server),
    then rewrite config.VLLM_BASE_URL (+ os.environ, for subprocesses that re-import
    config) so every consumer targets this instance's own server.

    Runs once per process: the reserved port is held for the process lifetime, and
    restart() stops/starts our own server on the same port (it reads as free once we
    stop, and _port_assigned keeps us from moving).
    """
    global _port_assigned
    if _port_assigned:
        return
    host, port = _host_port()
    base = int(port)
    for candidate in range(base, base + 128):
        if not utils.reserve_port(candidate):
            continue  # another pipeline instance owns this port
        if _port_bindable(host, candidate):
            if candidate != base:
                new_url = _with_port(config.VLLM_BASE_URL, candidate)
                logger.warning(
                    "vLLM port %d on %s is already in use; moving this instance to "
                    "free port %d (%s) so it doesn't collide with the other server.",
                    base, host, candidate, new_url,
                )
                config.VLLM_BASE_URL = new_url
            # Exported unconditionally, not only when the port moved: a consumer
            # that reads VLLM_BASE_URL should see the endpoint we actually settled
            # on in every case, including "kept the configured port".
            os.environ["VLLM_BASE_URL"] = config.VLLM_BASE_URL
            # os.environ only reaches children we spawn from here on. Anything
            # started out of band — an operator's second shell, a benchmark queued
            # before us — reads the repo-root .env and gets the pre-shift port back,
            # with nothing to signal it went to the wrong server. Publish the
            # resolved endpoint so those callers can look it up instead.
            endpoint.publish(
                config.VLLM_BASE_URL,
                model=config.VLLM_MODEL,
                gpus=config.VLLM_GPUS,
            )
            # Discovery treats an endpoint as dead once its port reservation is
            # gone, so a crash leaves nothing misleading behind either way; this
            # just keeps the directory tidy on the ordinary exit path.
            atexit.register(endpoint.unpublish, config.VLLM_BASE_URL)
            _port_assigned = True
            return
        # We hold the lock but a foreign server owns the socket — release and advance.
        utils.release_port(candidate)
    raise RuntimeError(
        f"no free vLLM port found in [{base}, {base + 128}) on {host} for this "
        f"instance; is the box saturated with servers?"
    )


def _tensor_parallel_size() -> int:
    """Tensor-parallel degree to serve with — always the count of config.VLLM_GPUS.

    VLLM_TENSOR_PARALLEL defaults to 2, sized for the 120B layout, while VLLM_GPUS
    is what actually reaches the subprocess as CUDA_VISIBLE_DEVICES. The two drift
    apart constantly: a 20B run pinned to one card (VLLM_GPUS=3) but left on the
    default degree makes vLLM abort at startup ("engine requires 2 GPUs, 1 visible"),
    and the reverse silently wastes a reserved card. The visible cards are the
    ground truth — a degree that doesn't match them cannot work — so derive it and
    warn when the configured value disagreed.
    """
    gpus = [g for g in config.VLLM_GPUS.split(",") if g.strip()]
    if not gpus:
        return max(1, config.VLLM_TENSOR_PARALLEL)
    if len(gpus) != config.VLLM_TENSOR_PARALLEL:
        logger.warning(
            "VLLM_TENSOR_PARALLEL=%d but VLLM_GPUS names %d card(s) (%s); serving "
            "with tensor-parallel-size %d to match the GPUs vLLM can actually see.",
            config.VLLM_TENSOR_PARALLEL, len(gpus), config.VLLM_GPUS, len(gpus),
        )
    return len(gpus)


# Exec'd in the child instead of `vllm` directly: arms PR_SET_PDEATHSIG, then
# replaces itself with the real serve command (execvp keeps the pid/pgid we
# recorded, so no extra process joins the tree).
#
# stop() can only reap the server if THIS process is alive to call it. A SIGKILL,
# a segfault, or the OOM-killer runs no atexit hook and no `finally`, so the whole
# vLLM session — launcher, engine core, TP workers — is re-parented to init and
# keeps every byte of its VRAM plus the bound port, forever. Measured on this box:
# `kill -9` of the pipeline left 73.8 GiB allocated on the card and port 8000
# LISTENing until the orphans were killed by hand, which is exactly the "GPU locks
# up / memory stays allocated after a dirty exit" failure.
#
# PR_SET_PDEATHSIG makes the KERNEL deliver SIGTERM to the launcher the moment its
# parent dies, by any means — the one cleanup path that survives an uncatchable
# signal. The launcher then runs vLLM's own graceful shutdown, which tears down the
# engine and frees the GPUs. Set from a fresh exec rather than Popen's preexec_fn,
# which is documented as unsafe in a process that has other threads running (the
# pipeline's epoch windows do).
#
# The flag is cleared by execve only for set-uid/set-gid binaries, so it survives
# the execvp below. The getppid() check closes the race where the parent died
# between fork and prctl, leaving nobody to signal us.
_PDEATHSIG_PRELUDE = """
import ctypes, os, signal, sys
try:
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0)
except Exception:
    pass
if os.getppid() == 1:
    os._exit(0)
os.execvp(sys.argv[1], sys.argv[1:])
"""


def _serve_command(model_path: str) -> list[str]:
    host, port = _host_port()
    cmd = [
        "vllm", "serve", model_path,
        "--host", host,
        "--port", port,
        "--served-model-name", config.VLLM_MODEL,
        "--tensor-parallel-size", str(_tensor_parallel_size()),
        "--gpu-memory-utilization", str(config.VLLM_GPU_MEM_UTIL),
        "--max-model-len", str(config.VLLM_MAX_MODEL_LEN),
        "--max-logprobs", str(config.VLLM_MAX_LOGPROBS),
    ]
    # Seed the server's sampling RNG for reproducibility (skip when disabled).
    if config.SEED >= 0:
        cmd += ["--seed", str(config.SEED)]
    # Non-expert LoRA is hot-loaded into the live server instead of merged into a
    # full checkpoint; enable LoRA serving and size the slot for our adapter rank.
    if not config.LORA_TARGET_EXPERTS:
        # max-loras 2 leaves room to hot-load the new adapter before unloading the
        # previous one, so a load failure never destroys the still-serving adapter.
        cmd += [
            "--enable-lora",
            "--max-lora-rank", str(config.LORA_RANK),
            "--max-loras", "2",
        ]
    if config.VLLM_SERVE_EXTRA_ARGS.strip():
        cmd.extend(shlex.split(config.VLLM_SERVE_EXTRA_ARGS))
    return cmd


def is_alive() -> bool:
    """True while the managed vLLM subprocess exists and has not exited.

    Lets a serving loop react to the engine dying (crash during inference, OOM,
    killed worker) instead of blocking forever — the caller can then tear down and
    release its GPU reservation rather than sitting idle on a dead server.
    """
    return _proc is not None and _proc.poll() is None


def is_ready() -> bool:
    """True if the server is actually live and serving, not just answering on the port.

    Checks /health, which vLLM answers 200 only while its engine is up — unlike
    /models, whose 200 merely echoes the configured model name and so keeps
    answering from a zombie API front-end whose engine has already died (freeing
    its GPUs). wait_ready() must not latch onto such a corpse, so /health is the
    gate. Falls back to /models only if /health is absent (older vLLM).

    Note this answers "is a healthy server reachable", NOT "is the port free" —
    a server that is bound but unhealthy reads as not-ready here while still
    owning the socket. wait_down() must therefore test bindability, not this.
    """
    # Once we own a subprocess and it has exited, no 200 on this port can be ours:
    # it is an orphaned or foreign vLLM serving weights we know nothing about, and
    # accepting it would silently run the epoch against the wrong model. Checked
    # ahead of BOTH probes below, not just the /models fallback.
    if _proc is not None and _proc.poll() is not None:
        return False
    base = config.VLLM_BASE_URL.rstrip("/")
    # /health is mounted at the server ROOT (host:port/health), not under the
    # OpenAI /v1 path that /models lives under — derive the origin from the URL.
    parsed = urlparse(config.VLLM_BASE_URL)
    health_url = urlunparse(parsed._replace(path="/health", params="", query="", fragment=""))
    headers = {"Authorization": f"Bearer {config.VLLM_API_KEY}"}
    try:
        health = requests.get(health_url, headers=headers, timeout=5)
        if health.status_code == 200:
            return True
        if health.status_code != 404:
            return False
        # /health missing (older vLLM): fall back to /models.
        resp = requests.get(f"{base}/models", headers=headers, timeout=5)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def wait_ready(timeout: Optional[int] = None) -> None:
    """Block until the server is ready, or raise RuntimeError on timeout/exit.

    is_ready() cannot tell WHOSE server answered, only that a healthy one did — so
    two things keep us from latching onto somebody else's. start() pins an
    instance-private port that was free the instant before launch, and both this
    loop and is_ready() refuse a 200 once our own subprocess has exited. If we lost
    a same-port race the dead child is detected (bind failure is near-instant, well
    before any winner finishes loading) rather than silently accepting the winner's
    server and training against its weights.
    """
    deadline = time.time() + (timeout or config.VLLM_STARTUP_TIMEOUT)
    while time.time() < deadline:
        if _proc is not None and _proc.poll() is not None:
            raise RuntimeError(
                f"vLLM server exited during startup (code {_proc.returncode}). "
                f"Check its stderr / the serve command."
            )
        if is_ready():
            logger.info("vLLM server is ready at %s.", config.VLLM_BASE_URL)
            return
        time.sleep(5)
    raise RuntimeError(
        f"vLLM server did not become ready within "
        f"{timeout or config.VLLM_STARTUP_TIMEOUT}s."
    )


def wait_down(timeout: int = 120) -> None:
    """Block until the server port is free to bind again.

    Gates restart(): the replacement `vllm serve` cannot come up while anything
    still holds the socket, so we wait for the port itself to be claimable rather
    than for a *readiness* probe to go quiet. Those are different questions, and
    the difference is exactly the failure this guards. A vLLM whose engine has died
    but whose API front-end is still bound reports NOT ready (is_ready checks
    /health) while continuing to own the port — so gating on is_ready() would let
    restart() proceed into an immediate "address already in use" death, or, worse,
    have the new server land on a port the corpse is still answering on.

    If the port never frees, a foreign or unkillable server owns it: fail loudly
    rather than start a replacement that cannot bind.
    """
    host, port = _host_port()
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_bindable(host, int(port)):
            return
        time.sleep(2)
    raise RuntimeError(
        f"Port {port} on {host} is still bound {timeout}s after stopping the managed "
        f"vLLM process — an orphaned or foreign server owns it (and likely still "
        f"holds its GPUs). Refusing to continue: a replacement server could not bind, "
        f"and the run would otherwise serve stale weights."
    )


def wait_for_gpu_memory(timeout: float = 120.0, poll: float = 5.0) -> None:
    """Block until every serving GPU has the VRAM `vllm serve` is about to demand.

    vLLM sizes its KV cache as a FRACTION OF THE WHOLE CARD, not of what is free:
    each worker checks `free < gpu_memory_utilization * total` at init and aborts
    with "Free memory on device cuda:N (58.75/79.25 GiB) on startup is less than
    desired GPU memory utilization (0.9, 71.33 GiB)". So anything still holding a
    few GiB on a card — a server that was Ctrl-C'd moments ago and is still
    draining, an orphaned engine, a sibling job, an in-process HF load whose
    allocator has not returned its cache — kills the launch outright.

    Two failures made that error hard to act on, and this addresses both. The
    transient case (a predecessor mid-teardown) is now simply waited out, because
    the GPUs come back within seconds. The persistent case fails with the numbers
    AND the pids actually holding the memory, instead of leaving the operator to
    match vLLM's CUDA-ordinal-relative message against nvidia-smi by hand.

    The check is vLLM's, computed off nvidia-smi rather than torch.cuda.mem_get_info
    — whose "total" is ~0.75 GiB smaller (driver-reserved), so our threshold comes
    out marginally STRICTER than the one the worker will apply, which happens to
    cover the ~0.5 GiB CUDA context the worker creates before measuring. Even at
    VLLM_GPU_MEM_UTIL=0.99 an empty card still passes, so this does not invent
    failures — but it is a gate, not a guarantee: the point is to catch the multi-GiB
    occupant, not to arbitrate the last few hundred MiB.

    A box without nvidia-smi (or a card nvidia-smi does not report) skips the gate
    rather than blocking a launch that might well have worked.
    """
    gpus = [int(g) for g in config.VLLM_GPUS.split(",") if g.strip()]
    if not gpus:
        return
    need_frac = config.VLLM_GPU_MEM_UTIL
    deadline = time.time() + timeout
    warned = False
    while True:
        try:
            mem = utils.gpu_memory_gib(gpus)
        except RuntimeError as exc:
            logger.debug("Skipping the pre-launch VRAM check: %s", exc)
            return
        short = {
            idx: (free, total) for idx, (free, total) in mem.items()
            if free < need_frac * total
        }
        if not short:
            return
        detail = "; ".join(
            f"GPU {idx}: {free:.2f} GiB free of {total:.2f} GiB, needs "
            f"{need_frac * total:.2f} GiB"
            for idx, (free, total) in sorted(short.items())
        )
        if time.time() >= deadline:
            occupants = utils.gpu_compute_apps(sorted(short))
            raise RuntimeError(
                f"vLLM claims {need_frac:g}× the TOTAL VRAM of every card it serves "
                f"on, and {len(short)} of its GPUs ({config.VLLM_GPUS}) did not free "
                f"up within {timeout:g}s — {detail}. "
                + (f"Still holding memory — {'; '.join(occupants)}. "
                   if occupants else
                   "nvidia-smi reports no compute apps on them, so the memory is "
                   "held by a process that has not exited yet or by a dead one the "
                   "driver has not reaped. ")
                + "Refusing to start a server that would abort at init; free those "
                  "GPUs (or lower VLLM_GPU_MEM_UTIL) and retry."
            )
        if not warned:
            logger.warning(
                "Waiting up to %.0fs for GPU memory to be released before starting "
                "vLLM — %s.", timeout, detail,
            )
            warned = True
        time.sleep(poll)


def start(model_path: str, wait: bool = True) -> None:
    """Launch the vLLM server on `model_path` (a HF id or a local checkpoint dir).

    The subprocess runs in its own session/process group so stop() can signal the
    whole tree, with CUDA_VISIBLE_DEVICES restricted to config.VLLM_GPUS.
    """
    global _proc
    if _proc is not None and _proc.poll() is None:
        raise RuntimeError("vLLM server already running; call restart() instead.")
    # Pin a free, instance-private port BEFORE building the serve command so a
    # sibling pipeline process on other GPUs doesn't collide on the port (or, worse,
    # get silently latched onto by our readiness probe). Confirmed free just below,
    # so any 200 wait_ready() then sees on this port is our own server.
    _assign_instance_port()
    cmd = _serve_command(model_path)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=config.VLLM_GPUS)
    # run_pipeline.py sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for the
    # HF process before importing torch. That setting leaks into this inherited
    # env, but expandable_segments breaks vLLM (CUDA "invalid argument"). vLLM runs
    # in its own process and manages its own allocator, so drop the key and let
    # vLLM use its default.
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    # Runtime LoRA load/unload endpoints are gated behind this env flag; only the
    # hot-load path (non-expert LoRA) needs them.
    if not config.LORA_TARGET_EXPERTS:
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
    # Pin the date gpt-oss carries in its harmony system message. vLLM otherwise
    # calls datetime.now() while building EVERY request's prompt, so a server that
    # crosses midnight starts conditioning on a different system message mid-run —
    # and the training-side render (pipeline.prompts) would disagree with it from
    # then on. One value for the whole run, shared by both sides.
    env["VLLM_SYSTEM_START_DATE"] = config.SYSTEM_START_DATE
    # `vllm` is resolved through PATH by the prelude's execvp. A pipeline started
    # with an absolute venv interpreter (nohup, systemd, cron, a subprocess that
    # scrubbed the environment) often does NOT have the venv's bin dir on PATH,
    # and the server then dies at spawn with a bare FileNotFoundError. The
    # interpreter running us is by definition next to the `vllm` we want.
    env["PATH"] = os.pathsep.join(
        [os.path.dirname(sys.executable), env.get("PATH", os.defpath)]
    )
    # Last thing before the launch, so it measures the state the workers will see:
    # confirm the cards actually have the VRAM vLLM is about to claim. A predecessor
    # still draining is waited out; a persistent occupant fails here, naming pids,
    # instead of inside a worker as vLLM's free-memory abort.
    wait_for_gpu_memory()
    logger.info("Starting vLLM (GPUs=%s): %s", config.VLLM_GPUS, " ".join(cmd))
    global _pgid
    # start_new_session isolates the tree from terminal Ctrl-C (stop() signals it
    # explicitly instead); the prelude arms PR_SET_PDEATHSIG so it still dies with
    # us when we are killed in a way that runs no cleanup at all.
    _proc = subprocess.Popen(
        [sys.executable, "-c", _PDEATHSIG_PRELUDE, *cmd],
        env=env, start_new_session=True,
    )
    # start_new_session makes the child a session/group leader, so its pgid equals
    # its pid. Record it now: once the launcher is reaped we can no longer look it
    # up, and it is the handle stop() needs to reach the engine/worker processes.
    _pgid = _proc.pid
    if wait:
        try:
            wait_ready()
        except BaseException:
            # A server that never became ready is still a server: on a startup
            # TIMEOUT it is alive and part-way through loading weights, holding
            # most of its GPU memory. Leaving it for the caller's error path to
            # notice means a retry (or a fallback to another model) OOMs against
            # VRAM nothing is using. Tear it down here so the failure is clean and
            # the GPUs are actually free by the time the exception surfaces.
            # BaseException, not Exception: a Ctrl-C or a signal-driven SystemExit
            # during the (many-minute) weight load is the likeliest way out of this
            # wait, and it must not be the one path that leaks the server.
            stop()
            raise


@contextlib.contextmanager
def _deferred_interrupts():
    """Ignore SIGINT/SIGTERM for the duration of the block, then restore handlers.

    vLLM teardown is slow — SIGTERM triggers a graceful shutdown that tears down
    tensor-parallel workers and releases GPU memory, taking many seconds. During
    that window an impatient second Ctrl-C would raise KeyboardInterrupt *inside*
    stop() (usually itself invoked from an atexit handler after the first Ctrl-C),
    unwinding before the SIGKILL escalation ever runs and leaving vLLM orphaned,
    still holding VRAM. Masking the signals guarantees the SIGTERM→wait→SIGKILL
    sequence runs to completion no matter how many times the user hits Ctrl-C.

    Signal handlers can only be (re)installed from the main thread; off the main
    thread this is a no-op, which is fine — the interrupt-storm race only exists on
    the main thread, where signals are delivered.
    """
    try:
        previous = {
            sig: signal.signal(sig, signal.SIG_IGN)
            for sig in (signal.SIGINT, signal.SIGTERM)
        }
    except (ValueError, OSError):
        # Not on the main thread (or handler can't be set); nothing to defer.
        yield
        return
    try:
        yield
    finally:
        for sig, handler in previous.items():
            with contextlib.suppress(ValueError, OSError, TypeError):
                signal.signal(sig, handler)


def _pgroup_alive(pgid: int) -> bool:
    """True while any process remains in group `pgid` (signal 0 = existence probe)."""
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists but isn't ours to signal — treat as alive rather than assume gone.
        return True


def _wait_pgroup_gone(pgid: int, timeout: float) -> bool:
    """Poll until process group `pgid` is empty; True if it drained within `timeout`."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pgroup_alive(pgid):
            return True
        time.sleep(0.5)
    return not _pgroup_alive(pgid)


def _reap_process_group(pgid: int, launcher_pid: int) -> None:
    """Ensure no process is left in `pgid` — the engine/worker procs holding the GPUs.

    Called after the launcher has been waited on. The group is the vLLM subprocess's
    own session (start_new_session=True), so it contains only vLLM's tree and never
    this process — killing it is always safe. Gives the group a grace period to
    finish its own graceful teardown (workers releasing NCCL/CUDA contexts takes a
    few seconds), then SIGKILLs whatever remains. Never raises: a stop() that failed
    to fully clean up must still return so the caller can proceed to wait_down(),
    which is the authoritative "is this port actually usable" check.

    The kill targets a pgid recorded at launch, and only after observing the group
    still occupied during this call — so reaching it requires the pid to have been
    recycled into a new group leader within the same teardown, which the kernel's
    sequential pid allocation makes a non-issue over these timescales.
    """
    if _wait_pgroup_gone(pgid, timeout=30):
        return
    logger.warning(
        "vLLM launcher (pid %d) exited but its engine/worker processes are still "
        "alive in process group %d, still holding GPU memory; killing the group.",
        launcher_pid, pgid,
    )
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(pgid, signal.SIGKILL)
    if not _wait_pgroup_gone(pgid, timeout=30):
        logger.error(
            "Process group %d survived SIGKILL; VRAM on GPUs %s may stay occupied "
            "until those processes are cleared manually (check `nvidia-smi`).",
            pgid, config.VLLM_GPUS,
        )


def stop() -> None:
    """Terminate the managed server and wait for its whole process GROUP to exit.

    Waiting on process exit is what releases the GPUs (CUDA frees on teardown) —
    but waiting on the direct child alone is not enough. vLLM's engine core and its
    tensor-parallel workers are separate processes, and they are the ones actually
    holding the VRAM. When the launcher dies first (especially after SIGKILL) those
    children are re-parented to init and keep their GPU allocations, so a stop()
    that returned as soon as `_proc` was reaped left tens of GiB pinned and the next
    start() OOMed on cards that looked free to the pipeline. We signal the group,
    reap our child, then poll until the group is genuinely empty, escalating to
    SIGKILL for stragglers.

    The whole teardown runs under _deferred_interrupts() so a repeated Ctrl-C can't
    abort it midway and orphan a VRAM-holding server. `_proc` is always cleared,
    even if the process is unkillable within the timeout, so a wedged teardown
    can't leave restart() crashing on a stale handle — restart() relies on
    wait_down() to confirm the port is actually free.
    """
    global _proc, _pgid
    if _proc is None or _proc.poll() is not None:
        # The launcher is already gone — but that is precisely the case where the
        # engine/worker processes it spawned are still running (an engine crash or
        # OOM takes down the launcher first). Drain the recorded group before
        # dropping the handle, or their VRAM stays allocated for good.
        if _pgid is not None:
            with _deferred_interrupts():
                _reap_process_group(_pgid, _proc.pid if _proc is not None else _pgid)
        _proc = None
        _pgid = None
        return
    pid = _proc.pid
    logger.info("Stopping vLLM server (pid %d) …", pid)
    with _deferred_interrupts():
        try:
            pgid = _pgid if _pgid is not None else os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                _proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                logger.warning("vLLM did not exit on SIGTERM; sending SIGKILL.")
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(pgid, signal.SIGKILL)
                try:
                    _proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    logger.error(
                        "vLLM (pid %d) did not exit even after SIGKILL; it may still "
                        "be holding GPUs.", pid,
                    )
            _reap_process_group(pgid, pid)
        except ProcessLookupError:
            pass
        finally:
            _proc = None
            _pgid = None


_handlers_installed = False


def install_shutdown_handlers() -> None:
    """Make every exit path tear the server down, not just the tidy ones.

    stop() only runs if something calls it. A plain `try/finally` covers a normal
    return and Ctrl-C, but NOT the signals that actually kill long-running serve
    jobs: SIGTERM (`kill <pid>`, an orchestrator, a scheduler) and SIGHUP (closing
    the terminal) have a default disposition that terminates the process outright,
    running neither `finally` blocks nor atexit hooks. The vLLM subprocess lives in
    its own session, so it never receives that signal itself — it is simply orphaned
    with every GPU it had still allocated, and the port still bound. Converting those
    signals into a clean sys.exit() lets the registered atexit hook run stop().

    SIGINT is handled here too, even though its default already raises
    KeyboardInterrupt. The handler runs promptly, but the exception it raises does
    NOT: a main thread parked in Future.result() inside an epoch window only sees
    it once every in-flight worker has finished (measured: 17s on this box, and
    proportional to the request timeout in a real run). Setting the cooperative
    shutdown flag from the handler is what reaches the worker threads immediately
    — utils.post_with_retry polls it and gives up instead of retrying. The
    KeyboardInterrupt is still raised afterwards so existing `except
    KeyboardInterrupt` teardown keeps working unchanged.

    Idempotent, so callers may invoke it defensively. Signal handlers can only be
    installed from the main thread; elsewhere the atexit hook alone is registered.
    """
    global _handlers_installed
    if _handlers_installed:
        return
    _handlers_installed = True
    atexit.register(stop)

    def _graceful_shutdown(signum, _frame):
        # First thing, before any logging or unwinding: unblock the workers.
        utils.request_shutdown()
        logger.warning("Received signal %d; shutting down vLLM and exiting.", signum)
        if signum == signal.SIGINT:
            # Preserve Ctrl-C semantics for the callers that catch it.
            raise KeyboardInterrupt
        sys.exit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        with contextlib.suppress(ValueError, OSError, AttributeError):
            signal.signal(sig, _graceful_shutdown)


def restart(model_path: str) -> None:
    """Stop the current server (if any) and start fresh on `model_path`."""
    stop()
    # Pin the instance port BEFORE waiting on it. A flow whose first vLLM call is
    # restart() (sft/train.py validates and serves without ever calling start())
    # has not run _assign_instance_port yet, so wait_down() would poll the
    # *configured* port — and a foreign server sitting on it, exactly the case
    # _assign_instance_port exists to sidestep by moving to a free port, would
    # instead time out as "an orphaned or foreign server owns it". Assigning first
    # makes wait_down() ask about the port we will actually bind. On a genuine
    # restart this is a no-op (_port_assigned pins us to our own port, which stop()
    # just released), so the leftover/orphan guard below is unchanged.
    _assign_instance_port()
    # Confirm the port is actually free before starting — guards against a
    # content-blind readiness latch onto a leftover/orphaned server.
    wait_down()
    start(model_path)


def load_adapter(name: str, path: str) -> None:
    """Hot-load a LoRA adapter into the running server under `name`.

    Used by the non-expert LoRA path (LORA_TARGET_EXPERTS=0) to refresh the served
    weights at an epoch boundary without restarting vLLM or merging a full
    checkpoint. After this returns the adapter answers requests whose `model` field
    is `name` (see inference_script.set_active_model). Requires the server to have
    been started with --enable-lora and VLLM_ALLOW_RUNTIME_LORA_UPDATING=True.
    """
    url = f"{config.VLLM_BASE_URL.rstrip('/')}/load_lora_adapter"
    headers = {"Authorization": f"Bearer {config.VLLM_API_KEY}"}
    logger.info("Hot-loading LoRA adapter '%s' from %s …", name, path)
    resp = requests.post(
        url, headers=headers,
        json={"lora_name": name, "lora_path": path},
        timeout=300,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"vLLM refused to load LoRA adapter '{name}' from {path} "
            f"(HTTP {resp.status_code}): {resp.text}"
        )
    logger.info("LoRA adapter '%s' is live.", name)


def unload_adapter(name: str) -> None:
    """Unload a previously hot-loaded LoRA adapter, freeing its slot.

    Best-effort: a failure here is logged but not raised — the new adapter is
    already live, and a lingering old one only wastes a LoRA slot.
    """
    url = f"{config.VLLM_BASE_URL.rstrip('/')}/unload_lora_adapter"
    headers = {"Authorization": f"Bearer {config.VLLM_API_KEY}"}
    try:
        resp = requests.post(
            url, headers=headers, json={"lora_name": name}, timeout=60,
        )
        if resp.status_code != 200:
            logger.warning(
                "vLLM failed to unload LoRA adapter '%s' (HTTP %d): %s",
                name, resp.status_code, resp.text,
            )
        else:
            logger.info("Unloaded LoRA adapter '%s'.", name)
    except requests.RequestException as exc:
        logger.warning("Error unloading LoRA adapter '%s': %s", name, exc)
