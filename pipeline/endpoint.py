"""Discovery for the vLLM endpoint a run actually ended up on.

The configured VLLM_BASE_URL is a *request*, not a fact. vllm_server pins each
concurrent instance to a private port at start() time (_assign_instance_port), so
the port a server is really listening on is only known after launch — and only
inside the process that did the launching. Exporting it to os.environ there
covers the children that process spawns afterwards, and nothing else: an operator
running a benchmark in a second shell, or a harness started before the server,
reads the repo-root .env and gets the original port back. Measured: server on
8001, out-of-band harness resolving 8000, with no error on either side.

So the resolved endpoint is published here instead of only being pushed into an
environment, and readers look it up when they need it.

Liveness comes from the port reservation that already exists. utils.reserve_port
holds an flock on `port_<n>.lock` for the owner's lifetime and the kernel drops it
when that process dies by any means, so a published endpoint is live exactly when
its lock is still held — which a non-destructive LOCK_NB probe can ask. A file
left behind by a crashed run therefore reads as dead rather than being trusted.

Ambiguity is refused, never guessed. Two live servers and no explicit choice is
precisely the situation that produced the original bug (four runs all talking to
the first server, each writing the answers out under its own model's name), and
picking one would reintroduce it in a quieter form.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from .utils import _LOCK_DIR, _lock_path, logger

try:
    import fcntl  # POSIX only; matches utils' own guard.
except ImportError:  # pragma: no cover - discovery degrades to "nothing published".
    fcntl = None

_PREFIX = "vllm_endpoint_"

# Hosts that mean "a vLLM on this box". Mirrors the harness config validators.
LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0")


def is_local_url(url: str) -> bool:
    """True if `url` names a local self-hosted endpoint rather than a hosted API."""
    return any(h in (url or "") for h in LOCAL_HOSTS)


def _endpoint_path(port: int) -> Path:
    return _LOCK_DIR / f"{_PREFIX}{port}.json"


def publish(url: str, **meta) -> None:
    """Record `url` as this instance's live vLLM endpoint.

    Best-effort: a run that cannot write the file still serves correctly, it just
    is not discoverable, so a failure here must never take the server down with it.
    """
    port = urlparse(url).port
    if port is None:
        return
    payload = {"url": url, "pid": os.getpid(), "published": time.time(), **meta}
    try:
        _LOCK_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _endpoint_path(port).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(_endpoint_path(port))  # atomic: readers never see a partial file
    except OSError as exc:
        logger.debug("Could not publish vLLM endpoint %s: %s", url, exc)


def unpublish(url: str) -> None:
    """Drop a previously published endpoint (best-effort)."""
    port = urlparse(url).port
    if port is None:
        return
    try:
        _endpoint_path(port).unlink(missing_ok=True)
    except OSError:
        pass


def _port_lock_held(port: int) -> bool:
    """True if some live process still holds the reservation for `port`.

    Probes with LOCK_NB and immediately releases anything it managed to take, so
    asking the question never steals the reservation from a live owner or leaves
    one behind.
    """
    path = _lock_path(f"port_{port}")
    if not path.exists():
        return False
    if fcntl is None:
        return True  # cannot tell; assume live rather than discard a real server
    try:
        with open(path, "w") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True  # still held by its owner -> live
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return False
    except OSError:
        return False


def discover() -> list[dict]:
    """Every published endpoint whose owning process is still alive.

    Stale files (owner crashed, lock released by the kernel) are pruned as they
    are found, so the directory does not accumulate ghosts across runs.
    """
    out: list[dict] = []
    try:
        candidates = sorted(_LOCK_DIR.glob(f"{_PREFIX}*.json"))
    except OSError:
        return out
    for path in candidates:
        try:
            rec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        port = urlparse(rec.get("url", "")).port
        if port is None:
            continue
        if not _port_lock_held(port):
            with_suppress = getattr(path, "unlink", None)
            if with_suppress:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            continue
        out.append(rec)
    return out


def resolve(configured_url: str, explicit_url: Optional[str] = None) -> str:
    """The URL a local-vLLM caller should actually use.

    Order:
      1. `explicit_url` — an operator who set VLLM_BASE_URL in this shell means it.
      2. The single live published endpoint, if there is exactly one. This is what
         repairs the shift: the config says 8000, the server moved to 8001, and a
         benchmark launched out of band still lands on the right one.
      3. The configured URL, when nothing is published (a hand-started server, or
         discovery unavailable) — never worse than the behaviour before discovery
         existed.

    Raises RuntimeError when several servers are live and the caller did not say
    which it wants: that is unanswerable, and answering it by guessing is the
    original bug.
    """
    if not is_local_url(configured_url):
        return configured_url            # hosted route; nothing to discover
    if explicit_url:
        return explicit_url

    live = discover()
    if not live:
        return configured_url
    if len(live) == 1:
        url = live[0]["url"]
        if url.rstrip("/") != (configured_url or "").rstrip("/"):
            logger.warning(
                "Configured vLLM endpoint %s, but the live server is on %s "
                "(started for %s on GPU(s) %s). Using the live one — the pipeline "
                "moves each instance to a private port after launch.",
                configured_url, url, live[0].get("model", "?"), live[0].get("gpus", "?"),
            )
        return url

    listing = "\n".join(
        f"    {r['url']}   model={r.get('model', '?')}  gpus={r.get('gpus', '?')}  pid={r.get('pid', '?')}"
        for r in sorted(live, key=lambda r: r["url"])
    )
    raise RuntimeError(
        f"{len(live)} vLLM servers are live on this box and VLLM_BASE_URL is not "
        f"set, so there is no way to tell which one this run should use:\n"
        f"{listing}\n"
        f"Set VLLM_BASE_URL to the one you mean — e.g. "
        f"`VLLM_BASE_URL={sorted(r['url'] for r in live)[0]} <your command>`. "
        f"Guessing here is how benchmark results end up attributed to the wrong "
        f"model."
    )
