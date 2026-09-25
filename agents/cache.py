"""A disk cache for model responses, keyed by the exact request that produced them.

Free API tiers meter tokens per minute, which makes re-running an analysis
expensive in time even when it is free in money. That matters here more than it
would in most projects: developing the framework means running the same datasets
through the same nine agents repeatedly, and almost every prompt is identical to
the one sent on the previous run.

Caching is sound rather than merely convenient because the framework fixes
temperature at 0.0. The same model given the same system and user prompt is
asked to be deterministic, so replaying a stored response is replaying what the
provider would have returned. A cache under a sampling temperature would be
quietly substituting one draw for another; under greedy decoding it is not.

Three consequences worth having:

* A re-run after a code change costs nothing for the prompts that did not change,
  so rate limits stop governing iteration speed.
* The run reported in the paper can be replayed exactly, months later, without
  depending on a provider still serving that model version.
* Cost accounting stays honest, because cached responses are counted separately
  rather than being reported as fresh calls that never happened.

The cache is keyed on backend, model, and the full text of both prompts. Any
change to an agent's instructions changes the key, so a stale answer cannot
survive an edit to the prompt that produced it.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from config import CONFIG, ROOT

CACHE_DIR = ROOT / ".llm_cache"


def _key(backend: str, model: str, system: str, user: str) -> str:
    """A stable identifier for one request.

    The prompts are hashed rather than stored in the filename because they run to
    thousands of characters. A separator that cannot appear in a hash input is
    used between fields so that concatenation cannot make two different requests
    collide.
    """
    digest = hashlib.sha256()
    for part in (backend, model, system, user):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _path(key: str) -> Path:
    # Two-character prefix directories keep any single directory small enough
    # that listing it stays fast once a few thousand runs have accumulated.
    return CACHE_DIR / key[:2] / f"{key}.json"


def lookup(backend: str, model: str, system: str, user: str) -> dict[str, Any] | None:
    """Return a stored response for this exact request, or None."""
    if not CONFIG.llm_cache_enabled:
        return None

    path = _path(_key(backend, model, system, user))
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A truncated file from an interrupted write is not worth recovering:
        # the cost of a cache miss is one API call, and failing loudly here
        # would turn a cosmetic problem into a broken run.
        return None


def store(backend: str, model: str, system: str, user: str,
          text: str, input_tokens: int, output_tokens: int,
          provider: str | None = None) -> None:
    """Record a response for future replay.

    `backend` is what was configured and `provider` is what answered. They
    differ on a failover chain, where the configured backend is the chain
    itself, and only the second identifies the model that wrote the text.
    """
    if not CONFIG.llm_cache_enabled:
        return

    path = _path(_key(backend, model, system, user))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": backend,
        "model": model,
        "provider": provider or backend,
        "text": text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "stored_at": time.time(),
    }
    try:
        # Written to a temporary name and moved into place, so a process killed
        # mid-write leaves the previous entry intact rather than a half file.
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        # A cache that cannot be written must not stop an analysis.
        pass


def evict(backend: str, model: str, system: str, user: str) -> bool:
    """Remove one stored response. Returns whether anything was removed."""
    path = _path(_key(backend, model, system, user))
    try:
        path.unlink()
        return True
    except OSError:
        return False


def statistics() -> dict[str, Any]:
    """How much is stored, for the run record and for deciding when to clear."""
    if not CACHE_DIR.exists():
        return {"entries": 0, "size_mb": 0.0}
    entries = list(CACHE_DIR.rglob("*.json"))
    total = sum(p.stat().st_size for p in entries)
    return {"entries": len(entries), "size_mb": round(total / 1024 ** 2, 3)}


def clear() -> int:
    """Delete every stored response. Returns how many were removed."""
    if not CACHE_DIR.exists():
        return 0
    removed = 0
    for path in CACHE_DIR.rglob("*.json"):
        try:
            path.unlink()
            removed += 1
        except OSError:
            continue
    return removed
