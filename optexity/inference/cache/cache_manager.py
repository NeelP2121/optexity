"""
Cache manager: load, save, and invalidate AgenticTaskCache entries.

Cache entries are keyed by a hash of the task text + base URL and stored as
JSON files in ~/.optexity_cache/. This location is intentionally separate from
task log directories so cached entries persist across automation runs.
"""

import hashlib
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

from optexity.inference.cache.action_cache import AgenticTaskCache

logger = logging.getLogger(__name__)

# Allow override via env var for testing
CACHE_DIR = Path(os.environ.get("OPTEXITY_CACHE_DIR", Path.home() / ".optexity_cache"))


def compute_cache_key(task_text: str, base_url: str) -> str:
    """Return a 16-char hex key derived from the task text and base URL.

    The base URL is normalised (scheme + host only) so that query strings and
    paths don't prevent cache hits for the same site.
    """
    parsed = urlparse(base_url)
    normalised_url = f"{parsed.scheme}://{parsed.netloc}".lower()
    raw = f"{task_text.strip()}:{normalised_url}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def save_cache(key: str, cache: AgenticTaskCache) -> None:
    """Persist a cache entry to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    path.write_text(cache.model_dump_json(indent=2))
    logger.info(f"[cache] Saved {len(cache.actions)} actions → {path}")


def load_cache(key: str) -> AgenticTaskCache | None:
    """Return the cache entry for *key*, or None if not found / corrupt."""
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        return AgenticTaskCache.model_validate_json(path.read_text())
    except Exception as exc:
        logger.warning(f"[cache] Failed to load {path}: {exc}. Ignoring stale cache.")
        return None


def invalidate_cache(key: str) -> None:
    """Delete the cache entry for *key*."""
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        path.unlink()
        logger.info(f"[cache] Invalidated {path}")


def record_failure(key: str, step_index: int) -> None:
    """Increment the failure counter for the step at *step_index*."""
    cache = load_cache(key)
    if cache is None:
        return
    for action in cache.actions:
        if action.step_index == step_index:
            action.failure_count += 1
            break
    save_cache(key, cache)
