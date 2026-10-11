"""
cache.py
────────
Persistent disk-backed caches for:
  - EmbedCache: Caches query embedding vectors (avoids redundant API calls).
  - SemanticCache: Caches full RAG answers and retrieved docs (instant responses).

Both caches are loaded into memory at startup from JSON files in CACHE_DIR
and written back to disk asynchronously in background threads to keep requests non-blocking.
"""

import time
import logging
import json
import threading
import os
import re
from collections import OrderedDict
from pathlib import Path

try:
    from .config import EMBED_CACHE_SIZE, SEMANTIC_CACHE_SIZE, CACHE_DIR, EMBEDDING_DIM, LOCAL_EMBEDDING_MODEL
except ImportError:
    from config import EMBED_CACHE_SIZE, SEMANTIC_CACHE_SIZE, CACHE_DIR, EMBEDDING_DIM, LOCAL_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

MODEL_CACHE_SUFFIX = re.sub(r"[^a-z0-9]+", "-", LOCAL_EMBEDDING_MODEL.lower()).strip("-")
EMBED_CACHE_FILE = CACHE_DIR / f"embed_cache.{MODEL_CACHE_SUFFIX}.json"
SEMANTIC_CACHE_FILE = CACHE_DIR / f"semantic_cache.{MODEL_CACHE_SUFFIX}.json"
CACHE_VERSION = 5  # bumped: invalidates old dimension-mismatched entries

_disk_write_lock = threading.Lock()


def _save_to_disk_async(filepath: Path, cache_data: dict) -> None:
    """Save memory cache data to disk asynchronously in a background thread."""
    def target():
        with _disk_write_lock:
            try:
                # Convert OrderedDict to a normal dict for clean JSON serialization
                tmp_path = filepath.with_name(f"{filepath.name}.{os.getpid()}.tmp")
                tmp_path.write_text(json.dumps(cache_data), encoding="utf-8")
                
                # Windows atomic replace workaround:
                # If filepath exists, try to replace it. If it fails due to locking,
                # retry after a short delay, or write directly as fallback.
                try:
                    if filepath.exists():
                        tmp_path.replace(filepath)
                    else:
                        tmp_path.rename(filepath)
                except OSError:
                    import time
                    for _ in range(5):
                        time.sleep(0.1)
                        try:
                            tmp_path.replace(filepath)
                            break
                        except OSError:
                            pass
                    else:
                        # Fallback to direct write if replace/rename keeps failing
                        with open(filepath, "w", encoding="utf-8") as f:
                            json.dump(cache_data, f)
                        if tmp_path.exists():
                            tmp_path.unlink()
            except Exception as e:
                logger.warning("[CACHE] Disk write failed for %s: %s", filepath.name, e)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()


# ── Load Caches from Disk at Startup ──────────────────────────────────────────

def _load_embed_cache() -> OrderedDict:
    cache = OrderedDict()
    if EMBED_CACHE_FILE.exists():
        try:
            data = json.loads(EMBED_CACHE_FILE.read_text(encoding="utf-8"))
            skipped = 0
            for k, v in data.items():
                # Discard any entry whose dimension doesn't match current model
                if isinstance(v, list) and len(v) == EMBEDDING_DIM:
                    cache[k] = v
                else:
                    skipped += 1
            if skipped:
                logger.warning(
                    "[CACHE] Discarded %d embed cache entries with wrong dimension "
                    "(expected %d). Run ingest.py to rebuild.",
                    skipped, EMBEDDING_DIM,
                )
            logger.info("[CACHE] Loaded %d embedding cache entries from disk.", len(cache))
        except Exception as e:
            logger.warning("[CACHE] Failed to load embedding cache from disk: %s", e)
    return cache


def _load_semantic_cache() -> OrderedDict:
    cache = OrderedDict()
    if SEMANTIC_CACHE_FILE.exists():
        try:
            data = json.loads(SEMANTIC_CACHE_FILE.read_text(encoding="utf-8"))
            for k, v in data.items():
                cache[k] = v
            logger.info("[CACHE] Loaded %d semantic cache entries from disk.", len(cache))
        except Exception as e:
            logger.warning("[CACHE] Failed to load semantic cache from disk: %s", e)
    return cache


_embed_cache: OrderedDict[str, list[float]] = _load_embed_cache()
_semantic_cache: OrderedDict[str, dict] = _load_semantic_cache()


# ── Embedding cache API ───────────────────────────────────────────────────────

def get_cached_embedding(text: str) -> list[float] | None:
    key = text.strip().lower()
    val = _embed_cache.get(key)
    if val is not None:
        if len(val) != EMBEDDING_DIM:
            _embed_cache.pop(key, None)
            return None
        _embed_cache.move_to_end(key)
    return val


def set_cached_embedding(text: str, vector: list[float]) -> None:
    if len(vector) != EMBEDDING_DIM:
        logger.warning(
            "[CACHE] Rejecting cached embedding with wrong dimension: %d (expected %d)",
            len(vector), EMBEDDING_DIM
        )
        return
    key = text.strip().lower()
    _embed_cache[key] = vector
    if len(_embed_cache) > EMBED_CACHE_SIZE:
        _embed_cache.popitem(last=False)
    # Persist cache to disk asynchronously
    _save_to_disk_async(EMBED_CACHE_FILE, dict(_embed_cache))


# ── Semantic cache API ────────────────────────────────────────────────────────

def _normalise(query: str) -> str:
    return " ".join(query.lower().strip().split())


def get_cached_rag(query: str) -> dict | None:
    key = _normalise(query)
    entry = _semantic_cache.get(key)
    if entry and entry.get("version") != CACHE_VERSION:
        _semantic_cache.pop(key, None)
        return None
    if entry:
        _semantic_cache.move_to_end(key)
        logger.info("[CACHE] HIT for: '%s'", query)
    return entry


def set_cached_rag(query: str, docs: list[dict], answer: str = "") -> None:
    key = _normalise(query)
    _semantic_cache[key] = {"docs": docs, "answer": answer, "ts": time.time(), "version": CACHE_VERSION}
    if len(_semantic_cache) > SEMANTIC_CACHE_SIZE:
        _semantic_cache.popitem(last=False)
    # Persist cache to disk asynchronously
    _save_to_disk_async(SEMANTIC_CACHE_FILE, dict(_semantic_cache))
