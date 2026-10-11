# """
# embedder.py
# ───────────
# Provides embed_texts (sync) and embed_texts_async (async).

# Uses the standard SentenceTransformer BAAI/bge-m3 model — 1024-dimensional vectors.
# No API cost, no network dependency at inference time.

# Cold-start optimisation: the SentenceTransformer model is loaded in a
# background daemon thread at import time so the first user query doesn't
# pay the 30+ second library-import penalty.
# """

# import asyncio
# import logging
# import os
# import threading

# import numpy as np

# try:
#     from .config import (
#         EMBEDDING_BATCH_SIZE,
#         LOCAL_EMBEDDING_MODEL,
#         EMBEDDING_DIM,
#     )
#     from .cache import get_cached_embedding, set_cached_embedding
# except ImportError:
#     from config import (
#         EMBEDDING_BATCH_SIZE,
#         LOCAL_EMBEDDING_MODEL,
#         EMBEDDING_DIM,
#     )
#     from cache import get_cached_embedding, set_cached_embedding

# logger = logging.getLogger(__name__)

# # ── Load local model once at import time ──────────────────────────────────────

# _local_encoder = None
# _model_ready = threading.Event()

# os.environ.setdefault("HF_HUB_OFFLINE", "1")
# os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# def _preload_local_model() -> None:
#     """Background thread: import sentence_transformers and load BGE-M3.

#     Loading is done in a daemon thread started at module import time so
#     the ~30 s library import does NOT block the first user query.
#     """
#     global _local_encoder
#     try:
#         from sentence_transformers import SentenceTransformer  # noqa: PLC0415

#         try:
#             _local_encoder = SentenceTransformer(
#                 LOCAL_EMBEDDING_MODEL,
#                 local_files_only=True,
#             )
#         except Exception:
#             logger.info(
#                 "[EMBED] Model %s not found locally. Attempting download...",
#                 LOCAL_EMBEDDING_MODEL,
#             )
#             orig_hf_offline = os.environ.get("HF_HUB_OFFLINE")
#             orig_tf_offline = os.environ.get("TRANSFORMERS_OFFLINE")

#             os.environ.pop("HF_HUB_OFFLINE", None)
#             os.environ.pop("TRANSFORMERS_OFFLINE", None)

#             try:
#                 _local_encoder = SentenceTransformer(
#                     LOCAL_EMBEDDING_MODEL,
#                     local_files_only=False,
#                 )
#             finally:
#                 if orig_hf_offline is not None:
#                     os.environ["HF_HUB_OFFLINE"] = orig_hf_offline
#                 if orig_tf_offline is not None:
#                     os.environ["TRANSFORMERS_OFFLINE"] = orig_tf_offline

#         logger.info("[EMBED] Local SentenceTransformer ready: %s", LOCAL_EMBEDDING_MODEL)
#     except Exception as exc:
#         logger.error("[EMBED] Failed to preload local model: %s", exc)
#     finally:
#         _model_ready.set()


# # ── Helpers ───────────────────────────────────────────────────────────────────


# def _get_local_encoder():
#     """Return the local encoder, blocking until it is ready."""
#     if not _model_ready.wait(timeout=120):
#         raise RuntimeError(
#             "Timed out waiting for local embedding model to load. "
#             "Check that sentence-transformers is installed and the model is cached."
#         )
#     if _local_encoder is None:
#         raise RuntimeError(
#             "Local embedding model failed to load. "
#             f"Check that LOCAL_EMBEDDING_MODEL={LOCAL_EMBEDDING_MODEL!r} is available."
#         )
#     return _local_encoder


# def _embed_local(texts: list[str]) -> list[list[float]]:
#     """Run SentenceTransformer on texts, using cache where possible.

#     Cached vectors whose dimension does not match the current model's output
#     dimension are silently evicted so stale embeddings never corrupt ingestion.
#     """
#     encoder = _get_local_encoder()
#     expected_dim = EMBEDDING_DIM

#     result_map: dict[int, list[float]] = {}
#     uncached_indices: list[int] = []
#     uncached_texts: list[str] = []

#     for i, text in enumerate(texts):
#         cached = get_cached_embedding(text)
#         if cached is not None:
#             if len(cached) == expected_dim:
#                 result_map[i] = cached
#                 continue
#             # Dimension mismatch → treat as cache miss (stale entry)
#             logger.debug(
#                 "[EMBED] Cache dim mismatch (%d vs %d), re-embedding.",
#                 len(cached), expected_dim,
#             )
#         uncached_indices.append(i)
#         uncached_texts.append(text)

#     if uncached_texts:
#         vectors = encoder.encode(
#             uncached_texts,
#             batch_size=EMBEDDING_BATCH_SIZE,
#             normalize_embeddings=True,
#             show_progress_bar=True,
#         )
#         for idx, text, vector in zip(uncached_indices, uncached_texts, vectors):
#             vec_array = np.asarray(vector, dtype=np.float32)
#             if vec_array.shape[0] != expected_dim:
#                 raise RuntimeError(
#                     f"Embedding dimension mismatch: got {vec_array.shape[0]}, "
#                     f"expected {expected_dim}."
#                 )
#             vec: list[float] = vec_array.tolist()
#             set_cached_embedding(text, vec)
#             result_map[idx] = vec

#     return [result_map[i] for i in range(len(texts))]


# # Start loading immediately — ready well before the first user query.
# threading.Thread(
#     target=_preload_local_model, name="embed-preload", daemon=True
# ).start()
# logger.info(
#     "[EMBED] Local model preload started in background (%s)", LOCAL_EMBEDDING_MODEL
# )


# # ── Public API ────────────────────────────────────────────────────────────────


# def embed_texts(texts: list[str]) -> np.ndarray:
#     """
#     Synchronous embedding. Returns shape (N, 1024) float array.
#     Used by ingest scripts and non-async routes.
#     """
#     if not texts:
#         return np.empty((0, 0), dtype=float)

#     return np.array(_embed_local(texts), dtype=float)


# async def embed_texts_async(texts: list[str]) -> np.ndarray:
#     """
#     Async embedding. Returns shape (N, 1024) float array.
#     Offloads the CPU-bound model to a thread so the event loop stays free.
#     """
#     if not texts:
#         return np.empty((0, 0), dtype=float)

#     vecs = await asyncio.to_thread(_embed_local, texts)
#     return np.array(vecs, dtype=float)




"""
embedder.py
───────────
Provides embed_texts (sync) and embed_texts_async (async).

Uses the standard SentenceTransformer BAAI/bge-m3 model — 1024-dimensional vectors.
No API cost, no network dependency at inference time.

Cold-start optimisation: the SentenceTransformer model is loaded in a
background daemon thread at import time so the first user query doesn't
pay the 30+ second library-import penalty.
"""

import asyncio
import logging
import os
import threading
import warnings

import numpy as np

# Suppress the harmless PyTorch internal class inspection warning that appears
# when sentence-transformers imports torch. This is a known torch/ST version
# mismatch issue and does NOT affect embedding correctness.
warnings.filterwarnings(
    "ignore",
    message=".*Examining the path of torch.classes.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*__path__._path.*",
    category=UserWarning,
)

try:
    from .config import (
        EMBEDDING_BATCH_SIZE,
        LOCAL_EMBEDDING_MODEL,
        EMBEDDING_DIM,
    )
    from .cache import get_cached_embedding, set_cached_embedding
except ImportError:
    from config import (
        EMBEDDING_BATCH_SIZE,
        LOCAL_EMBEDDING_MODEL,
        EMBEDDING_DIM,
    )
    from cache import get_cached_embedding, set_cached_embedding

logger = logging.getLogger(__name__)

# ── Load local model once at import time ──────────────────────────────────────

_local_encoder = None
_model_ready = threading.Event()

# NOTE: Do NOT set HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE here.
# The preload thread handles local-cache-first with a download fallback internally.


def _preload_local_model() -> None:
    """Background thread: import sentence_transformers and load BGE-M3.

    Loading is done in a daemon thread started at module import time so
    the ~30 s library import does NOT block the first user query.
    """
    global _local_encoder
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        try:
            _local_encoder = SentenceTransformer(
                LOCAL_EMBEDDING_MODEL,
                local_files_only=True,
            )
        except Exception:
            logger.info(
                "[EMBED] Model %s not found locally. Attempting download...",
                LOCAL_EMBEDDING_MODEL,
            )
            orig_hf_offline = os.environ.get("HF_HUB_OFFLINE")
            orig_tf_offline = os.environ.get("TRANSFORMERS_OFFLINE")

            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)

            try:
                _local_encoder = SentenceTransformer(
                    LOCAL_EMBEDDING_MODEL,
                    local_files_only=False,
                )
            finally:
                if orig_hf_offline is not None:
                    os.environ["HF_HUB_OFFLINE"] = orig_hf_offline
                if orig_tf_offline is not None:
                    os.environ["TRANSFORMERS_OFFLINE"] = orig_tf_offline

        logger.info("[EMBED] Local SentenceTransformer ready: %s", LOCAL_EMBEDDING_MODEL)
    except Exception as exc:
        logger.error("[EMBED] Failed to preload local model: %s", exc)
    finally:
        _model_ready.set()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_local_encoder():
    """Return the local encoder, blocking until it is ready."""
    if not _model_ready.wait(timeout=120):
        raise RuntimeError(
            "Timed out waiting for local embedding model to load. "
            "Check that sentence-transformers is installed and the model is cached."
        )
    if _local_encoder is None:
        raise RuntimeError(
            "Local embedding model failed to load. "
            f"Check that LOCAL_EMBEDDING_MODEL={LOCAL_EMBEDDING_MODEL!r} is available."
        )
    return _local_encoder


def _embed_local(texts: list[str]) -> list[list[float]]:
    """Run SentenceTransformer on texts, using cache where possible.

    Cached vectors whose dimension does not match the current model's output
    dimension are silently evicted so stale embeddings never corrupt ingestion.
    """
    encoder = _get_local_encoder()
    expected_dim = EMBEDDING_DIM

    result_map: dict[int, list[float]] = {}
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    for i, text in enumerate(texts):
        cached = get_cached_embedding(text)
        if cached is not None:
            if len(cached) == expected_dim:
                result_map[i] = cached
                continue
            # Dimension mismatch → treat as cache miss (stale entry)
            logger.debug(
                "[EMBED] Cache dim mismatch (%d vs %d), re-embedding.",
                len(cached), expected_dim,
            )
        uncached_indices.append(i)
        uncached_texts.append(text)

    if uncached_texts:
        vectors = encoder.encode(
            uncached_texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        for idx, text, vector in zip(uncached_indices, uncached_texts, vectors):
            vec_array = np.asarray(vector, dtype=np.float32)
            if vec_array.shape[0] != expected_dim:
                raise RuntimeError(
                    f"Embedding dimension mismatch: got {vec_array.shape[0]}, "
                    f"expected {expected_dim}."
                )
            vec: list[float] = vec_array.tolist()
            set_cached_embedding(text, vec)
            result_map[idx] = vec

    return [result_map[i] for i in range(len(texts))]


# Start loading immediately — ready well before the first user query.
threading.Thread(
    target=_preload_local_model, name="embed-preload", daemon=True
).start()
logger.info(
    "[EMBED] Local model preload started in background (%s)", LOCAL_EMBEDDING_MODEL
)


# ── Public API ────────────────────────────────────────────────────────────────


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Synchronous embedding. Returns shape (N, 1024) float array.
    Used by ingest scripts and non-async routes.
    """
    if not texts:
        return np.empty((0, 0), dtype=float)

    return np.array(_embed_local(texts), dtype=float)


async def embed_texts_async(texts: list[str]) -> np.ndarray:
    """
    Async embedding. Returns shape (N, 1024) float array.
    Offloads the CPU-bound model to a thread so the event loop stays free.
    """
    if not texts:
        return np.empty((0, 0), dtype=float)

    vecs = await asyncio.to_thread(_embed_local, texts)
    return np.array(vecs, dtype=float)