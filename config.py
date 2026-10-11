"""Shared configuration for the RAG package.

Embedding is handled by SentenceTransformer with all-MiniLM-L6-v2 (384-dim).
Text generation is handled by Gemini (set GEMINI_API_KEY in backend/env/.env).
No OpenAI API is used.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Ensure parent backend folder is in path to import env_loader
_BACKEND_DIR = ROOT.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

try:
    from env_loader import load_backend_env
    load_backend_env()
except ImportError:
    pass


logger = logging.getLogger(__name__)

_raw_env = os.getenv("RAG_RAW_DIR", "Data")
RAW_DIR = Path(_raw_env) if Path(_raw_env).is_absolute() else ROOT / _raw_env

# Resolve OKF_DIR relative to ROOT if a relative path is given.
# Default is okf_output/ (separate from the okf/ Python module code).
_okf_env = os.getenv("RAG_OKF_DIR", "okf_output")
OKF_DIR = Path(_okf_env) if Path(_okf_env).is_absolute() else ROOT / _okf_env

_cache_env = os.getenv("RAG_CACHE_DIR", "rag_cache")
CACHE_DIR = Path(_cache_env) if Path(_cache_env).is_absolute() else ROOT / _cache_env

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OKF_DIR.mkdir(parents=True, exist_ok=True)

# ── Embedding (all-MiniLM-L6-v2 only,  384-dimensional) ─────────────────────────────────
LOCAL_EMBEDDING_MODEL: str = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM: int = 384

# ── Retrieval settings ────────────────────────────────────────────────────────
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
EXPANSION_QUERIES = int(os.getenv("RAG_EXPANSION_QUERIES", "5"))
VOICE_CHUNK_MAX_CHARS = int(os.getenv("RAG_VOICE_CHUNK_CHARS", "500"))
# Max number of expanded query variants sent to the embedding model.
# Keeping this low (3) is the single biggest latency saving — each extra
# variant adds ~1 s of CPU embedding time on a typical machine.
MAX_EXPANSION_VARIANTS = int(os.getenv("RAG_MAX_EXPANSION_VARIANTS", "3"))

# ── Chunking settings ─────────────────────────────────────────────────────────
CHUNK_MIN = int(os.getenv("RAG_MIN_CHUNK_CHARS", "150"))
CHUNK_MAX = int(os.getenv("RAG_MAX_CHUNK_CHARS", "1000"))  # MiniLM 256-token window
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "0"))

# ── Batch / cache sizes ───────────────────────────────────────────────────────
EMBEDDING_BATCH_SIZE = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "64"))  # MiniLM is lightweight
# Raised from 500 → 2000 and 200 → 1000: larger caches mean more cache hits
# across sessions, turning 5-6 s embedding passes into 0 ms lookups.
EMBED_CACHE_SIZE = int(os.getenv("RAG_EMBED_CACHE_SIZE", "2000"))
SEMANTIC_CACHE_SIZE = int(os.getenv("RAG_CACHE_SIZE", "1000"))

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
PGVECTOR_TABLE = os.getenv("RAG_PGVECTOR_TABLE", "rag_chunks")
PGVECTOR_DIM = int(os.getenv("RAG_PGVECTOR_DIM", str(EMBEDDING_DIM)))


def print_config_summary() -> None:
    print("-- RAG config ------------------------")
    print(f"  Embedding model:  {LOCAL_EMBEDDING_MODEL}  ({EMBEDDING_DIM}-dim, local all-MiniLM-L6-v2)")
    print(f"  Text generation:  Gemini (key from GEMINI_API_KEY)")
    print(f"  Raw dir:          {RAW_DIR}")
    print(f"  OKF dir:          {OKF_DIR}")
    print(f"  pgvector table:   {PGVECTOR_TABLE}  (dim={PGVECTOR_DIM})")
    print(f"  TOP_K:            {TOP_K} | Expansion queries: {EXPANSION_QUERIES}")
    print(f"  Chunk min/max:    {CHUNK_MIN}/{CHUNK_MAX} | Overlap: {CHUNK_OVERLAP}")
    print("--------------------------------------")
