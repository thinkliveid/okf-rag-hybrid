"""pgvector-backed storage and retrieval."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from contextlib import contextmanager
from typing import Any, Iterable

try:
    from .cache import get_cached_rag, set_cached_rag
    from .config import (
        DATABASE_URL,
        PGVECTOR_DIM,
        PGVECTOR_TABLE,
        TOP_K,
        VOICE_CHUNK_MAX_CHARS,
    )
    from .embedder import embed_texts, embed_texts_async
except ImportError:
    from cache import get_cached_rag, set_cached_rag
    from config import (
        DATABASE_URL,
        PGVECTOR_DIM,
        PGVECTOR_TABLE,
        TOP_K,
        VOICE_CHUNK_MAX_CHARS,
    )
    from embedder import embed_texts, embed_texts_async

logger = logging.getLogger(__name__)


def _driver():
    try:
        import psycopg

        return "psycopg", psycopg
    except ImportError:
        try:
            import psycopg2
            import psycopg2.extras

            return "psycopg2", psycopg2
        except ImportError as exc:
            raise RuntimeError("Install psycopg or psycopg2 to use pgvector") from exc


@contextmanager
def _connect():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is required for pgvector storage")
    name, module = _driver()
    try:
        conn = module.connect(DATABASE_URL, connect_timeout=5)
    except Exception as exc:
        raise RuntimeError(
            "Could not connect to Postgres. Make sure the server is running, "
            f"accepting connections on DATABASE_URL={DATABASE_URL!r}, and that "
            "the database exists.\n\n"
            "For this repo's default local setup, start pgvector with:\n"
            "  docker compose up -d postgres\n\n"
            "Note: Your ingested embeddings are saved persistently in the Docker volume.\n"
            "You only need to run 'python ingest.py' again if you have added or changed raw documents."
        ) from exc
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _quote_ident(identifier: str) -> str:
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", identifier):
        raise ValueError(f"Unsafe SQL identifier: {identifier!r}")
    return f'"{identifier}"'


TABLE = _quote_ident(PGVECTOR_TABLE)

SEARCH_VECTOR_EXPR = (
    "setweight(to_tsvector('simple', coalesce(metadata->'okf_frontmatter'->>'title', '')), 'A') || "
    "setweight(to_tsvector('simple', coalesce(metadata->'okf_frontmatter'->>'tags', '')), 'B') || "
    "setweight(to_tsvector('simple', coalesce(document, '')), 'D')"
)


def _vector_literal(values: Iterable[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"


def _json_loads(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value or {}


def ensure_schema(dimension: int | None = None) -> None:
    dim = dimension or PGVECTOR_DIM
    if not dim:
        existing = _existing_dimension()
        if existing:
            dim = existing
    if not dim:
        raise RuntimeError("Embedding dimension is required to initialize pgvector schema")

    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                id TEXT PRIMARY KEY,
                document TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding vector({int(dim)}) NOT NULL,
                search_vector tsvector GENERATED ALWAYS AS ({SEARCH_VECTOR_EXPR}) STORED,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        _migrate_search_vector(cur)
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {PGVECTOR_TABLE}_search_idx "
            f"ON {TABLE} USING GIN (search_vector)"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {PGVECTOR_TABLE}_embedding_hnsw_idx "
            f"ON {TABLE} USING hnsw (embedding vector_cosine_ops)"
        )
        cur.close()


def _migrate_search_vector(cur) -> None:
    cur.execute(
        """
        SELECT pg_get_expr(d.adbin, d.adrelid)
        FROM pg_attribute a
        LEFT JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
        WHERE a.attrelid = %s::regclass AND a.attname = 'search_vector' AND NOT a.attisdropped
        """,
        (TABLE,),
    )
    row = cur.fetchone()
    if row and row[0] and "setweight" in row[0]:
        return
    logger.info("[STORE] Rebuilding %s.search_vector with weighted OKF title/tags", PGVECTOR_TABLE)
    cur.execute(f"ALTER TABLE {TABLE} DROP COLUMN IF EXISTS search_vector")
    cur.execute(
        f"ALTER TABLE {TABLE} ADD COLUMN search_vector tsvector "
        f"GENERATED ALWAYS AS ({SEARCH_VECTOR_EXPR}) STORED"
    )


def _existing_dimension() -> int | None:
    try:
        with _connect() as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT vector_dims(embedding) FROM {TABLE} LIMIT 1")
            row = cur.fetchone()
            cur.close()
            return int(row[0]) if row else None
    except Exception:
        return None


def delete_collection() -> None:
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {TABLE}")
        cur.close()


def get_or_create_collection() -> str:
    ensure_schema()
    return PGVECTOR_TABLE


def insert_embeddings(records: list[dict[str, Any]], replace: bool = True) -> int:
    if not records:
        return 0
    dim = len(records[0]["embedding"])
    ensure_schema(dim)

    with _connect() as conn:
        cur = conn.cursor()
        for record in records:
            metadata = json.dumps(record.get("metadata") or {})
            vector = _vector_literal(record["embedding"])
            if replace:
                cur.execute(
                    f"""
                    INSERT INTO {TABLE} (id, document, metadata, embedding)
                    VALUES (%s, %s, %s::jsonb, %s::vector)
                    ON CONFLICT (id) DO UPDATE SET
                        document = EXCLUDED.document,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                    """,
                    (record["id"], record["text"], metadata, vector),
                )
            else:
                cur.execute(
                    f"""
                    INSERT INTO {TABLE} (id, document, metadata, embedding)
                    VALUES (%s, %s, %s::jsonb, %s::vector)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (record["id"], record["text"], metadata, vector),
                )
        cur.close()
    return len(records)


def build_records(chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk, embedding in zip(chunks, embeddings):
        text = chunk["text"]
        metadata = dict(chunk.get("metadata") or {})
        stable = json.dumps({"text": text, "metadata": metadata}, sort_keys=True)
        chunk_id = hashlib.sha256(stable.encode("utf-8")).hexdigest()
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        records.append({"id": chunk_id, "text": text, "metadata": metadata, "embedding": embedding})
    return records


def _search(query: str, query_embedding: list[float], top_k: int, trim: bool) -> list[dict[str, Any]]:
    ensure_schema(len(query_embedding))
    vector = _vector_literal(query_embedding)
    # Sanitize query for full-text search: keep ASCII alphanumeric only (handles Roman Urdu)
    ts_query = " | ".join(w for w in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(w) > 1)

    # Larger candidate pool → better BM25 reranking coverage
    candidate_pool = max(top_k * 20, 60)

    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            WITH ranked AS (
                SELECT
                    id,
                    document,
                    metadata,
                    embedding <=> %s::vector AS distance,
                    CASE
                        WHEN to_tsquery('simple', %s)::text = '' THEN 0
                        ELSE ts_rank_cd(search_vector, to_tsquery('simple', %s), 32)
                    END AS bm25_score
                FROM {TABLE}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            )
            SELECT id, document, metadata, distance, bm25_score
            FROM ranked
            ORDER BY ((1.0 / (1.0 + distance)) * 0.70 + LEAST(bm25_score * 5.0, 0.30) * 0.30) DESC
            LIMIT %s
            """,
            (vector, ts_query, ts_query, vector, candidate_pool, top_k),
        )
        rows = cur.fetchall()
        cur.close()

    docs: list[dict[str, Any]] = []
    for _id, text, metadata, distance, bm25_score in rows:
        similarity = 1.0 / (1.0 + float(distance))
        # Cap bm25 contribution to 0.30 max so vector always dominates
        bm25_contrib = min(float(bm25_score or 0.0) * 5.0, 0.30)
        score = (similarity * 0.70) + (bm25_contrib * 0.30)
        doc_text = text[:VOICE_CHUNK_MAX_CHARS] if trim and len(text) > VOICE_CHUNK_MAX_CHARS else text
        docs.append({"text": doc_text, "metadata": _json_loads(metadata), "score": score, "id": _id})
    return docs


def search_documents(query: str, top_k: int = TOP_K, trim_for_voice: bool = True) -> list[dict[str, Any]]:
    cached = get_cached_rag(query)
    if cached and len(cached.get("docs", [])) >= top_k:
        return cached["docs"]

    t0 = time.time()
    try:
        query_embedding = embed_texts([query])[0].tolist()
        docs = _search(query, query_embedding, top_k, trim_for_voice)
        logger.info("[STORE] pgvector hybrid search %.3fs - %d docs", time.time() - t0, len(docs))
        set_cached_rag(query, docs)
        return docs
    except Exception as exc:
        logger.error("[STORE] search_documents failed: %s", exc, exc_info=True)
        return []


async def search_documents_async(
    query: str,
    top_k: int = TOP_K,
    trim_for_voice: bool = True,
) -> list[dict[str, Any]]:
    cached = get_cached_rag(query)
    if cached and len(cached.get("docs", [])) >= top_k:
        return cached["docs"]

    t0 = time.time()
    try:
        query_embedding = (await embed_texts_async([query]))[0].tolist()
        docs = await asyncio.to_thread(_search, query, query_embedding, top_k, trim_for_voice)
        logger.info("[STORE] async pgvector hybrid search %.3fs - %d docs", time.time() - t0, len(docs))
        set_cached_rag(query, docs)
        return docs
    except Exception as exc:
        logger.error("[STORE] search_documents_async failed: %s", exc, exc_info=True)
        return []


def search_documents_batch(
    queries: list[str],
    top_k: int = TOP_K,
    trim_for_voice: bool = True,
) -> list[list[dict[str, Any]]]:
    """Perform pgvector hybrid search for multiple queries, batching their embedding computation."""
    results: list[list[dict[str, Any]] | None] = [None] * len(queries)
    uncached_queries: list[str] = []
    uncached_indices: list[int] = []

    for i, q in enumerate(queries):
        cached = get_cached_rag(q)
        if cached and len(cached.get("docs", [])) >= top_k:
            results[i] = cached["docs"]
        else:
            uncached_queries.append(q)
            uncached_indices.append(i)

    if uncached_queries:
        t0 = time.time()
        try:
            # Safety cap: never embed more than MAX_EXPANSION_VARIANTS queries in one
            # batch — extra queries beyond the cap are still searched if their
            # embeddings happen to land in the cache, so recall isn't impacted.
            try:
                from .config import MAX_EXPANSION_VARIANTS as _MAX_VAR
            except ImportError:
                from config import MAX_EXPANSION_VARIANTS as _MAX_VAR

            if len(uncached_queries) > _MAX_VAR:
                logger.debug(
                    "[STORE] Capping uncached queries from %d → %d to limit CPU embedding time.",
                    len(uncached_queries), _MAX_VAR,
                )
                uncached_queries = uncached_queries[:_MAX_VAR]
                uncached_indices = uncached_indices[:_MAX_VAR]

            # Batch embed all uncached queries in one call
            query_embeddings = embed_texts(uncached_queries)
            logger.info("[STORE] Batched embedding of %d queries took %.3fs", len(uncached_queries), time.time() - t0)
            
            for idx, q, emb in zip(uncached_indices, uncached_queries, query_embeddings):
                t_search = time.time()
                docs = _search(q, emb.tolist(), top_k, trim_for_voice)
                logger.debug("[STORE] Single search for '%s' took %.3fs", q, time.time() - t_search)
                set_cached_rag(q, docs)
                results[idx] = docs
        except Exception as exc:
            logger.error("[STORE] search_documents_batch failed: %s", exc, exc_info=True)
            # Fallback to sequential search_documents for the uncached ones
            for idx, q in zip(uncached_indices, uncached_queries):
                results[idx] = search_documents(q, top_k, trim_for_voice)

    return [r if r is not None else [] for r in results]


# ── Real-Time Upsert & Deletion API ──────────────────────────────────────────

def delete_documents_by_source(source: str) -> int:
    """Delete all vector store entries matching a specific metadata source."""
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            f"DELETE FROM {TABLE} WHERE metadata->>'source' = %s",
            (source,),
        )
        count = cur.rowcount if hasattr(cur, "rowcount") else 0
        cur.close()
    logger.info("[STORE] Deleted %d chunks for source: %s", count, source)
    return count


def upsert_document_realtime(
    text: str,
    source: str = "realtime_upload",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Real-time document ingestion:
    Chunks input text, embeds vectors instantly, and upserts into pgvector.
    Does NOT wipe existing index.
    """
    try:
        from .chunker import chunk_text
    except ImportError:
        from chunker import chunk_text

    meta = metadata or {}
    meta["source"] = source
    meta["timestamp"] = time.time()

    chunks = chunk_text(text, source)
    if not chunks:
        return {"status": "error", "message": "No valid text chunks produced", "chunks_stored": 0}

    for c in chunks:
        c["metadata"].update(meta)

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts).tolist()
    records = build_records(chunks, embeddings)

    inserted = insert_embeddings(records, replace=True)
    logger.info("[STORE] Real-time upsert completed: %d chunks for source '%s'", inserted, source)
    return {
        "status": "success",
        "source": source,
        "chunks_stored": inserted,
        "total_chars": len(text),
    }


