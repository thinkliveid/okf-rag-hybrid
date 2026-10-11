"""Advanced retrieval helpers on top of the pgvector store."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from .config import TOP_K
    from .embedder import embed_texts
    from .vector_store import search_documents
except ImportError:
    from config import TOP_K
    from embedder import embed_texts
    from vector_store import search_documents

logger = logging.getLogger(__name__)


@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


def _tokenize(text: str) -> list[str]:
    return [tok for tok in re.findall(r"\w+", text.lower()) if len(tok) > 2]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


# ── Query Expansion ───────────────────────────────────────────────────────────

import json
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent / "domain_config.json"


def _load_synonyms() -> dict[str, list[str]]:
    """Load dynamic domain synonyms from configuration if present."""
    if _CONFIG_PATH.exists():
        try:
            data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            return data.get("synonyms", {})
        except Exception as exc:
            logger.warning("[RETRIEVER] Failed to load domain synonyms: %s", exc)
    return {}


# Domain synonym map for abbreviation expansion
_SYNONYMS: dict[str, list[str]] = _load_synonyms()



def expand_query(query: str) -> list[str]:
    """Expand query with synonym variants for better recall."""
    queries = [query]
    q_lower = query.lower()

    for term, synonyms in _SYNONYMS.items():
        if term in q_lower:
            for syn in synonyms[:2]:  # Limit to 2 synonyms per term to avoid explosion
                variant = q_lower.replace(term, syn)
                if variant not in queries:
                    queries.append(variant)

    return queries[:4]  # Cap total at 4 variants


# ── BM25 Search ───────────────────────────────────────────────────────────────

def bm25_search(query: str, chunks: list[dict], top_k: int = TOP_K) -> list[Document]:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("[RETRIEVER] rank_bm25 not installed; relying on pgvector full-text rank.")
        return []

    if not chunks:
        return []
    tokenized = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(_tokenize(query))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        Document(text=chunks[i]["text"], metadata=chunks[i].get("metadata", {}), score=float(scores[i]))
        for i in top_indices
        if scores[i] > 0
    ]


# ── Reranking ─────────────────────────────────────────────────────────────────

def rerank(query: str, docs: list[Document], top_k: int = TOP_K) -> list[Document]:
    """Cross-encode rerank using cosine similarity against query embedding."""
    if not docs:
        return []

    query_emb = embed_texts([query])[0]
    doc_embs = embed_texts([doc.text for doc in docs])
    scored = [(doc, _cosine(query_emb, doc_embs[i])) for i, doc in enumerate(docs)]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [Document(text=doc.text, metadata=doc.metadata, score=score) for doc, score in scored[:top_k]]


# ── MMR (Maximal Marginal Relevance) ─────────────────────────────────────────

def mmr_select(
    query_emb: np.ndarray,
    docs: list[Document],
    doc_embs: list[np.ndarray],
    top_k: int,
    lambda_: float = 0.65,
) -> list[Document]:
    """
    Select a diverse, relevant set of documents using MMR.
    lambda_ controls the relevance-diversity tradeoff:
      - 1.0 = pure relevance (no diversity)
      - 0.0 = pure diversity
    """
    if not docs:
        return []

    selected_indices: list[int] = []
    remaining = list(range(len(docs)))

    for _ in range(min(top_k, len(docs))):
        best_idx = -1
        best_score = -float("inf")

        for i in remaining:
            relevance = _cosine(query_emb, doc_embs[i])
            if not selected_indices:
                redundancy = 0.0
            else:
                redundancy = max(_cosine(doc_embs[i], doc_embs[j]) for j in selected_indices)
            mmr_score = lambda_ * relevance - (1 - lambda_) * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        if best_idx == -1:
            break
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [
        Document(text=docs[i].text, metadata=docs[i].metadata, score=docs[i].score)
        for i in selected_indices
    ]


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate(docs: list[Document]) -> list[Document]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[Document] = []
    for doc in docs:
        key = (doc.metadata.get("source"), doc.metadata.get("chunk"), doc.text[:120])
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def doc_to_dict(doc: Document) -> dict[str, Any]:
    return {"text": doc.text, "metadata": doc.metadata, "score": doc.score}


# ── Main Advanced Search ──────────────────────────────────────────────────────

def advanced_search(query: str, chunks: list[dict] | None = None, top_k: int = TOP_K) -> list[Document]:
    """
    Advanced hybrid search with:
    1. Multi-query expansion (synonym-based)
    2. pgvector hybrid search (vector + BM25 ts_rank)
    3. Optional BM25 re-scoring on provided chunks
    4. Deduplication
    5. MMR diversity selection for varied, relevant results
    """
    docs: list[Document] = []
    expanded_queries = expand_query(query)

    for sub_query in expanded_queries:
        for result in search_documents(sub_query, top_k=top_k, trim_for_voice=False):
            docs.append(
                Document(
                    text=result["text"],
                    metadata=result.get("metadata", {}),
                    score=float(result.get("score", 0.0)),
                )
            )

    if chunks:
        docs.extend(bm25_search(query, chunks, top_k=top_k))

    deduped = deduplicate(docs)

    if not deduped:
        return []

    # Use MMR for diverse, relevant final selection
    try:
        query_emb = embed_texts([query])[0]
        doc_embs_arr = embed_texts([d.text for d in deduped])
        doc_embs = [doc_embs_arr[i] for i in range(len(deduped))]
        return mmr_select(query_emb, deduped, doc_embs, top_k=top_k, lambda_=0.65)
    except Exception as e:
        logger.warning("[RETRIEVER] MMR failed, falling back to score sort: %s", e)
        deduped.sort(key=lambda doc: doc.score, reverse=True)
        return deduped[:top_k]
