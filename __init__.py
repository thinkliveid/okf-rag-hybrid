"""
rag/__init__.py
───────────────
Public API for the rag package.
"""

from __future__ import annotations

from typing import Any

try:
    from .vector_store import search_documents, search_documents_async
    from .intent import should_use_rag
    from .retriever import advanced_search
except ImportError:
    from vector_store import search_documents, search_documents_async
    from intent import should_use_rag
    from retriever import advanced_search


def query_rag_advanced(
    question: str,
    expanded_queries: list[str] | None = None,
    n_results: int = 5,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Legacy wrapper matching session.py expectations, routed to pgvector."""
    # Combine question and expansions if provided
    queries = [question]
    if expanded_queries:
        queries.extend(expanded_queries)

    # Perform retrieval using search_documents
    seen_ids: set[str] = set()
    merged_docs: list[dict[str, Any]] = []

    for q in queries:
        if not q or not q.strip():
            continue
        docs = search_documents(q, top_k=n_results, trim_for_voice=False)
        for doc in docs:
            doc_id = doc.get("id", doc.get("text", "")[:80])
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                merged_docs.append(doc)

    # Re-rank by similarity score descending
    merged_docs.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
    return merged_docs[:n_results]


def format_context(docs: list[dict[str, Any]]) -> str:
    """Format pgvector documents into context block for session.py."""
    parts = []
    for i, doc in enumerate(docs, 1):
        metadata = doc.get("metadata") or {}
        frontmatter = metadata.get("okf_frontmatter") or {}
        product = frontmatter.get("title") or metadata.get("segment_title") or "N/A"
        chunk_size = len(doc.get("text", ""))
        score = float(doc.get("score", 0.0))
        parts.append(
            f"[{i}] (Product: {product} | Size: {chunk_size}ch | Score: {score:.4f})\n"
            f"{doc.get('text', '')}"
        )
    return "\n\n".join(parts)


__all__ = [
    "search_documents",
    "search_documents_async",
    "should_use_rag",
    "advanced_search",
    "query_rag_advanced",
    "format_context",
]
