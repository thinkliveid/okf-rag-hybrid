"""
api.py
──────
Real-Time Document Upsert & RAG Search REST API.

Provides HTTP endpoints for:
  - POST /api/v1/documents/upsert : Real-time document chunking, embedding & vector upsert
  - DELETE /api/v1/documents      : Delete document vectors by source ID
  - POST /api/v1/query             : Execute real-time hybrid search & retrieval
  - GET  /api/v1/health            : Service health check
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Body
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

try:
    from .vector_store import (
        delete_documents_by_source,
        upsert_document_realtime,
        search_documents,
    )
    from .retriever import advanced_search, doc_to_dict
    from .intent import should_use_rag
    from .llm import LLMConfigError, LLMRefusalError, generate_answer
except ImportError:
    from vector_store import (
        delete_documents_by_source,
        upsert_document_realtime,
        search_documents,
    )
    from retriever import advanced_search, doc_to_dict
    from intent import should_use_rag
    from llm import LLMConfigError, LLMRefusalError, generate_answer

logger = logging.getLogger("rag_api")
logging.basicConfig(level=logging.INFO)

if HAS_FASTAPI:
    app = FastAPI(
        title="Open Universal RAG Engine API",
        description="Domain-Agnostic, High-Scale, Real-Time Ingestion RAG API",
        version="2.0.0",
    )

    class DocumentUpsertRequest(BaseModel):
        text: str = Field(..., description="Raw text content of the document")
        source: str = Field("realtime_doc", description="Unique source path or document ID")
        metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom document metadata")

    class QueryRequest(BaseModel):
        query: str = Field(..., description="User search query")
        top_k: int = Field(5, ge=1, le=100, description="Number of context results to return")
        hybrid: bool = Field(True, description="Enable hybrid semantic + BM25 search & MMR diversity")

    class AnswerRequest(BaseModel):
        query: str = Field(..., description="User question")
        top_k: int = Field(5, ge=1, le=50, description="Number of context passages sent to the LLM")

    @app.get("/api/v1/health")
    def health():
        return {"status": "ok", "engine": "Open-RAG-Engine", "version": "2.0.0"}

    @app.post("/api/v1/documents/upsert")
    def upsert_document(payload: DocumentUpsertRequest):
        try:
            res = upsert_document_realtime(
                text=payload.text,
                source=payload.source,
                metadata=payload.metadata,
            )
            return res
        except Exception as exc:
            logger.error("[API] Upsert failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    @app.delete("/api/v1/documents")
    def delete_document(source: str):
        try:
            deleted_count = delete_documents_by_source(source)
            return {"status": "success", "source": source, "chunks_deleted": deleted_count}
        except Exception as exc:
            logger.error("[API] Delete failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/v1/query")
    def query_rag(payload: QueryRequest):
        try:
            if not should_use_rag(payload.query):
                return {"use_rag": False, "results": []}

            if payload.hybrid:
                docs = advanced_search(payload.query, top_k=payload.top_k)
                results = [doc_to_dict(d) for d in docs]
            else:
                results = search_documents(payload.query, top_k=payload.top_k)

            return {"use_rag": True, "count": len(results), "results": results}
        except Exception as exc:
            logger.error("[API] Query failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/v1/answer")
    def answer_rag(payload: AnswerRequest):
        try:
            docs = [doc_to_dict(d) for d in advanced_search(payload.query, top_k=payload.top_k)]
            answer = generate_answer(payload.query, docs)
            return {
                "answer": answer.text,
                "provider": answer.provider,
                "model": answer.model,
                "count": len(docs),
                "sources": docs,
            }
        except LLMConfigError as exc:
            logger.error("[API] LLM misconfigured: %s", exc)
            raise HTTPException(status_code=503, detail=str(exc))
        except LLMRefusalError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error("[API] Answer failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

else:
    # Lightweight fallback server script helper
    logger.warning("FastAPI not installed. To run the REST API server: pip install fastapi uvicorn")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
