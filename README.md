# ⚡ OKF-RAG-Engine

> **Enterprise-Grade, Domain-Agnostic Hybrid RAG Engine powered by OKF, pgvector HNSW, BM25 Fusion, and Real-Time REST APIs.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 💡 Overview

**OKF-RAG-Engine** is a modern, production-grade Retrieval-Augmented Generation (RAG) framework designed for high-accuracy document search, zero SaaS vector database costs, and real-time streaming ingestion.

Unlike basic RAG tutorials that suffer from context loss, fixed sentence slicing, and vector-only search failures, **OKF-RAG-Engine** standardizes documents into an intermediate **Open Knowledge Format (OKF)**, applies **Section-Aware Context Preservation**, and executes **Hybrid `pgvector` HNSW + BM25 `tsvector` Fusion** directly inside PostgreSQL.

---

## 🔥 Key Features

* 📑 **OKF Document Normalization**: Ingests raw PDFs, DOCX, CSVs, JSON, and Markdown files, normalizing them into structured Markdown with YAML metadata frontmatter.
* 🎯 **Section-Aware Context Chunking**: Preserves section and product headings in split sub-chunks so semantic context is never lost.
* 🔍 **Hybrid Vector + BM25 Search**: Fuses dense vector embeddings (70% weight) with sparse BM25 keyword matching (30% weight) in a single PostgreSQL query.
* 🔄 **MMR Diversity Reranking**: Applies Maximal Marginal Relevance (MMR) to eliminate repetitive chunks and deliver non-redundant search context.
* ⚡ **Real-Time Ingestion REST API**: Ingest, upsert, or delete documents on-the-fly via HTTP requests without needing to wipe database index tables.
* 🚀 **Two-Tier Disk & Memory Caching**: LRU memory caching with asynchronous disk persistence (`embed_cache.json` & `semantic_cache.json`) for zero-latency query hits.
* 🌐 **100% Domain-Agnostic**: Configurable via `domain_config.json` for any industry worldwide (Tech, Legal, Healthcare, Finance, E-Commerce).

---

## 🏗️ Architecture

```
 ┌────────────────┐     ┌───────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
 │ Data Sources   │ ──> │ Real-Time Ingest API  │ ──> │ Task Queue           │ ──> │ Async Worker Cluster │
 │ (Files / Web)  │     │ (REST / Webhooks)     │     │ (Redis / Kafka)      │     │ (Docling / Chunkers) │
 └────────────────┘     └───────────────────────┘     └──────────────────────┘     └──────────────────────┘
                                                                                              │
                                                                                              ▼
 ┌────────────────┐     ┌───────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
 │ RAG Retrieval  │ <── │ Hybrid Query Engine   │ <── │ Vector DB Cluster    │ <── │ Batch Embedder       │
 │ API & UI       │     │ (HNSW Vector + BM25)  │     │ (pgvector / Qdrant)  │     │ (GPU / vLLM / TEI)   │
 └────────────────┘     └───────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

For a walkthrough of how the current code actually flows (ingest, upsert, hybrid search, LLM answer, caching), see [docs/flow.md](docs/flow.md) (in Indonesian).

---

## 📁 Repository Structure

```text
OKF-RAG-Engine/
├── api.py               # Real-time FastAPI REST Server (Upsert, Delete, Query, Health)
├── vector_store.py      # pgvector HNSW & BM25 hybrid search storage layer
├── retriever.py         # Query expansion, BM25 reranking, and MMR diversity selection
├── chunker.py           # Section-aware contextual text splitter
├── embedder.py          # SentenceTransformers neural embedding pipeline (Background Preload)
├── cache.py             # Dual-tier LRU memory + async disk persistence cache
├── intent.py            # Dynamic query routing and intent classifier
├── config.py            # System configuration & environment settings
├── domain_config.json   # Domain-agnostic configuration (Keywords, Synonyms, Chunking)
├── ingest.py            # Batch ingestion pipeline orchestrator
├── requirements.txt     # Dependency declaration file
└── okf/                 # Open Knowledge Format Submodule
    ├── analyzer.py      # File discovery and metadata extraction
    ├── parser.py        # Multi-format raw document text extraction
    ├── classifier.py    # Document categorization & entity extraction
    ├── generator.py     # OKF document generation engine
    ├── validator.py     # OKF document structure validator
    ├── writer.py        # OKF disk writer
    └── models.py        # Shared data contracts
```

---

## 🚀 Quick Start

### 1. Prerequisites & Installation

Clone the repository and install required dependencies:

```bash
git clone https://github.com/your-username/OKF-RAG-Engine.git
cd OKF-RAG-Engine
pip install -r requirements.txt
```

### 2. Start pgvector Database

Run a PostgreSQL container with `pgvector` enabled:

```bash
docker run -d \
  --name pgvector \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=rag_db \
  -p 5432:5432 \
  ankane/pgvector
```

Configure your environment variable:

```bash
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/rag_db"
```

### 3. Run the REST API Server

Start the real-time API server using Uvicorn:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Access the interactive API documentation at: `http://localhost:8000/docs`

### 4. Web UI

A simple static web UI lives in `ui/` (plain HTML/CSS/JS served by nginx, which proxies `/api` to the API service). It covers asking questions (`/answer`), hybrid search (`/query`), and upserting or deleting documents.

```bash
docker compose up -d --build
```

Open `http://localhost:3005` (override with `UI_PORT`). To point the UI at an API running elsewhere, set `API_UPSTREAM` on the `ui` container, e.g. `http://host.docker.internal:8000`.

---

## 📡 REST API Reference

### 1. Real-Time Document Upsert
**`POST /api/v1/documents/upsert`**

Upsert raw text into the vector index in real time.

```json
{
  "text": "# API Integration Guide\nTo integrate the payment endpoint, send a POST request with payload...",
  "source": "docs/api_guide.md",
  "metadata": {
    "category": "developer_docs",
    "author": "Tech Team"
  }
}
```

### 2. Execute Hybrid Search Query
**`POST /api/v1/query`**

Perform hybrid vector + BM25 search with MMR diversity reranking.

```json
{
  "query": "How to integrate payment API?",
  "top_k": 5,
  "hybrid": true
}
```

### 3. Delete Document by Source
**`DELETE /api/v1/documents?source=docs/api_guide.md`**

Deletes all vectors associated with the specified source ID.

---

## ⚙️ Customization (`domain_config.json`)

Adapt the engine for any specific domain (Healthcare, Legal, E-Commerce, Tech) without changing code:

```json
{
  "domain_name": "Tech Documentation RAG",
  "routing": {
    "always_use_rag": true,
    "intent_keywords": ["api", "sdk", "authentication", "setup"]
  },
  "synonyms": {
    "auth": ["authentication", "oauth", "jwt"],
    "db": ["database", "postgres", "sql"]
  },
  "chunking": {
    "min_chars": 150,
    "max_chars": 1000,
    "overlap": 0
  }
}
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to submit a Pull Request or open an Issue for feature requests or bug fixes.

---
