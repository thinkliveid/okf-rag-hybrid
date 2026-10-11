---
type: fact
title: OKF-RAG-Engine — Embedding model
description: Which embedding model and vector dimension the engine uses, and what changing it requires.
tags: [engine, embedding]
sources:
  - config.py
  - Dockerfile
generated:
  by: claude-draft
  at: 2026-10-11
verified:
  tier: unverified
status: active
---

The default embedding model is `BAAI/bge-m3`, a multilingual model producing 1024-dimensional vectors.
It is configured with `LOCAL_EMBEDDING_MODEL` and `EMBEDDING_DIM`, and baked into the API image at build time.

Changing the model invalidates every stored vector: all documents must be re-ingested, and a table created
with a different dimension cannot be reused. See [hybrid scoring](hybrid-scoring.md) for how vectors are ranked.
