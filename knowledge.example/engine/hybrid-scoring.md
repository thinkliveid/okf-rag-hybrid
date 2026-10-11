---
type: rule
title: OKF-RAG-Engine — Hybrid search scoring
description: How vector similarity and BM25 full-text rank are combined into one search score.
tags: [engine, search, bm25]
sources:
  - vector_store.py
generated:
  by: claude-draft
  at: 2026-10-11
verified:
  tier: unverified
status: active
---

Candidates are the nearest neighbours by cosine distance (HNSW index), at least 60 or `top_k × 20`.
Each candidate is scored as `0.70 × 1/(1 + distance) + 0.30 × min(ts_rank_cd × 5, 0.30)`.

The full-text query ORs the query terms. The `search_vector` column weights the OKF title (A) and tags (B)
above the chunk body (D). The final answer set is then reordered by MMR (λ = 0.65).
