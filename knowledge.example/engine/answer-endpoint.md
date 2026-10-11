---
type: api
title: OKF-RAG-Engine — Answer endpoint
description: Request and response shape of POST /api/v1/answer, including the curated knowledge router.
tags: [engine, api, answer]
sources:
  - api.py
generated:
  by: claude-draft
  at: 2026-10-11
verified:
  tier: unverified
status: active
---

`POST /api/v1/answer` takes `query`, `top_k` (1–50, default 5) and `use_knowledge` (default `true`).

When curated concepts exist, a first LLM call picks relevant concepts from the knowledge index; their full
text is sent to the answering call as `[K#]` passages alongside the RAG passages `[#]`.
The response contains `answer`, `provider`, `model`, `count`, `sources`, `concepts` and `router`.
