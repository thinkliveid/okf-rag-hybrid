---
feature: okf-knowledge-router
project_type: backend
appendix_key: N/A
---

# Dev Plan: OKF Knowledge Router

Sumber: [`SPEC.md`](../../SPEC.md). Urutan pengerjaan: **Bagian 3 → Bagian 1 → Bagian 2**.

Project type: backend Python (FastAPI). Ini repo personal, jadi `appendix_key` adalah N/A dan alur WP/Jira tidak dipakai.

## Dependency graph

```
T1 knowledge loader + CLI ─┬─> T3 llm.complete + select_concepts ─> T4 /answer router ─┬─> T5 UI
T2 example bundle ─────────┘                                                          ├─> T6 eval_router.py
                                                                                       └─> T7 draft konsep
                                                     ── Checkpoint A ──
T8 PyYAML frontmatter ─> T9 generator/validator v0.2
                                                     ── Checkpoint B ──
T1 + T4 ─> T10 stale label konsep
T9 ──────> T11 lifecycle filter chunk
                                                     ── Checkpoint C ──
```

## Keputusan desain (sudah disepakati)

| Topik | Keputusan |
|---|---|
| Pemilihan konsep | LLM membaca listing (path, title, description), lalu membalas JSON |
| Context jawaban | Konsep `[K1]` + passage RAG `[1]`; konsep diutamakan kalau bertentangan |
| Lokasi bundle | `RAG_KNOWLEDGE_DIR`, default `knowledge/` (gitignored); contoh generik di `knowledge.example/` |
| Kurasi awal | Draft dari dokumen dengan `verified.tier: unverified`; di-review oleh pemilik |

## Risiko

| Risiko | Mitigasi |
|---|---|
| Tidak ada API key LLM di environment pengembangan | Router di-test dengan LLM palsu; `eval_router.py` dijalankan pemilik |
| Tambahan latency 1 panggilan LLM per `/answer` | Router hanya aktif kalau ada konsep; `use_knowledge=false` untuk mematikannya |
| Listing makin panjang kalau konsep bertambah banyak | Batas praktis puluhan konsep (sesuai artikel); kalau lebih, pertimbangkan pre-filter embedding |
| Draft konsep salah fakta | Tier `unverified` terlihat di UI dan response; review manual sebelum dinaikkan ke `human` |
| Container API berjalan dengan image lama dan `rag_chunks` masih 384-dim | Test dijalankan di container sementara; migrasi tabel tetap menunggu izin pemilik |

## Fase A: Bagian 3, Curated Knowledge Router

- **T1 — Loader & CLI.** Membuat `knowledge.py`:
  - `load_concepts(dir)`, `render_listing(concepts)`, `write_indexes(dir)`, `check(dir)`;
  - PyYAML di `requirements.txt`, pytest di `requirements-dev.txt`;
  - `RAG_KNOWLEDGE_DIR` dan `RAG_MAX_CONCEPTS` di `config.py`.
- **T2 — Example bundle.** `knowledge.example/` dengan 2–3 konsep generik tentang engine ini sendiri; `knowledge/` masuk `.gitignore`.
- **T3 — LLM helper.** Di `llm.py`:
  - `complete(system, user)` lewat dispatch `PROVIDERS`;
  - `select_concepts(question, concepts, max_n)` beserta parser balasan yang toleran;
  - `generate_answer(question, docs, concepts)` dengan context `[K#]`.
- **T4 — Router di `/answer`.**
  - `use_knowledge` di request, `concepts` dan `router` di response;
  - semua failure mode di SPEC §2.2 kembali ke RAG-only.
- **T5 — UI.** Tab "Tanya" menampilkan konsep yang dipakai beserta badge tier.
- **T6 — `eval_router.py`.** Menghitung precision/recall pemilihan dari field `concepts` di eval case.
- **T7 — Draft konsep.** Konsep untuk pertanyaan eval yang masih gagal ditulis di `knowledge/` dengan tier `unverified`, plus file eval case dengan field `concepts` di scratchpad.

**Checkpoint A**
- pytest hijau;
- `/answer` tanpa folder knowledge berperilaku sama seperti sebelumnya;
- `knowledge.py check` lolos untuk `knowledge/` dan `knowledge.example/`;
- pemilik me-review draft konsep;
- pemilik menjalankan `eval_router.py` dengan API key.

## Fase B: Bagian 1, Frontmatter OKF v0.2

- **T8 — PyYAML untuk frontmatter.** `okf/models._render_frontmatter` memakai `yaml.safe_dump`; `ingest._split_frontmatter` memakai `yaml.safe_load`.
- **T9 — Field v0.2.**
  - generator: `generated`, `sources`, `verified: {tier: machine}`;
  - validator: field wajib baru, `generated.at` divalidasi sebagai ISO-8601.

**Checkpoint B**
- `ingest.py --validate` lolos untuk corpus eval;
- `eval_retrieval.py` menghasilkan angka yang sama dengan baseline `bge-m3` (hit@5 0.784, MRR@5 0.591).

## Fase C: Bagian 2, Lifecycle filtering

- **T10 — Stale konsep.** Konsep yang lewat `stale_after` diberi label di listing dan context, dan response menandai `concepts[].stale: true`.
- **T11 — Lifecycle chunk.**
  - filter `status` di `_search`, dengan fallback `metadata` → `okf_frontmatter`;
  - penalti stale lewat `RAG_STALE_PENALTY`;
  - `stale: true` di hasil.

**Checkpoint C**
- pytest hijau;
- chunk `archived` tidak pernah muncul;
- eval retrieval tanpa field lifecycle menghasilkan angka identik dengan baseline.
