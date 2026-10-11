# Flow Project OKF-RAG-Hybrid

Project ini adalah **RAG engine hybrid**: dokumen dimasukkan ke PostgreSQL + pgvector, lalu dicari dengan gabungan vector search dan BM25, dan hasilnya bisa dikirim ke LLM untuk dijawab. Ada dua jalur utama, yaitu **ingest** (memasukkan data) dan **query/answer** (mencari dan menjawab).

## Komponen runtime (`docker-compose.yml`)

```
Browser ──> ui (nginx :3005) ──/api──> api (FastAPI :8000) ──> postgres (pgvector)
                                              │
                                              └──> LLM provider (Anthropic/OpenAI/Gemini/Ollama)
ingest (profile "ingest", jalan sekali) ─────> postgres
```

Catatan: diagram arsitektur di README menyebut Redis/Kafka, worker cluster, dan GPU embedder. Bagian itu **belum ada di kode**. Kenyataannya semuanya sinkron dalam satu proses FastAPI, dan embedding jalan di CPU.

## Flow 1: Batch ingest (`ingest.py`)

Dijalankan dengan `python ingest.py` atau `docker compose --profile ingest run ingest`.

```
Data/ (raw files)
  │ okf/analyzer.scan        → cari semua file
  │ okf/parser.parse         → ekstrak teks (txt/md/json/csv/docx/pdf);
  │                            kalau ada "Product:", dipecah per produk/QA;
  │                            title diambil dari H1 markdown, fallback ke nama file
  │ okf/classifier.analyze   → title, tipe dokumen, sections, keywords, entities
  │ okf/generator.generate   → OKF doc: Markdown + YAML frontmatter
  │ okf/validator.validate   → gagal = stop
  │ okf/writer.write         → simpan ke okf_output/<kategori>/<slug>.md
  ▼
load_okf_chunks
  │ pisahkan frontmatter, lalu chunker.chunk_text(body)
  │ tiap chunk diberi prefix "Document: <title>"
  ▼
_embed_in_batches            → BAAI/bge-m3, 1024-dim, batch 64
  ▼
build_records                → id = sha256(text + metadata), dedup
  ▼
insert_embeddings            → upsert ke tabel rag_chunks
```

Secara default `ingest()` melakukan **DROP table lalu rebuild**. Pakai `--append` kalau tidak mau tabelnya dihapus, atau `--validate` untuk dry run sampai tahap chunking saja.

### Strategi chunking (`chunker.py`)

- Teks dipecah per section header (heading markdown, `Product:`, huruf kapital semua, dan sejenisnya).
- Section yang lebih panjang dari 1000 karakter dipecah lagi per paragraf, lalu per kalimat. **Heading section di-prepend ke setiap sub-chunk** supaya konteksnya tidak hilang.
- Chunk di bawah 150 karakter dibuang. Overlap sengaja dibuat 0.

### Schema tabel (`vector_store.ensure_schema`)

Kolom `embedding vector(1024)` dengan index HNSW (cosine), plus kolom generated `search_vector tsvector` dengan index GIN untuk full-text. `search_vector` diberi bobot: title OKF (A), tags OKF (B), isi chunk (D). Chunk tanpa frontmatter OKF (dari upsert API) hanya punya bagian D. Tabel lama dengan `search_vector` tanpa bobot dimigrasi otomatis oleh `ensure_schema`.

Model dan dimensi diatur lewat env `LOCAL_EMBEDDING_MODEL` dan `EMBEDDING_DIM`. Mengganti model berarti semua dokumen harus di-ingest ulang.

## Flow 2: Real-time upsert & delete (`api.py`)

- `POST /api/v1/documents/upsert` → `upsert_document_realtime`: chunk, lalu embed, lalu upsert. Jalur ini **tidak lewat pipeline OKF**; teks mentah langsung di-chunk.
- `DELETE /api/v1/documents?source=...` menghapus semua chunk dengan `metadata->>'source'` yang cocok.

## Flow 3: Search (`POST /api/v1/query`)

```
query
  │ intent.should_use_rag    → saat ini selalu True (_ALWAYS_USE_RAG = True)
  ▼
retriever.advanced_search
  │ 1. expand_query          → variasi query dari synonyms di domain_config.json (maks 4)
  │ 2. per variasi: search_documents
  │      ├─ cek semantic cache (key = query yang dinormalisasi)
  │      ├─ embed query
  │      └─ SQL _search:
  │           - ambil kandidat via HNSW (top_k×20, minimal 60)
  │           - BM25: kata-kata query digabung dengan OR (to_tsquery 'a | b | c')
  │           - skor = 0.70 × 1/(1+distance) + 0.30 × min(ts_rank_cd×5, 0.30)
  │ 3. deduplicate           → berdasarkan (source, chunk, 120 karakter pertama)
  │ 4. mmr_select λ=0.65     → re-embed semua kandidat, pilih yang relevan dan tidak redundant
  ▼
top_k dokumen
```

Kalau `hybrid=false`, langkah 1, 3, dan 4 dilewati; hanya `search_documents` yang dijalankan.

## Flow 4: Answer (`POST /api/v1/answer`)

```
knowledge.load_concepts(RAG_KNOWLEDGE_DIR) → konsep OKF terkurasi berstatus active
  │ kalau ada konsep dan use_knowledge=true:
  │   llm.select_concepts → LLM menerima listing (path | title | description) dan membalas JSON path
  │   gagal atau balasan tidak valid → lanjut tanpa konsep, router.error diisi
  ▼
advanced_search(query) → docs
  ▼
llm.build_user_message  → <concepts>[K1] ...</concepts> (kalau ada) + <context>[1] (source: ...) ...</context> + Question
  ▼
llm.generate_answer     → dispatch berdasarkan LLM_PROVIDER
                          (anthropic default, openai, gemini, ollama)
                          system prompt: jawab hanya dari context, sertakan citation [n]
  ▼
{answer, provider, model, count, sources}
```

Response juga berisi `concepts` (konsep yang dikirim ke LLM) dan `router` (`enabled`, `selected`, `error`). Bundle konsep, format frontmatter, dan CLI-nya (`python knowledge.py index|check`) dijelaskan di [`SPEC.md`](../SPEC.md) §2.

Mapping error: konfigurasi LLM salah menghasilkan 503, model menolak (refusal) menghasilkan 422, error lain menghasilkan 500.

## Caching (`cache.py`)

- **Embed cache**: teks (di-lowercase) dipetakan ke vector, supaya teks yang sama tidak di-embed ulang.
- **Semantic cache**: query yang dinormalisasi dipetakan ke hasil search. Disimpan di memory (LRU) dan ditulis ke `rag_cache/*.json` secara async.
- Nama file cache memuat nama model embedding (`embed_cache.<model>.json`), jadi mengganti model tidak memakai vector dari model lama.

## Hal yang perlu diperhatikan

1. **Semantic cache tidak di-invalidate saat upsert atau delete.** Query yang sudah pernah di-cache tetap mengembalikan hasil lama walaupun dokumennya sudah diubah atau dihapus.
2. **Dua jalur ingest menghasilkan bentuk chunk yang berbeda.** Batch ingest lewat OKF (ada prefix `Document: <title>` dan metadata frontmatter), sedangkan API upsert tidak. Format `source` juga berbeda: batch memakai `path#segment_title`, API memakai string bebas. Akibatnya `DELETE ?source=` harus memakai format source yang persis sama.
3. **Ada komentar yang sudah usang.** Docstring dan komentar di `config.py` masih menyebut Gemini dan MiniLM 384-dim, padahal default sekarang `bge-m3` 1024-dim. Di `embedder.py` juga masih ada versi lama yang di-comment.
4. **`/answer` tidak melewati `should_use_rag`**, berbeda dengan `/query`. Untuk sekarang tidak berpengaruh karena nilainya selalu True.
5. **Biaya MMR cukup besar.** Semua kandidat di-embed ulang di setiap request. Dengan `bge-m3` di CPU, median search sekitar 1.5 detik (terburuk sekitar 4.7 detik). Embed cache membantu untuk query yang berulang.
6. **MMR mengabaikan skor hybrid.** `mmr_select` memakai kemiripan vector sebagai relevansi, sehingga kontribusi BM25 hilang di urutan akhir.
7. **Chunker bisa membuang konten.** Section di bawah `CHUNK_MIN` dibuang, dan hard-split bisa memotong kata di tengah.

## Mengukur kualitas retrieval

`eval_retrieval.py` menghitung hit@k dan MRR dari file JSONL berisi pertanyaan berlabel:

```json
{"id": "q1", "query": "...", "relevant": ["potongan teks jawaban"], "source": "bagian path dokumen (opsional)"}
```

```bash
python eval_retrieval.py --cases cases.jsonl --save before.json
python eval_retrieval.py --cases cases.jsonl --baseline before.json
```

Script ini memakai cache sementara, jadi hasilnya tidak terpengaruh semantic cache. Pakai `RAG_PGVECTOR_TABLE` untuk mengarahkannya ke tabel lain.
