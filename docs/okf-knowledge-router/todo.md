# TODO: okf-knowledge-router

Status: `todo` | `in-progress` | `done` | `failed`

## Fase A: Curated Knowledge Router

### T1 — Knowledge loader & CLI
- Status: done
- Files: `knowledge.py`, `config.py`, `requirements.txt`, `requirements-dev.txt`, `tests/test_knowledge.py`
- Acceptance:
  - `load_concepts` memuat file `*.md` (kecuali `index.md` dan `log.md`), mem-parse frontmatter dengan PyYAML, dan memakai path relatif `/` sebagai ID.
  - File tanpa `type`, `title`, atau `description` di-skip dengan warning, tanpa exception.
  - Konsep dengan `status` selain `active` dikeluarkan; `status` kosong dianggap `active`.
  - Folder tidak ada menghasilkan list kosong.
  - `render_listing` deterministik (diurutkan per path).
  - `python knowledge.py index` menulis `index.md` per folder; dijalankan dua kali menghasilkan isi yang identik.
  - `python knowledge.py check` exit 1 kalau ada file invalid.
- Verify: `python -m pytest tests/test_knowledge.py -q`

### T2 — Example bundle
- Status: done
- Files: `knowledge.example/**`, `.gitignore`
- Acceptance:
  - Ada 2–3 konsep generik tentang engine ini sendiri (misalnya embedding model, hybrid scoring); isinya tidak berasal dari dokumen pribadi.
  - `knowledge/` di-gitignore.
  - `knowledge.py check` lolos untuk bundle contoh.
- Verify: `RAG_KNOWLEDGE_DIR=knowledge.example python knowledge.py check`

### T3 — LLM helpers
- Status: done
- Files: `llm.py`, `tests/test_llm_router.py`
- Acceptance:
  - `complete(system, user)` memakai dispatch `PROVIDERS` yang sama dengan `generate_answer`.
  - `parse_selection(text, known_paths, max_n)` menerima JSON murni maupun JSON yang dibungkus teks atau code fence, membuang path yang tidak dikenal dan duplikat, dan memotong ke `max_n`. Teks sampah memunculkan `ValueError`.
  - `generate_answer(question, docs, concepts=())` merender `<concepts>[K1] (concept: path, verified: tier) body</concepts>` sebelum `<context>`; tanpa konsep, pesan yang dikirim identik dengan sekarang.
- Verify: `python -m pytest tests/test_llm_router.py -q`

### T4 — Router di /answer
- Status: done
- Files: `api.py`, `tests/test_answer_router.py`
- Acceptance:
  - Request menerima `use_knowledge` (default `true`); response berisi `concepts` dan `router`.
  - Tanpa konsep atau dengan `use_knowledge=false`: tidak ada panggilan pemilihan, `router.enabled=false`, `concepts=[]`.
  - Error LLM pemilihan atau balasan yang tidak valid: `router.error` diisi, jawaban tetap dihasilkan dari RAG.
  - `LLMConfigError` tetap menghasilkan 503.
- Verify: `python -m pytest tests/test_answer_router.py -q` (memakai FastAPI TestClient, LLM dan `advanced_search` di-fake)

### T5 — UI konsep
- Status: done
- Files: `ui/app.js`, `ui/style.css`
- Acceptance: tab "Tanya" menampilkan daftar konsep (title, path, badge tier) kalau `concepts` tidak kosong; kalau kosong, tampilan sama seperti sekarang.
- Verify: buka UI dan render response contoh; `node --check ui/app.js`

### T6 — eval_router.py
- Status: done
- Files: `eval_router.py`
- Acceptance:
  - Membaca eval case dengan field opsional `concepts`.
  - Menghitung precision, recall, dan exact-match pemilihan per case beserta ringkasannya.
  - Mendukung `--save` dan `--baseline`.
- Verify: dijalankan dengan provider palsu untuk smoke test lokal; eval yang sebenarnya dijalankan pemilik dengan API key.

### T7 — Draft konsep
- Status: done
- Files: `knowledge/**` (gitignored), eval case di scratchpad
- Acceptance:
  - Ada konsep untuk pertanyaan eval yang masih gagal (database, tech stack, deploy, status job, dan sejenisnya), semuanya dengan `verified.tier: unverified` dan `sources` terisi.
  - `knowledge.py check` lolos; `index.md` sudah di-generate.
- Verify: `knowledge.py check`; `git status` tidak menampilkan file dari `knowledge/`.

### Checkpoint A
- Status: in-progress (menunggu review draft konsep dan eval_router dengan API key)
- pytest hijau, `/answer` tanpa knowledge tidak berubah, draft konsep sudah di-review pemilik.

## Fase B: Frontmatter OKF v0.2

### T8 — PyYAML untuk frontmatter
- Status: todo
- Files: `okf/models.py`, `ingest.py`, `tests/test_okf_frontmatter.py`
- Acceptance: render → parse round-trip mempertahankan nested map dan list; frontmatter OKF lama (format v0.1) tetap bisa di-parse.
- Verify: `python -m pytest tests/test_okf_frontmatter.py -q`

### T9 — Field v0.2 di generator & validator
- Status: todo
- Files: `okf/generator.py`, `okf/validator.py`
- Acceptance:
  - Generator menulis `generated: {by, at}`, `sources: [...]`, dan `verified: {tier: machine}`; `timestamp` dan `source` dihapus.
  - Validator mewajibkan `id`, `type`, `title`, `description`, `generated.at`.
- Verify: `ingest.py --validate` pada corpus eval; `eval_retrieval.py` menghasilkan angka sama dengan baseline `bge-m3`.

### Checkpoint B
- Status: todo

## Fase C: Lifecycle filtering

### T10 — Stale konsep
- Status: todo
- Files: `knowledge.py`, `llm.py`, `api.py`, tests
- Acceptance: konsep dengan `stale_after` yang sudah lewat diberi label `stale` di listing dan context; response `concepts[].stale = true`.
- Verify: pytest

### T11 — Lifecycle chunk
- Status: todo
- Files: `vector_store.py`, `config.py`, tests
- Acceptance:
  - Chunk dengan `status` yang ada dan bukan `active` dikecualikan.
  - Chunk stale mendapat skor `× RAG_STALE_PENALTY` dan `stale: true`.
  - Field lifecycle dibaca dari metadata top-level, dengan fallback ke `okf_frontmatter`.
- Verify: pytest dengan tabel sementara; eval retrieval menghasilkan angka identik dengan baseline.

### Checkpoint C
- Status: todo
