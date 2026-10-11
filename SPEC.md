# SPEC: OKF Alignment & Curated Knowledge Router

Status: draft · 2026-10-11

## 1. Objective

Saat ini modul `okf/` hanya menormalisasi dokumen mentah menjadi Markdown + frontmatter, lalu langsung di-chunk dan di-embed. Nilai utama OKF, yaitu fakta terkurasi yang dibaca langsung oleh agent, belum dipakai sama sekali. Eval retrieval (37 pertanyaan, `bge-m3`) menunjukkan bahwa pertanyaan yang masih gagal kebanyakan berupa fakta ringkas: database apa, tech stack apa, cara deploy, status job. Pertanyaan seperti ini lebih tepat dijawab dari konsep terkurasi daripada lewat vector search.

Spec ini mencakup tiga bagian, yang dikerjakan dengan urutan **3 → 1 → 2**:

| # | Bagian | Hasil |
|---|---|---|
| 3 | Curated knowledge router | `/answer` memilih konsep OKF terkurasi lewat index sebelum RAG, lalu menjawab dari konsep + passage RAG |
| 1 | Frontmatter OKF v0.2 | Output `okf/generator` memakai `generated`, `sources`, `verified`, bukan `timestamp`/`source` |
| 2 | Lifecycle filtering | `status` dan `stale_after` dipakai saat retrieval: konsep/chunk non-aktif dikeluarkan, yang stale ditandai |

Pengguna: operator yang memakai `/answer` lewat UI atau API, serta kurator yang menulis dan me-review konsep.

Referensi format: artikel [OKF vs RAG](https://geotoolbox.ai/blog/okf-vs-rag), yang merangkum OKF v0.2 dari Google Cloud. Detail field di spec ini mengikuti deskripsi artikel tersebut dan **belum dicocokkan dengan spec resmi**. Kalau nanti ada perbedaan, spec resmi yang dipakai.

## 2. Bagian 3: Curated Knowledge Router

### 2.1 Knowledge bundle

Lokasi diatur lewat env `RAG_KNOWLEDGE_DIR`, dengan default `knowledge/` relatif ke root repo. Folder ini **di-gitignore** karena isinya berasal dari dokumen pribadi. Repo hanya menyertakan bundle contoh generik di `knowledge.example/`.

```
knowledge/
  index.md                 ← di-generate, jangan diedit manual
  healthbot/
    index.md               ← di-generate
    tech-stack.md
    database.md
  ugc-v5-studio/
    index.md
    job-states.md
```

Satu file berisi satu konsep. Frontmatter:

```yaml
---
type: fact                     # wajib: fact | architecture-decision | schema | rule | api | metric | policy
title: HealthBot Research — Database   # wajib
description: Database dan hosting yang dipakai HealthBot Research.   # wajib, satu kalimat; dipakai router
tags: [healthbot, database]
sources:
  - BLUEPRINT.md
generated:
  by: claude-draft
  at: 2026-10-11
verified:
  tier: unverified             # unverified | machine | human
  by: null                     # isi "human:<nama>" kalau tier = human
status: active                 # active | deprecated | archived; default active
stale_after: 2027-04-11        # opsional
---
```

Body berisi Markdown biasa. Link ke konsep lain memakai link relatif, misalnya `[Tech stack](tech-stack.md)`.

**Aturan loader (`knowledge.py`):**
- Membaca semua `*.md` di bawah `RAG_KNOWLEDGE_DIR`, kecuali `index.md` dan `log.md`.
- Frontmatter di-parse dengan PyYAML (`yaml.safe_load`).
- File yang tidak punya `type`, `title`, atau `description` di-skip dengan warning; loader tidak crash.
- Konsep dengan `status` selain `active` tidak dimuat ke router.
- Path konsep (relatif terhadap `RAG_KNOWLEDGE_DIR`, memakai `/`) dipakai sebagai ID konsep.
- Folder tidak ada atau kosong: router mati, dan `/answer` berjalan persis seperti sekarang tanpa panggilan LLM tambahan.

**CLI index:** `python knowledge.py index` menulis `index.md` di root dan di setiap subfolder, berisi daftar `- [title](path) — description`. Router **tidak** membaca `index.md` dari disk; listing yang sama di-render langsung dari frontmatter, jadi router selalu sinkron dengan isi konsep. `index.md` tetap dibuat untuk dibaca manusia dan agent lain.

### 2.2 Alur `/api/v1/answer`

```
question
  │ 1. knowledge.load_concepts()           → konsep aktif (bisa kosong)
  │ 2. kalau ada konsep:
  │      llm.complete(SELECT_PROMPT, listing + question)
  │      → JSON {"concepts": ["healthbot/database.md", ...]}
  │      → parse, buang path yang tidak dikenal, maksimal RAG_MAX_CONCEPTS (default 3)
  │ 3. advanced_search(question, top_k)    → passage RAG (tetap jalan)
  │ 4. llm.generate_answer(question, docs, concepts=terpilih)
  │      context: <concepts>[K1] ...</concepts> + <context>[1] ...</context>
  ▼
{answer, provider, model, count, sources, concepts, router}
```

- **Prompt pemilihan:** model hanya menerima listing (path, title, description), tidak menerima isi konsep. Model wajib membalas JSON saja, dan `{"concepts": []}` kalau tidak ada konsep yang relevan.
- **Prompt jawaban:** konsep dikutip sebagai `[K1]`, `[K2]` dan passage RAG sebagai `[1]`, `[2]`. Kalau keduanya bertentangan, konsep yang diutamakan. Kalau konsep yang dipakai berstatus `verified.tier = unverified`, jawaban tetap boleh memakainya.
- **Field response baru:**
  - `concepts`: `[{path, title, verified_tier}]` yang dikirim ke LLM jawaban;
  - `router`: `{"enabled": bool, "selected": [...], "error": str | null}`.
- **Request baru:** `use_knowledge: bool = true`, untuk A/B test dan untuk mematikan router per request.

**Failure mode, semuanya harus kembali ke jalur RAG-only dan tidak boleh membuat request gagal:**

| Kondisi | Perilaku |
|---|---|
| Folder knowledge tidak ada atau kosong | Router mati, tidak ada panggilan LLM pemilihan |
| Panggilan LLM pemilihan error | Log warning, `router.error` diisi, lanjut RAG-only |
| Balasan pemilihan bukan JSON valid | Log warning, `router.error` diisi, lanjut RAG-only |
| Path yang dipilih tidak dikenal | Path tersebut dibuang diam-diam |
| `LLMConfigError` (provider belum dikonfigurasi) | Tetap 503, sama seperti sekarang |

### 2.3 UI

Tab "Tanya" menampilkan konsep yang dipakai (title, path, dan badge `unverified`/`human`) di atas daftar passage RAG.

### 2.4 Acceptance criteria

1. Tanpa folder knowledge, `/answer` mengembalikan field yang sama seperti sekarang ditambah `concepts: []` dan `router.enabled: false`, dan hanya ada satu panggilan LLM.
2. Dengan konsep yang relevan, LLM jawaban menerima isi konsep tersebut, dan response mencantumkannya di `concepts`.
3. Kegagalan LLM pemilihan tidak menyebabkan request gagal.
4. Konsep dengan `status: deprecated` tidak pernah masuk listing.
5. `python knowledge.py index` menghasilkan `index.md` yang deterministik (dijalankan dua kali menghasilkan isi yang sama).
6. Draft konsep untuk pertanyaan eval yang masih gagal tersedia di `knowledge/` dengan `verified.tier: unverified`.
7. Tidak ada isi dokumen pribadi yang masuk ke git.

## 3. Bagian 1: Frontmatter OKF v0.2

- `okf/generator` mengganti `timestamp` dengan `generated: {by: okf-generator, at: <ISO-8601>}`, dan `source` dengan `sources: [<path>]`.
- Menambahkan `verified: {tier: machine}`, karena output generator otomatis tidak boleh mengaku sudah di-review manusia.
- `okf/models._render_frontmatter` dan parser frontmatter di `ingest.py` diganti dengan PyYAML (`yaml.safe_dump` / `yaml.safe_load`), supaya nested map terbaca dengan benar.
- `okf/validator`: field wajib menjadi `id`, `type`, `title`, `description`, `generated.at`; `generated.at` divalidasi sebagai ISO-8601.
- Kolom `search_vector` tetap membaca `okf_frontmatter->>'title'` dan `->>'tags'`, sehingga tidak perlu migrasi DB.

**Acceptance:** `python ingest.py --validate` tetap lolos untuk corpus eval, dan frontmatter hasilnya bisa di-parse ulang dengan `yaml.safe_load` tanpa kehilangan field.

## 4. Bagian 2: Lifecycle filtering

- **Konsep (router):**
  - konsep dengan `status != active` sudah dikeluarkan di Bagian 3;
  - konsep yang lewat `stale_after` tetap dimuat, tapi diberi label `stale` di listing dan di context jawaban, dan response menandai `concepts[].stale: true`.
- **Chunk (RAG):**
  - lifecycle dibaca dari `metadata->>'status'` dan `metadata->>'stale_after'`, dengan fallback ke `metadata->'okf_frontmatter'`;
  - upsert API meneruskan kedua field ini apa adanya lewat `metadata`;
  - `_search` mengecualikan chunk dengan `status` yang ada dan bukan `active`;
  - chunk yang stale mendapat penalti skor (`score × RAG_STALE_PENALTY`, default 0.85) dan field `stale: true` di hasil.
- Tidak ada perubahan schema tabel; filter memakai expression di `WHERE` atas kolom JSONB.

**Acceptance:**
- chunk dengan `status: archived` tidak pernah muncul di `/query` maupun `/answer`;
- chunk stale tetap bisa muncul, tapi ditandai;
- eval retrieval pada corpus tanpa field lifecycle menghasilkan angka yang sama persis dengan sebelumnya.

## 5. Commands

```bash
docker compose up -d --build                          # postgres + api + ui
docker compose --profile ingest run --rm ingest       # batch ingest Data/
python knowledge.py index                             # regenerate index.md di RAG_KNOWLEDGE_DIR
python knowledge.py check                             # validasi frontmatter semua konsep
docker compose run --rm -v "$PWD:/app" api python -m pytest tests -q
python eval_retrieval.py --cases <cases.jsonl> [--baseline before.json]
python eval_router.py --cases <cases.jsonl>           # butuh API key LLM
```

## 6. Struktur project (perubahan)

```
knowledge.py              ← baru: loader, listing, CLI index/check
knowledge.example/        ← baru: bundle contoh generik (di-commit)
knowledge/                ← baru: bundle asli (gitignored)
llm.py                    ← tambah complete(), select_concepts(), context konsep
api.py                    ← router di /answer, field request/response baru
config.py                 ← RAG_KNOWLEDGE_DIR, RAG_MAX_CONCEPTS, RAG_STALE_PENALTY
eval_router.py            ← baru: akurasi pemilihan konsep
tests/                    ← baru: pytest untuk knowledge + router (LLM di-fake)
requirements.txt          ← tambah pyyaml
requirements-dev.txt      ← baru: pytest
ui/app.js                 ← tampilkan konsep di tab Tanya
```

## 7. Code style

- Tidak ada komentar maupun docstring baru, di file apa pun. Kode harus menjelaskan dirinya sendiri lewat penamaan.
- Ikuti idiom yang sudah ada:
  - pola import ganda `try: from .x import ... except ImportError: from x import ...`;
  - prefix log `[ROUTER]` / `[KNOWLEDGE]`;
  - type hints dengan `from __future__ import annotations`;
  - konfigurasi lewat `os.getenv` di `config.py`.
- Provider LLM tetap pluggable: `complete(system, user)` memakai dispatch `PROVIDERS` yang sudah ada, tanpa logic khusus per provider.

## 8. Testing strategy

- **Unit (pytest, tanpa DB dan tanpa LLM):**
  - parsing frontmatter;
  - skip file invalid;
  - filter status;
  - render listing deterministik;
  - parsing balasan pemilihan (JSON valid, JSON di dalam teks, sampah, path tidak dikenal, melebihi maksimum).
- **Router (pytest, LLM di-fake lewat monkeypatch `llm.complete`, `advanced_search` di-stub):** setiap baris di tabel failure mode §2.2 punya test-nya sendiri.
- **Eval router (manual, butuh API key):** `eval_router.py` membaca field `concepts` di file eval case, lalu menghitung precision/recall pemilihan. Ini tidak bisa dijalankan di sesi ini karena belum ada API key LLM.
- **Regresi:**
  - `eval_retrieval.py` di corpus eval harus menghasilkan angka yang sama setelah Bagian 1;
  - setelah Bagian 2, angka juga harus sama untuk corpus tanpa field lifecycle.

## 9. Boundaries

**Selalu**
- Jalur RAG-only harus tetap berfungsi tanpa folder knowledge dan tanpa panggilan LLM tambahan.
- Konsep hasil draft otomatis diberi `verified.tier: unverified`; hanya kurator yang boleh menaikkannya ke `human`.
- Isi dokumen pribadi hanya ditulis ke `knowledge/` (gitignored) atau scratchpad.

**Tanya dulu**
- Operasi DB destruktif (drop/rename tabel, hapus data), migrasi schema.
- Menambah dependency berat, atau dependency selain `pyyaml` dan `pytest`.
- Mengubah prompt sistem jawaban yang sudah ada, di luar penambahan aturan konsep.

**Jangan**
- Commit isi `knowledge/`, eval case pribadi, atau secret.
- Mengirim isi konsep ke LLM pemilihan; LLM pemilihan hanya menerima listing.
- Menandai konsep sebagai `verified: human` atas nama Anda.
