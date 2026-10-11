"""Raw document to OKF to pgvector ingestion orchestrator."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

try:
    from .chunker import chunk_text
    from .config import OKF_DIR, RAW_DIR, PGVECTOR_TABLE, EMBEDDING_BATCH_SIZE, print_config_summary
    from .embedder import embed_texts
    from .okf import analyzer, classifier, generator, parser as okf_parser, validator, writer
    from .vector_store import build_records, delete_collection, insert_embeddings
except ImportError:
    _HERE = Path(__file__).resolve().parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    from chunker import chunk_text
    from config import OKF_DIR, RAW_DIR, PGVECTOR_TABLE, EMBEDDING_BATCH_SIZE, print_config_summary
    from embedder import embed_texts
    from okf import analyzer, classifier, generator, parser as okf_parser, validator, writer
    from vector_store import build_records, delete_collection, insert_embeddings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def raw_to_okf(raw_dir: Path = RAW_DIR, okf_dir: Path = OKF_DIR, write_output: bool = True):
    files = analyzer.scan(raw_dir)
    logger.info("Discovered %d files in %s", len(files), raw_dir)
    if not files:
        raise RuntimeError(f"No files found in {raw_dir}")

    parsed = okf_parser.parse(files)
    analyses = classifier.analyze(parsed)
    okf_docs = generator.generate(analyses)
    result = validator.validate(okf_docs)

    for warning in result.warnings:
        logger.warning(warning)
    if not result.valid:
        for error in result.errors:
            logger.error(error)
        raise RuntimeError("OKF validation failed")

    written: list[Path] = []
    if write_output:
        written = writer.write(okf_docs, okf_dir)
        logger.info("Wrote %d OKF documents to %s", len(written), okf_dir)
    return okf_docs, written


def load_okf_chunks(okf_dir: Path = OKF_DIR) -> list[dict]:
    chunks: list[dict] = []
    for path in sorted(okf_dir.rglob("*.md")):
        rel = path.relative_to(okf_dir).as_posix()
        file_chunks = _okf_markdown_to_chunks(path.read_text(encoding="utf-8", errors="ignore"), rel, str(path))
        chunks.extend(file_chunks)
        logger.info("%s -> %d chunks", rel, len(file_chunks))
    return chunks


def _okf_document_to_chunks(okf_doc) -> list[dict]:
    rel = okf_doc.relative_path.as_posix()
    return _okf_markdown_to_chunks(okf_doc.render(), rel, str(okf_doc.source_path))


def _okf_markdown_to_chunks(text: str, rel: str, source: str) -> list[dict]:
    frontmatter, body = _split_frontmatter(text)
    doc_title = frontmatter.get("title", "")
    file_chunks: list[dict] = []
    for chunk in chunk_text(body, rel):
        if doc_title:
            chunk["text"] = f"Document: {doc_title}\n{chunk['text']}"
        metadata = dict(chunk["metadata"])
        metadata["okf_path"] = rel
        metadata["okf_frontmatter"] = frontmatter
        metadata["okf_source"] = source
        chunk["metadata"] = metadata
        file_chunks.append(chunk)
    return file_chunks


def _split_frontmatter(text: str) -> tuple[dict[str, object], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    end_idx = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break

    if end_idx is None:
        return {}, text

    frontmatter_lines = lines[1:end_idx]
    body = "\n".join(lines[end_idx + 1 :]).lstrip()
    frontmatter: dict[str, object] = {}
    current_key: str | None = None
    current_list: list[str] | None = None

    for raw_line in frontmatter_lines:
        line = raw_line.rstrip()
        if not line:
            continue
        if line.startswith("- ") and current_key and current_list is not None:
            current_list.append(line[2:].strip())
            continue
        if ":" in line:
            if current_key and current_list is not None:
                frontmatter[current_key] = current_list
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"')
            current_key = key
            if value:
                frontmatter[key] = value
                current_list = None
            else:
                current_list = []
                frontmatter[key] = current_list
    if current_key and current_list is not None:
        frontmatter[current_key] = current_list
    return frontmatter, body


def _embed_in_batches(texts: list[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> list[list[float]]:
    """
    Embed texts in explicit batches with progress logging and timing.
    This gives fine-grained progress feedback and keeps memory bounded on CPU.
    """
    import numpy as np

    all_embeddings: list[list[float]] = []
    total = len(texts)
    n_batches = (total + batch_size - 1) // batch_size
    t_start = time.time()

    logger.info("[INGEST] Embedding %d chunks in %d batches (batch_size=%d) on CPU...",
                total, n_batches, batch_size)

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch = texts[start:end]

        t_batch = time.time()
        vecs = embed_texts(batch)  # returns np.ndarray shape (len(batch), dim)
        elapsed = time.time() - t_batch

        all_embeddings.extend(vecs.tolist())
        pct = (end / total) * 100
        eta = (time.time() - t_start) / end * (total - end) if end > 0 else 0
        logger.info(
            "[INGEST] Batch %d/%d done — chunks %d-%d | %.1fs/batch | %.1f%% done | ETA %.0fs",
            batch_idx + 1, n_batches, start, end - 1, elapsed, pct, eta,
        )

    total_time = time.time() - t_start
    logger.info("[INGEST] All %d chunks embedded in %.1fs (%.2fs/chunk avg)",
                total, total_time, total_time / total if total else 0)
    return all_embeddings


def ingest(rebuild: bool = True) -> dict[str, int | str]:
    print_config_summary()

    logger.info("Converting raw documents from %s to OKF in %s", RAW_DIR, OKF_DIR)
    raw_to_okf(write_output=True)

    logger.info("Loading and chunking documents from %s", OKF_DIR)
    chunks = load_okf_chunks()
    if not chunks:
        raise RuntimeError(f"No OKF chunks produced from {OKF_DIR}")

    logger.info("[INGEST] Total chunks to embed: %d", len(chunks))

    if rebuild:
        logger.info("Dropping existing pgvector table %s", PGVECTOR_TABLE)
        delete_collection()

    texts = [chunk["text"] for chunk in chunks]
    logger.info("Embedding %d chunks", len(texts))

    # Batch embed for better progress visibility and memory management
    embeddings = _embed_in_batches(texts, batch_size=EMBEDDING_BATCH_SIZE)

    records = build_records(chunks, embeddings)
    inserted = insert_embeddings(records)
    logger.info("Stored %d chunks in pgvector table %s", inserted, PGVECTOR_TABLE)

    return {
        "okf_documents": len(list(OKF_DIR.rglob("*.md"))),
        "chunks": len(chunks),
        "stored": inserted,
        "table": PGVECTOR_TABLE,
    }


def validate() -> None:
    raw_files = analyzer.scan(RAW_DIR)
    okf_docs, _ = raw_to_okf(write_output=False)
    chunks: list[dict] = []
    for okf_doc in okf_docs:
        chunks.extend(_okf_document_to_chunks(okf_doc))
    report = {
        "raw_files": len(raw_files),
        "okf_files": len(okf_docs),
        "chunks": len(chunks),
        "okf_dir": str(OKF_DIR),
    }
    print(json.dumps(report, indent=2))
    if not chunks:
        sys.exit(1)


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Ingest raw docs into OKF and pgvector")
    arg_parser.add_argument("--validate", action="store_true", help="Run raw->OKF validation and chunking only")
    arg_parser.add_argument("--append", action="store_true", help="Do not drop the existing pgvector table")
    args = arg_parser.parse_args()

    if args.validate:
        validate()
        return

    print(json.dumps(ingest(rebuild=not args.append), indent=2))


if __name__ == "__main__":
    main()
