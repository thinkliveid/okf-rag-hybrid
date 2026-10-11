"""
chunker.py
──────────
Text chunking for the ingest pipeline.
Never imported at runtime — only used by ingest.py.

chunk_text(text, source) → list of {"text": ..., "metadata": ...}

Strategy:
  1. First, split the raw text on SECTION HEADERS — lines that look like
     product names, section headings, or lettered categories (e.g.
     "JS Ready Finance", "Product: ...", "A. EXPORTS", "D. INLAND TRADE").
     This guarantees every product/section starts its own chunk with its
     title intact, which is critical for semantic search accuracy.

  2. Each section is then sub-split by max character length if it exceeds
     CHUNK_MAX, using paragraph and sentence boundaries.
"""

import re
import logging
from typing import Any

try:
    from .config import CHUNK_MIN, CHUNK_MAX, CHUNK_OVERLAP
except ImportError:
    from config import CHUNK_MIN, CHUNK_MAX, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


# ── Section header detection ──────────────────────────────────────────────────

# Patterns that indicate the START of a new product / section.
# A line matches if it looks like a heading on its own line.
_SECTION_HEADER_PATTERNS = [
    # Markdown H1 / H2 / H3 headings (e.g. "# Section", "## Title")
    re.compile(r"^\s*#{1,6}\s+[A-Z0-9]", re.MULTILINE),
    # "Product: <Something>" or "Section: <Something>"
    re.compile(r"^\s*(?:Product|Section|Category|Topic)\s*:", re.MULTILINE | re.IGNORECASE),
    # Lettered category headers (e.g. "A. EXPORTS", "B. DETAILS")
    # NOTE: (?![QqAa]\.) excludes Q&A lines like "Q. What is..." and "A. The answer..."
    re.compile(r"^\s*(?![QqAa]\.\s)[A-Z]\.\s+[A-Z]{2,}", re.MULTILINE),
    # Numbered major sections (e.g. "1. OVERVIEW")
    re.compile(r"^\s*\d+\.\s+[A-Z][A-Z ]", re.MULTILINE),
    # All-caps headings (e.g. "GENERAL TERMS AND CONDITIONS")
    re.compile(r"^\s*[A-Z][A-Z\s\-–&]{5,}\s*$", re.MULTILINE),
    # Common standard headings like "FAQs", "Overview", etc.
    re.compile(r"^\s*FAQs\s*$", re.IGNORECASE | re.MULTILINE),
    # For title-cased headings: require at least 2 words and >= 10 chars
    re.compile(r"^\s*([A-Z][a-zA-Z']+\s+){2,}[A-Z][a-zA-Z']+[!?]?\s*$", re.MULTILINE),
]


def _is_section_header(line: str) -> bool:
    """Return True if this line looks like a section or product heading."""
    stripped = line.strip()
    if not stripped:
        return False

    # Never treat Q&A lines as section headers — they must stay with their answers
    # Matches: "Q. What is...", "A. The answer...", "## Q. What is..."
    if re.match(r"^#+\s*[QqAa]\.\s+", stripped):
        return False
    if re.match(r"^[QqAa]\.\s+", stripped):
        return False
    # Never treat bullet points starting with • as section headers
    if stripped.startswith("•") or stripped.startswith("-") or stripped.startswith("*"):
        return False

    for pattern in _SECTION_HEADER_PATTERNS:
        if pattern.match(stripped):
            return True
    return False


def _split_by_sections(text: str) -> list[str]:
    """
    Split text on section / product headers.
    Returns a list of raw section strings (may still be too long).
    """
    lines = text.splitlines(keepends=True)
    sections: list[str] = []
    current: list[str] = []

    for line in lines:
        if _is_section_header(line) and current:
            section_text = "".join(current).strip()
            if section_text:
                sections.append(section_text)
            current = [line]
        else:
            current.append(line)

    if current:
        section_text = "".join(current).strip()
        if section_text:
            sections.append(section_text)

    return sections


# ── Sub-split long sections ───────────────────────────────────────────────────

def _subsplit_section(text: str, max_chunk: int) -> list[str]:
    """
    If a section is longer than max_chunk, split it by paragraphs then
    sentences. Preserves the section's first line (heading) in every piece
    so context is never lost.
    """
    if len(text) <= max_chunk:
        return [text]

    # Extract heading (first line) to prepend to continuation chunks
    first_newline = text.find("\n")
    heading = text[:first_newline].strip() if first_newline != -1 else ""
    body    = text[first_newline:].strip() if first_newline != -1 else text

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", body) if p.strip()]
    chunks: list[str] = []
    buffer = heading  # always start with the heading

    for para in paragraphs:
        candidate = f"{buffer}\n\n{para}" if buffer else para
        if len(candidate) <= max_chunk:
            buffer = candidate
            continue
        # Flush buffer
        if buffer and buffer != heading:
            chunks.append(buffer.strip())
        # Start a new buffer with heading prepended for context
        candidate2 = f"{heading}\n\n{para}" if heading else para
        if len(candidate2) <= max_chunk:
            buffer = candidate2
        else:
            # Paragraph itself too long — split by sentence
            sentences = re.split(r"(?<=[.!?])\s+", para)
            seg = heading if heading else ""
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                trial = f"{seg}\n{sentence}" if seg else sentence
                if len(trial) <= max_chunk:
                    seg = trial
                else:
                    if seg and seg != heading:
                        chunks.append(seg.strip())
                    seg = f"{heading}\n{sentence}" if heading else sentence
                    # Hard-split single long sentence
                    if len(seg) > max_chunk:
                        for i in range(0, len(seg), max_chunk):
                            part = seg[i: i + max_chunk].strip()
                            if part:
                                chunks.append(part)
                        seg = heading if heading else ""
            buffer = seg

    if buffer and buffer.strip() and buffer.strip() != heading:
        chunks.append(buffer.strip())

    return chunks if chunks else [text[:max_chunk]]


# ── Public API ────────────────────────────────────────────────────────────────

def _split_text(text: str, min_chunk: int, max_chunk: int, overlap: int = 0) -> list[str]:
    """
    Main splitting pipeline:
      1. Split on section/product headers
      2. Sub-split oversized sections
      3. Filter too-small chunks
    Overlap is intentionally disabled (set to 0) — it was prepending tails
    that stripped headings from chunks and broke semantic search.
    """
    sections = _split_by_sections(text)
    chunks: list[str] = []

    for section in sections:
        sub = _subsplit_section(section, max_chunk)
        chunks.extend(sub)

    # Filter tiny chunks
    chunks = [c for c in chunks if len(c) >= min_chunk]
    return chunks


def chunk_text(text: str, source: str) -> list[dict[str, Any]]:
    """
    Split text into chunks and attach metadata.

    Returns:
        [{"text": str, "metadata": {"source": str, "chunk": int}}, ...]
    """
    raw_chunks = _split_text(text, CHUNK_MIN, CHUNK_MAX, overlap=CHUNK_OVERLAP)
    logger.debug("[CHUNK] %s → %d chunks", source, len(raw_chunks))
    return [
        {"text": chunk, "metadata": {"source": source, "chunk": i}}
        for i, chunk in enumerate(raw_chunks)
    ]
