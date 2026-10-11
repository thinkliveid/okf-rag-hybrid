from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from .models import DocumentAnalysis, OKFDocument


def generate(analyses: list[DocumentAnalysis]) -> list[OKFDocument]:
    return [generate_document(analysis) for analysis in analyses]


def generate_document(analysis: DocumentAnalysis) -> OKFDocument:
    slug = _slugify(analysis.title or analysis.parsed.file.path.stem)
    category = _category(analysis)
    doc_id = hashlib.sha256(f"{analysis.parsed.file.path}#{analysis.title}".encode("utf-8")).hexdigest()[:16]
    timestamp = datetime.now(timezone.utc).isoformat()

    frontmatter = {
        "id": doc_id,
        "type": analysis.document_type,
        "title": analysis.title,
        "description": analysis.description,
        "tags": analysis.keywords,
        "entities": analysis.entities,
        "source": str(analysis.parsed.file.path),
        "timestamp": timestamp,
    }
    if analysis.referenced_documents:
        frontmatter["references"] = analysis.referenced_documents

    body = _body(analysis)
    return OKFDocument(
        id=doc_id,
        relative_path=Path(category) / f"{slug}.md",
        frontmatter=frontmatter,
        body=body,
        source_path=analysis.parsed.file.path,
    )


def _body(analysis: DocumentAnalysis) -> str:
    lines = [f"# {analysis.title}", "", analysis.description.strip(), ""]
    for section in analysis.sections:
        title = section["title"].strip()
        content = section["content"].strip()
        if not content:
            continue
        if title.lower() != "overview" and title.lower() != analysis.title.lower():
            lines.extend([f"## {title}", ""])
        lines.extend([content, ""])
    return "\n".join(lines).strip()


def _category(analysis: DocumentAnalysis) -> str:
    # Use parent directory name if present and not root
    parent = analysis.parsed.file.path.parent.name
    if parent and parent.lower() not in {"data", "raw", "documents", "."}:
        return _slugify(parent)
    return _slugify(analysis.document_type)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "document"
