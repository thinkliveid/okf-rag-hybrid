from __future__ import annotations

import re
from collections import Counter

from .models import DocumentAnalysis, ParsedDocument

_TYPE_KEYWORDS = {
    "API Documentation": ["api", "endpoint", "request", "response", "http", "json"],
    "Runbook": ["runbook", "incident", "alert", "restart", "rollback", "on-call"],
    "SQL": ["select ", "insert ", "update ", "delete ", "create table", "join "],
    "Policy": ["policy", "shall", "must", "compliance", "terms", "condition"],
    "Playbook": ["playbook", "procedure", "steps", "checklist", "workflow"],
    "Meeting Notes": ["meeting", "agenda", "minutes", "attendees", "action items"],
    "FAQ": ["faq", "frequently asked", "question", "answer", "q:", "a:"],
    "Dataset": ["csv", "dataset", "rows", "columns", "schema"],
    "Configuration": ["config", "configuration", "yaml", "env", "setting"],
}

_STOPWORDS = {
    "about", "after", "again", "because", "before", "being", "between",
    "could", "every", "from", "have", "into", "more", "only", "other", "should",
    "their", "there", "these", "they", "this", "through", "under", "with", "would",
}


def analyze(documents: list[ParsedDocument]) -> list[DocumentAnalysis]:
    return [analyze_document(document) for document in documents]


def analyze_document(document: ParsedDocument) -> DocumentAnalysis:
    text = document.text
    title = _extract_title(document)
    sections = _extract_sections(text)
    keywords = _keywords(text)
    document_type = _document_type(document, text)
    entities = _entities(text)
    refs = _references(text)
    description = _description(text, title)

    return DocumentAnalysis(
        parsed=document,
        document_type=document_type,
        title=title,
        description=description,
        keywords=keywords,
        entities=entities,
        referenced_documents=refs,
        sections=sections,
    )


def _extract_title(document: ParsedDocument) -> str:
    if "segment_title" in document.metadata:
        return document.metadata["segment_title"]
    for line in document.text.splitlines():
        stripped = line.strip(" #\t")
        if stripped:
            return stripped[:120]
    return document.file.path.stem.replace("_", " ").replace("-", " ").title()


def _document_type(document: ParsedDocument, text: str) -> str:
    haystack = f"{document.file.path.name}\n{text[:5000]}".lower()
    scores = {
        doc_type: sum(1 for keyword in keywords if keyword in haystack)
        for doc_type, keywords in _TYPE_KEYWORDS.items()
    }
    best_type, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score > 0:
        return best_type
    if document.file.extension in {"csv", "json"}:
        return "Dataset"
    return "Knowledge Article"


def _extract_sections(text: str) -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    current_title = "Overview"
    current_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if _looks_like_heading(stripped) and current_lines:
            sections.append({"title": current_title, "content": "\n".join(current_lines).strip()})
            current_title = stripped.strip("# ")
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append({"title": current_title, "content": "\n".join(current_lines).strip()})

    return [section for section in sections if section["content"]]


def _looks_like_heading(line: str) -> bool:
    if not line or len(line) > 100:
        return False
    # Never treat Q&A lines as headings — they must stay with their answers
    if re.match(r"^[QqAa]\.\s+", line.strip()):
        return False
    # Never treat bullet points as headings
    stripped = line.strip()
    if stripped.startswith("\u2022") or stripped.startswith("-") or stripped.startswith("*"):
        return False
    return (
        line.startswith("#")
        or bool(re.match(r"^\d+[\.)]\s+[A-Z]", line))
        or bool(re.match(r"^(?![QqAa]\.)([A-Z][A-Z0-9 &/\-]{4,})$", line))
        or bool(re.match(r"^[A-Z][\w &/\-]{3,}:$", line))
    )


def _keywords(text: str, limit: int = 12) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", text.lower())
    counts = Counter(word for word in words if word not in _STOPWORDS)
    return [word for word, _ in counts.most_common(limit)]


def _entities(text: str, limit: int = 20) -> list[str]:
    candidates = re.findall(r"\b(?:[A-Z][A-Za-z0-9&.-]+(?:\s+|$)){1,5}", text)
    cleaned = [" ".join(candidate.split()) for candidate in candidates]
    counts = Counter(item for item in cleaned if len(item) > 2)
    return [item for item, _ in counts.most_common(limit)]


def _references(text: str) -> list[str]:
    refs = set(re.findall(r"\[[^\]]+\]\(([^)]+)\)", text))
    refs.update(re.findall(r"\b[\w./-]+\.(?:md|txt|pdf|docx|csv|json)\b", text, re.I))
    return sorted(refs)


def _description(text: str, title: str) -> str:
    compact = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if compact.lower().startswith(title.lower()):
        compact = compact[len(title):].strip(" :-")
    return compact[:240] or title
