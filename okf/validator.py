from __future__ import annotations

import re
from datetime import datetime

from .models import OKFDocument, ValidationResult

REQUIRED_FIELDS = {"id", "type", "title", "description", "tags", "timestamp"}


def validate(documents: list[OKFDocument]) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    known_paths = {str(doc.relative_path).replace("\\", "/") for doc in documents}

    for doc in documents:
        missing = REQUIRED_FIELDS - set(doc.frontmatter)
        if missing:
            errors.append(f"{doc.relative_path}: missing required fields {sorted(missing)}")

        doc_id = str(doc.frontmatter.get("id", ""))
        if not doc_id:
            errors.append(f"{doc.relative_path}: missing id")
        elif doc_id in seen_ids:
            errors.append(f"{doc.relative_path}: duplicate id {doc_id}")
        seen_ids.add(doc_id)

        if not str(doc.frontmatter.get("title", "")).strip():
            errors.append(f"{doc.relative_path}: missing title")

        timestamp = str(doc.frontmatter.get("timestamp", ""))
        try:
            datetime.fromisoformat(timestamp)
        except ValueError:
            errors.append(f"{doc.relative_path}: invalid timestamp {timestamp!r}")

        if not doc.body.strip():
            errors.append(f"{doc.relative_path}: empty body")

        if "\ufffd" in doc.body:
            warnings.append(f"{doc.relative_path}: replacement characters found")

        for link in re.findall(r"\[[^\]]+\]\(([^)]+)\)", doc.body):
            if link.startswith(("http://", "https://", "#", "mailto:")):
                continue
            normalized = link.replace("\\", "/").lstrip("./")
            if normalized not in known_paths:
                warnings.append(f"{doc.relative_path}: possible broken markdown link {link}")

        for ref in doc.frontmatter.get("references", []):
            normalized = str(ref).replace("\\", "/").lstrip("./")
            if normalized.endswith(".md") and normalized not in known_paths:
                warnings.append(f"{doc.relative_path}: possible broken reference {ref}")

        if re.search(r"^##\s+\S+\s*$\n\s*(?=##|\Z)", doc.body, re.M):
            warnings.append(f"{doc.relative_path}: empty section detected")

    return ValidationResult(valid=not errors, errors=errors, warnings=warnings)
