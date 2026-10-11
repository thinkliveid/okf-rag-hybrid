from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FileInfo:
    path: Path
    extension: str
    size: int
    modified_at: datetime
    mime_type: str


@dataclass
class ParsedDocument:
    file: FileInfo
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentAnalysis:
    parsed: ParsedDocument
    document_type: str
    title: str
    description: str
    keywords: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    referenced_documents: list[str] = field(default_factory=list)
    sections: list[dict[str, str]] = field(default_factory=list)


@dataclass
class OKFDocument:
    id: str
    relative_path: Path
    frontmatter: dict[str, Any]
    body: str
    source_path: Path

    def render(self) -> str:
        return _render_frontmatter(self.frontmatter) + "\n\n" + self.body.strip() + "\n"


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _render_frontmatter(data: dict[str, Any]) -> str:
    lines = ["---"]
    for key, value in data.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"- {_yaml_scalar(item)}")
        elif isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                lines.append(f"  {sub_key}: {_yaml_scalar(sub_value)}")
        else:
            lines.append(f"{key}: {_yaml_scalar(value)}")
    lines.append("---")
    return "\n".join(lines)


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    if not text or any(ch in text for ch in [":", "#", "\n", "[", "]", "{", "}"]):
        return f'"{text}"'
    return text
