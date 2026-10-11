from __future__ import annotations

import mimetypes
from datetime import datetime, timezone
from pathlib import Path

from .models import FileInfo


def scan(data_dir: Path | str) -> list[FileInfo]:
    """Discover files and return filesystem metadata only."""
    root = Path(data_dir)
    if not root.exists():
        return []

    files: list[FileInfo] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        stat = path.stat()
        mime_type, _ = mimetypes.guess_type(path.name)
        files.append(
            FileInfo(
                path=path,
                extension=path.suffix.lower().lstrip("."),
                size=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                mime_type=mime_type or "application/octet-stream",
            )
        )
    return files
