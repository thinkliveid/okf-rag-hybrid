from __future__ import annotations

from pathlib import Path

from .models import OKFDocument


def write(documents: list[OKFDocument], output_dir: Path | str) -> list[Path]:
    """Persist OKF markdown documents and return written paths."""
    root = Path(output_dir)
    written: list[Path] = []
    for document in documents:
        path = root / document.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document.render(), encoding="utf-8", newline="\n")
        written.append(path)
    return written
