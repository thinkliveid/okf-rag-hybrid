from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

try:
    from .config import KNOWLEDGE_DIR
except ImportError:
    from config import KNOWLEDGE_DIR

logger = logging.getLogger(__name__)

RESERVED_FILENAMES = {"index.md", "log.md"}
REQUIRED_FIELDS = ("type", "title", "description")
ACTIVE_STATUS = "active"
VERIFIED_TIERS = ("unverified", "machine", "human")
ROOT_INDEX_TITLE = "Knowledge index"


class ConceptError(ValueError):
    pass


@dataclass(frozen=True)
class Concept:
    path: str
    type: str
    title: str
    description: str
    body: str
    status: str = ACTIVE_STATUS
    verified_tier: str = "unverified"
    frontmatter: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def is_active(self) -> bool:
        return self.status == ACTIVE_STATUS


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ConceptError("missing frontmatter block")
    for end, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            break
    else:
        raise ConceptError("unterminated frontmatter block")
    try:
        frontmatter = yaml.safe_load("\n".join(lines[1:end])) or {}
    except yaml.YAMLError as exc:
        raise ConceptError(f"invalid YAML frontmatter: {exc}") from exc
    if not isinstance(frontmatter, dict):
        raise ConceptError("frontmatter must be a mapping")
    return frontmatter, "\n".join(lines[end + 1 :]).strip()


def _verified_tier(frontmatter: dict[str, Any]) -> str:
    verified = frontmatter.get("verified")
    tier = verified.get("tier") if isinstance(verified, dict) else verified
    tier = str(tier or "unverified").strip().lower()
    if tier not in VERIFIED_TIERS:
        raise ConceptError(f"verified tier must be one of {', '.join(VERIFIED_TIERS)}, got {tier!r}")
    return tier


def parse_concept(text: str, path: str) -> Concept:
    frontmatter, body = split_frontmatter(text)
    missing = [name for name in REQUIRED_FIELDS if not str(frontmatter.get(name) or "").strip()]
    if missing:
        raise ConceptError(f"missing required fields: {', '.join(missing)}")
    if not body:
        raise ConceptError("empty body")
    return Concept(
        path=path,
        type=str(frontmatter["type"]).strip(),
        title=str(frontmatter["title"]).strip(),
        description=" ".join(str(frontmatter["description"]).split()),
        body=body,
        status=str(frontmatter.get("status") or ACTIVE_STATUS).strip().lower(),
        verified_tier=_verified_tier(frontmatter),
        frontmatter=frontmatter,
    )


def concept_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (p for p in root.rglob("*.md") if p.is_file() and p.name not in RESERVED_FILENAMES),
        key=lambda p: p.relative_to(root).as_posix(),
    )


def _read_concepts(root: Path) -> tuple[list[Concept], dict[str, str]]:
    concepts: list[Concept] = []
    errors: dict[str, str] = {}
    for file in concept_files(root):
        rel = file.relative_to(root).as_posix()
        try:
            concepts.append(parse_concept(file.read_text(encoding="utf-8"), rel))
        except (ConceptError, UnicodeDecodeError) as exc:
            errors[rel] = str(exc)
    return concepts, errors


def load_concepts(root: Path = KNOWLEDGE_DIR, include_inactive: bool = False) -> list[Concept]:
    concepts, errors = _read_concepts(root)
    for rel, error in errors.items():
        logger.warning("[KNOWLEDGE] Skipping %s: %s", rel, error)
    return concepts if include_inactive else [c for c in concepts if c.is_active]


def check(root: Path = KNOWLEDGE_DIR) -> dict[str, str]:
    return _read_concepts(root)[1]


def render_listing(concepts: list[Concept]) -> str:
    return "\n".join(
        f"- {c.path} | {c.title} | {c.description}" for c in sorted(concepts, key=lambda c: c.path)
    )


def _index_dirs(concepts: list[Concept]) -> set[PurePosixPath]:
    dirs = {PurePosixPath(".")}
    for concept in concepts:
        dirs.update(p for p in PurePosixPath(concept.path).parents)
    return dirs


def render_index(directory: PurePosixPath, concepts: list[Concept]) -> str:
    title = ROOT_INDEX_TITLE if directory == PurePosixPath(".") else directory.name
    subdirs = sorted(
        d for d in _index_dirs(concepts) if d != directory and d.parent == directory
    )
    own = sorted(
        (c for c in concepts if PurePosixPath(c.path).parent == directory), key=lambda c: c.path
    )
    lines = [f"# {title}", ""]
    lines += [f"- [{d.name}/]({d.name}/index.md)" for d in subdirs]
    lines += [f"- [{c.title}]({PurePosixPath(c.path).name}) — {c.description}" for c in own]
    return "\n".join(lines).rstrip() + "\n"


def write_indexes(root: Path = KNOWLEDGE_DIR) -> list[Path]:
    concepts = load_concepts(root)
    written: list[Path] = []
    for directory in sorted(_index_dirs(concepts)):
        target = root / directory / "index.md"
        target.write_text(render_index(directory, concepts), encoding="utf-8", newline="\n")
        written.append(target)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Maintain the curated OKF knowledge bundle")
    parser.add_argument("command", choices=["index", "check"])
    parser.add_argument("--dir", type=Path, default=KNOWLEDGE_DIR)
    args = parser.parse_args(argv)

    if args.command == "check":
        errors = check(args.dir)
        for rel, error in errors.items():
            print(f"INVALID {rel}: {error}")
        concepts = load_concepts(args.dir, include_inactive=True)
        print(f"{len(concepts)} valid concepts, {len(errors)} invalid in {args.dir}")
        return 1 if errors else 0

    for path in write_indexes(args.dir):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
