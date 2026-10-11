from pathlib import Path

import pytest

import knowledge


def write_concept(root: Path, rel: str, **fields) -> Path:
    body = fields.pop("body", "Concept body.")
    lines = ["---"] + [f"{key}: {value}" for key, value in fields.items()] + ["---", "", body]
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def valid(**overrides):
    fields = {"type": "fact", "title": "Database", "description": "Which database is used."}
    fields.update(overrides)
    return fields


def test_load_concepts_parses_frontmatter_and_uses_posix_relative_path(tmp_path):
    write_concept(tmp_path, "app/database.md", **valid(verified="{tier: human, by: 'human:owner'}"))

    [concept] = knowledge.load_concepts(tmp_path)

    assert concept.path == "app/database.md"
    assert concept.title == "Database"
    assert concept.verified_tier == "human"
    assert concept.status == "active"
    assert concept.body == "Concept body."


def test_reserved_files_are_not_concepts(tmp_path):
    write_concept(tmp_path, "index.md", **valid())
    write_concept(tmp_path, "app/log.md", **valid())
    write_concept(tmp_path, "app/real.md", **valid())

    assert [c.path for c in knowledge.load_concepts(tmp_path)] == ["app/real.md"]


@pytest.mark.parametrize(
    "fields",
    [
        {"title": "No type", "description": "x"},
        {"type": "fact", "description": "x"},
        {"type": "fact", "title": "No description"},
        {"type": "fact", "title": "t", "description": "d", "verified": "bogus"},
    ],
)
def test_invalid_concepts_are_skipped_and_reported(tmp_path, fields):
    write_concept(tmp_path, "bad.md", **fields)
    write_concept(tmp_path, "good.md", **valid())

    assert [c.path for c in knowledge.load_concepts(tmp_path)] == ["good.md"]
    assert list(knowledge.check(tmp_path)) == ["bad.md"]


def test_file_without_frontmatter_is_invalid(tmp_path):
    (tmp_path / "plain.md").write_text("# Just markdown\n", encoding="utf-8")

    assert knowledge.load_concepts(tmp_path) == []
    assert "plain.md" in knowledge.check(tmp_path)


def test_non_active_concepts_are_excluded_unless_requested(tmp_path):
    write_concept(tmp_path, "old.md", **valid(status="deprecated"))
    write_concept(tmp_path, "new.md", **valid())

    assert [c.path for c in knowledge.load_concepts(tmp_path)] == ["new.md"]
    assert len(knowledge.load_concepts(tmp_path, include_inactive=True)) == 2


def test_missing_directory_yields_no_concepts(tmp_path):
    assert knowledge.load_concepts(tmp_path / "absent") == []


def test_render_listing_is_sorted_and_single_line_per_concept(tmp_path):
    write_concept(tmp_path, "b.md", **valid(title="B", description="second\n  line"))
    write_concept(tmp_path, "a.md", **valid(title="A"))

    listing = knowledge.render_listing(knowledge.load_concepts(tmp_path))

    assert listing.splitlines() == [
        "- a.md | A | Which database is used.",
        "- b.md | B | second line",
    ]


def test_write_indexes_is_deterministic_and_links_subfolders(tmp_path):
    write_concept(tmp_path, "top.md", **valid(title="Top"))
    write_concept(tmp_path, "app/db.md", **valid(title="DB"))
    write_concept(tmp_path, "app/deep/rule.md", **valid(title="Rule"))

    first = {p: p.read_text() for p in knowledge.write_indexes(tmp_path)}
    second = {p: p.read_text() for p in knowledge.write_indexes(tmp_path)}

    assert first == second
    root_index = (tmp_path / "index.md").read_text()
    assert "- [app/](app/index.md)" in root_index
    assert "- [Top](top.md) — Which database is used." in root_index
    app_index = (tmp_path / "app" / "index.md").read_text()
    assert app_index.startswith("# app\n")
    assert "- [deep/](deep/index.md)" in app_index
    assert "- [DB](db.md)" in app_index
    assert (tmp_path / "app" / "deep" / "index.md").exists()
    assert [c.path for c in knowledge.load_concepts(tmp_path)] == ["app/db.md", "app/deep/rule.md", "top.md"]


def test_cli_check_exit_code(tmp_path):
    write_concept(tmp_path, "good.md", **valid())
    assert knowledge.main(["check", "--dir", str(tmp_path)]) == 0
    write_concept(tmp_path, "bad.md", type="fact")
    assert knowledge.main(["check", "--dir", str(tmp_path)]) == 1
