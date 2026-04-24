"""Tests for obsidian.read_frontmatter_field / upsert_frontmatter_field."""

from __future__ import annotations

from pathlib import Path

from pipeline_youtube.obsidian import read_frontmatter_field, upsert_frontmatter_field


class TestReadFrontmatterField:
    def test_quoted_value(self, tmp_path: Path):
        p = tmp_path / "note.md"
        p.write_text('---\ntitle: "hello"\ndate: 2026-04-18\n---\nbody\n', encoding="utf-8")
        assert read_frontmatter_field(p, "title") == "hello"

    def test_bare_value(self, tmp_path: Path):
        p = tmp_path / "note.md"
        p.write_text("---\nreviewed: true\n---\nbody\n", encoding="utf-8")
        assert read_frontmatter_field(p, "reviewed") == "true"

    def test_missing_field(self, tmp_path: Path):
        p = tmp_path / "note.md"
        p.write_text('---\ntitle: "x"\n---\n', encoding="utf-8")
        assert read_frontmatter_field(p, "one_liner") is None

    def test_no_frontmatter(self, tmp_path: Path):
        p = tmp_path / "note.md"
        p.write_text("plain text\n", encoding="utf-8")
        assert read_frontmatter_field(p, "title") is None

    def test_missing_file(self, tmp_path: Path):
        assert read_frontmatter_field(tmp_path / "nope.md", "title") is None

    def test_field_outside_head_500b_ignored(self, tmp_path: Path):
        p = tmp_path / "note.md"
        filler = "x: y\n" * 200
        p.write_text("---\n" + filler + "needle: found\n---\n", encoding="utf-8")
        assert read_frontmatter_field(p, "needle") is None


class TestUpsertFrontmatterField:
    def test_inserts_new_field(self):
        md = '---\ntitle: "x"\n---\nbody\n'
        out = upsert_frontmatter_field(md, "one_liner", "新サマリ")
        assert 'one_liner: "新サマリ"' in out
        assert "body" in out

    def test_updates_existing_field(self):
        md = '---\ntitle: "old"\n---\nbody\n'
        out = upsert_frontmatter_field(md, "title", "new")
        assert 'title: "new"' in out
        assert "old" not in out

    def test_no_frontmatter_returns_unchanged(self):
        md = "plain\n"
        assert upsert_frontmatter_field(md, "k", "v") == md

    def test_escapes_quotes_in_value(self):
        md = '---\ntitle: "x"\n---\n'
        out = upsert_frontmatter_field(md, "title", 'contains "quotes"')
        assert 'title: "contains \\"quotes\\""' in out

    def test_preserves_body(self):
        md = '---\ntitle: "x"\n---\n\n# body\n- item\n'
        out = upsert_frontmatter_field(md, "k", "v")
        assert "# body" in out
        assert "- item" in out
