"""Tests for obsidian.read_frontmatter_field / upsert_frontmatter_field."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pipeline_youtube.obsidian import read_frontmatter_field, upsert_frontmatter_field
from pipeline_youtube.services import obsidian as obsidian_mod
from pipeline_youtube.services.obsidian import build_frontmatter


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

    def test_field_beyond_read_limit_ignored(self, tmp_path: Path):
        """Pathological frontmatter past the read ceiling still returns None.

        The ceiling exists to bound batch scans; it must stay large enough for
        every frontmatter this pipeline writes (see
        ``test_reviewed_survives_cjk_title_playlist_and_one_liner``).
        """
        p = tmp_path / "note.md"
        # ~5 bytes per line → well past _FRONTMATTER_READ_LIMIT.
        filler_lines = max(obsidian_mod._FRONTMATTER_READ_LIMIT // 5 + 50, 2000)
        filler = "x: y\n" * filler_lines
        p.write_text("---\n" + filler + "needle: found\n---\n", encoding="utf-8")
        assert read_frontmatter_field(p, "needle") is None

    def test_reviewed_survives_cjk_title_playlist_and_one_liner(self, tmp_path: Path):
        """Phase 3 must still see ``reviewed`` after Stage 02's one_liner upsert.

        Concrete trigger under the old 500-byte cap: a 60-char CJK title, a
        short playlist, and a 40-char one_liner push the closing ``---`` past
        byte 500. ``read_frontmatter_field`` then returned None for every
        field — including ``reviewed`` — so ``--resume-reviewed`` silently
        skipped notes the operator had approved.
        """
        fm = build_frontmatter(
            dt=datetime(2026, 4, 18, 8, 0),
            title="あ" * 60,
            url="https://www.youtube.com/watch?v=_h3decBW12Q",
            tags=["memo", "youtube"],
            extra={
                "playlist": "い" * 10,
                "video_id": "_h3decBW12Q",
                "reviewed": "true",
            },
        )
        fm = upsert_frontmatter_field(fm, "one_liner", "う" * 40)
        assert len(fm.encode("utf-8")) > 500

        p = tmp_path / "02_Summary.md"
        p.write_text(fm + "\nbody\n", encoding="utf-8")

        assert read_frontmatter_field(p, "reviewed") == "true"
        assert read_frontmatter_field(p, "video_id") == "_h3decBW12Q"
        assert read_frontmatter_field(p, "one_liner") == "う" * 40

    def test_reviewed_survives_max_youtube_cjk_title(self, tmp_path: Path):
        """YouTube's 100-char title ceiling + a moderate playlist also fit."""
        fm = build_frontmatter(
            dt=datetime(2026, 4, 18, 8, 0),
            title="あ" * 100,
            url="https://www.youtube.com/watch?v=_h3decBW12Q",
            tags=["memo", "youtube"],
            extra={
                "playlist": "い" * 40,
                "video_id": "_h3decBW12Q",
                "reviewed": "true",
            },
        )
        fm = upsert_frontmatter_field(fm, "one_liner", "う" * 60)
        assert len(fm.encode("utf-8")) > 500

        p = tmp_path / "02_Summary.md"
        p.write_text(fm + "\nbody\n", encoding="utf-8")

        assert read_frontmatter_field(p, "reviewed") == "true"


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
