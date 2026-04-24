"""Tests for obsidian.py note naming, frontmatter, and collision avoidance."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pipeline_youtube.obsidian import (
    _escape_yaml,
    build_frontmatter,
    format_playlist_folder_name,
    format_video_note_base,
    resolve_unique_path,
    sanitize_title_for_filename,
)


class TestSanitizeTitle:
    def test_simple(self):
        assert sanitize_title_for_filename("hello world") == "hello world"

    def test_unsafe_chars_replaced(self):
        assert sanitize_title_for_filename("foo/bar:baz") == "foo bar baz"

    def test_quotes_replaced(self):
        assert sanitize_title_for_filename('test "quoted" name') == "test quoted name"

    def test_all_unsafe_chars(self):
        raw = "a\\b/c:d*e?f<g>h|i"
        assert sanitize_title_for_filename(raw) == "a b c d e f g h i"

    def test_collapse_multiple_spaces(self):
        assert sanitize_title_for_filename("a   b\tc") == "a b c"

    def test_strip_edges(self):
        assert sanitize_title_for_filename("  hello  ") == "hello"

    def test_empty(self):
        assert sanitize_title_for_filename("") == ""

    def test_none(self):
        assert sanitize_title_for_filename(None) == ""

    def test_japanese_preserved(self):
        assert sanitize_title_for_filename("ハーネス設計") == "ハーネス設計"

    def test_mixed_jp_en(self):
        assert (
            sanitize_title_for_filename("Anthropicが公開したハーネス設計、全部解説します")
            == "Anthropicが公開したハーネス設計、全部解説します"
        )


class TestFormatVideoNoteBase:
    def test_with_title(self):
        dt = datetime(2026, 4, 14, 21, 41)
        assert format_video_note_base(dt, "Test Video") == "2026-04-14-2141 Test Video"

    def test_empty_title(self):
        dt = datetime(2026, 4, 14, 21, 41)
        assert format_video_note_base(dt, "") == "2026-04-14 2141"

    def test_none_title(self):
        dt = datetime(2026, 4, 14, 21, 41)
        assert format_video_note_base(dt, None) == "2026-04-14 2141"

    def test_matches_dummy_data(self):
        """Must match the existing dummy-data filename in 08_YouTube学習."""
        dt = datetime(2026, 4, 14, 21, 41)
        result = format_video_note_base(dt, "Anthropicが公開したハーネス設計、全部解説します")
        assert result == "2026-04-14-2141 Anthropicが公開したハーネス設計、全部解説します"

    def test_unsafe_chars_in_title(self):
        dt = datetime(2026, 4, 14, 21, 41)
        assert format_video_note_base(dt, "Slash/and:colon") == "2026-04-14-2141 Slash and colon"

    def test_zero_padded_time(self):
        dt = datetime(2026, 1, 2, 3, 5)
        assert format_video_note_base(dt, "Test") == "2026-01-02-0305 Test"


class TestFormatPlaylistFolder:
    def test_with_title(self):
        dt = datetime(2026, 4, 14, 13, 45)
        assert (
            format_playlist_folder_name(dt, "Harness Engineering")
            == "2026-04-14-1345 Harness Engineering"
        )

    def test_midnight_pads_zeros(self):
        dt = datetime(2026, 4, 14, 0, 0)
        assert (
            format_playlist_folder_name(dt, "Harness Engineering")
            == "2026-04-14-0000 Harness Engineering"
        )

    def test_empty_title(self):
        dt = datetime(2026, 4, 14, 9, 5)
        assert format_playlist_folder_name(dt, "") == "2026-04-14-0905"

    def test_none_title(self):
        dt = datetime(2026, 4, 14, 9, 5)
        assert format_playlist_folder_name(dt, None) == "2026-04-14-0905"

    def test_strips_ascii_slash_category_prefix(self):
        """`2026Agent Teams/AI駆動経営` -> drop category, keep `AI駆動経営` only."""
        dt = datetime(2026, 4, 16, 9, 14)
        assert (
            format_playlist_folder_name(dt, "2026Agent Teams/AI駆動経営")
            == "2026-04-16-0914 AI駆動経営"
        )

    def test_strips_multiple_slashes(self):
        dt = datetime(2026, 4, 16, 9, 14)
        assert format_playlist_folder_name(dt, "A/B/C Title") == "2026-04-16-0914 C Title"

    def test_fullwidth_slash_is_kept(self):
        """Full-width `／` is legitimate Japanese punctuation, not a separator."""
        dt = datetime(2026, 4, 16, 9, 14)
        assert (
            format_playlist_folder_name(dt, "Agent Teams／3 人編成")
            == "2026-04-16-0914 Agent Teams／3 人編成"
        )


class TestResolveUniquePath:
    def test_fresh_folder(self, tmp_path: Path):
        assert resolve_unique_path(tmp_path, "note", ".md") == tmp_path / "note.md"

    def test_first_collision(self, tmp_path: Path):
        (tmp_path / "note.md").write_text("x")
        assert resolve_unique_path(tmp_path, "note", ".md") == tmp_path / "note-2.md"

    def test_multiple_collisions(self, tmp_path: Path):
        (tmp_path / "note.md").write_text("x")
        (tmp_path / "note-2.md").write_text("x")
        (tmp_path / "note-3.md").write_text("x")
        assert resolve_unique_path(tmp_path, "note", ".md") == tmp_path / "note-4.md"

    def test_nonexistent_folder(self, tmp_path: Path):
        folder = tmp_path / "does_not_exist"
        # Should still return the first candidate; caller is responsible for mkdir
        assert resolve_unique_path(folder, "note", ".md") == folder / "note.md"

    def test_custom_extension(self, tmp_path: Path):
        (tmp_path / "image.png").write_text("x")
        assert resolve_unique_path(tmp_path, "image", ".png") == tmp_path / "image-2.png"


class TestEscapeYaml:
    def test_plain(self):
        assert _escape_yaml("hello") == "hello"

    def test_quotes(self):
        assert _escape_yaml('a"b') == 'a\\"b'

    def test_backslash(self):
        assert _escape_yaml("a\\b") == "a\\\\b"

    def test_newline_to_space(self):
        assert _escape_yaml("a\nb") == "a b"

    def test_cr_removed(self):
        assert _escape_yaml("a\rb") == "ab"

    def test_yaml_separator_neutralized(self):
        assert _escape_yaml("foo---bar") == "foo\\-\\-\\-bar"

    def test_empty(self):
        assert _escape_yaml("") == ""

    def test_none(self):
        assert _escape_yaml(None) == ""


class TestBuildFrontmatter:
    def test_basic(self):
        dt = datetime(2026, 4, 14, 21, 41)
        fm = build_frontmatter(dt, "Test", url="https://example.com")
        assert fm.startswith("---\n")
        assert "date: 2026-04-14 21:41\n" in fm
        assert 'title: "Test"\n' in fm
        assert 'URL: "https://example.com"\n' in fm
        assert "tags: [memo, youtube]\n" in fm
        assert fm.endswith("---\n")

    def test_yaml_escaping_in_title(self):
        dt = datetime(2026, 1, 1, 0, 0)
        fm = build_frontmatter(dt, 'Title with "quotes"', url="")
        assert 'title: "Title with \\"quotes\\""' in fm

    def test_extra_fields(self):
        dt = datetime(2026, 4, 14, 21, 41)
        fm = build_frontmatter(
            dt,
            "Test",
            url="",
            extra={"playlist": "Harness Engineering", "video_id": "abc123"},
        )
        assert 'playlist: "Harness Engineering"' in fm
        assert 'video_id: "abc123"' in fm

    def test_custom_tags(self):
        dt = datetime(2026, 4, 14, 21, 41)
        fm = build_frontmatter(dt, "Test", tags=["custom", "another"])
        assert "tags: [custom, another]" in fm

    def test_empty_title(self):
        dt = datetime(2026, 4, 14, 21, 41)
        fm = build_frontmatter(dt, None)
        assert 'title: ""' in fm
