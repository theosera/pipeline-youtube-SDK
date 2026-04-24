"""Tests for H2: multi-layer defense in chapter filename construction."""

from __future__ import annotations

from pipeline_youtube.synthesis.chapter import chapter_filename


class TestChapterFilenameHardening:
    def test_rlo_override_stripped(self):
        name = chapter_filename(1, "normal\u202ereversed.md")
        assert "\u202e" not in name
        assert name.startswith("01_")
        assert name.endswith(".md")

    def test_zero_width_stripped(self):
        raw = "hel\u200blo\u200cwor\u200dld"
        name = chapter_filename(2, raw)
        for zw in "\u200b\u200c\u200d":
            assert zw not in name
        assert name.startswith("02_")
        assert name.endswith(".md")

    def test_long_japanese_utf8_boundary_safe(self):
        label = "日本語" * 80
        name = chapter_filename(3, label)
        assert len(name.encode("utf-8")) <= 200
        assert name.endswith(".md")
        name.encode("utf-8").decode("utf-8")

    def test_control_chars_stripped(self):
        name = chapter_filename(4, "hello\x00\x01\x1fworld")
        assert "\x00" not in name
        assert "\x01" not in name
        assert "\x1f" not in name

    def test_os_unsafe_chars_stripped(self):
        name = chapter_filename(5, 'a/b:c*d?e"f<g>h|i\\j')
        for ch in '\\/:*?"<>|':
            assert ch not in name.split("_", 1)[1][:-3]
        assert name.endswith(".md")

    def test_empty_label_uses_fallback(self):
        assert chapter_filename(7, "").startswith("07_chapter-7")
        assert chapter_filename(7, "").endswith(".md")

    def test_index_zero_padded(self):
        assert chapter_filename(1, "foo").startswith("01_")
        assert chapter_filename(10, "foo").startswith("10_")

    def test_md_suffix_always_present(self):
        for label in ["", "x", "日本語" * 100, "a\u202eb\u200c"]:
            assert chapter_filename(1, label).endswith(".md")
