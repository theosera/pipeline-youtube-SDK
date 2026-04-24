"""Tests for the parse_capture_mapping helper in stage 04.

Locks the behavior that learning.py extracts [range, filename] pairs
explicitly (not via claude inference) so prompt can include an
allow-list that prevents hallucinated filenames.
"""

from __future__ import annotations

from pipeline_youtube.stages.learning import (
    CaptureMapping,
    _format_mapping_table,
    parse_capture_mapping,
)

SAMPLE_CAPTURE_BODY = """[00:00 ~ 01:03]
![[2026-04-15-2123 Anthropicが公開したハーネス設計、全部解説します.webp]]

[01:45 ~ 03:25]
![[2026-04-15-2123 Anthropicが公開したハーネス設計、全部解説します-1.webp]]

[03:26 ~ 05:05]
![[2026-04-15-2123 Anthropicが公開したハーネス設計、全部解説します-2.webp]]
"""


class TestParseCaptureMapping:
    def test_parses_standard_format(self):
        mappings = parse_capture_mapping(SAMPLE_CAPTURE_BODY)
        assert len(mappings) == 3

        assert mappings[0].range_str == "[00:00 ~ 01:03]"
        assert mappings[0].filename.endswith(".webp")
        assert "-" not in mappings[0].filename.rsplit(".", 1)[0][-3:]  # base, no suffix

        assert mappings[1].range_str == "[01:45 ~ 03:25]"
        assert "-1.webp" in mappings[1].filename

        assert mappings[2].range_str == "[03:26 ~ 05:05]"
        assert "-2.webp" in mappings[2].filename

    def test_empty_body(self):
        assert parse_capture_mapping("") == []

    def test_capture_with_failure_comments(self):
        """When captures failed, there are no ![[...]] embeds — return empty."""
        body = """[00:00 ~ 01:03]
<!-- capture failed: ffmpeg_error -->

[01:45 ~ 03:25]
<!-- capture failed: timeout -->
"""
        assert parse_capture_mapping(body) == []

    def test_mixed_success_and_failure(self):
        body = """[00:00 ~ 01:03]
![[first.webp]]

[01:45 ~ 03:25]
<!-- capture failed -->

[03:26 ~ 05:05]
![[third.webp]]
"""
        mappings = parse_capture_mapping(body)
        assert len(mappings) == 2
        assert mappings[0].filename == "first.webp"
        assert mappings[1].filename == "third.webp"

    def test_tolerates_fullwidth_tilde(self):
        body = "[00:10 〜 00:20]\n![[file.webp]]\n"
        mappings = parse_capture_mapping(body)
        assert len(mappings) == 1
        assert mappings[0].range_str == "[00:10 ~ 00:20]"

    def test_japanese_filenames(self):
        body = "[00:00 ~ 00:30]\n![[2026-04-15-2123 日本語タイトル.webp]]\n"
        mappings = parse_capture_mapping(body)
        assert len(mappings) == 1
        assert "日本語タイトル" in mappings[0].filename


class TestFormatMappingTable:
    def test_formats_as_markdown_table(self):
        mappings = [
            CaptureMapping("[00:00 ~ 01:03]", "base.webp"),
            CaptureMapping("[01:45 ~ 03:25]", "base-1.webp"),
        ]
        table = _format_mapping_table(mappings)
        assert "| タイムスタンプ範囲 | 画像ファイル名 |" in table
        assert "|---|---|" in table
        assert "| [00:00 ~ 01:03] | base.webp |" in table
        assert "| [01:45 ~ 03:25] | base-1.webp |" in table

    def test_empty_mappings(self):
        assert _format_mapping_table([]) == "(画像無し)"
