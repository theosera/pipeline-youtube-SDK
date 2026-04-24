"""Tests for sanitize.py (prompt injection mitigation)."""

from __future__ import annotations

from pipeline_youtube.sanitize import (
    UNTRUSTED_CLOSE,
    UNTRUSTED_OPEN,
    sanitize_untrusted_text,
    wrap_untrusted,
)


class TestSanitizeUntrustedText:
    def test_empty_string(self):
        assert sanitize_untrusted_text("", 100) == ""

    def test_none(self):
        assert sanitize_untrusted_text(None, 100) == ""

    def test_non_string(self):
        assert sanitize_untrusted_text(42, 100) == ""  # type: ignore[arg-type]

    def test_control_chars_removed(self):
        result = sanitize_untrusted_text("hello\x01world\x08test", 100)
        assert result == "helloworldtest"

    def test_tab_preserved(self):
        assert sanitize_untrusted_text("a\tb", 100) == "a\tb"

    def test_newline_preserved(self):
        assert sanitize_untrusted_text("a\nb", 100) == "a\nb"

    def test_zero_width_space_removed(self):
        result = sanitize_untrusted_text("a\u200bb\u200cc\u200dd", 100)
        assert "\u200b" not in result
        assert "\u200c" not in result
        assert "\u200d" not in result
        assert result == "abcd"

    def test_bom_removed(self):
        result = sanitize_untrusted_text("\ufeffhello", 100)
        assert result == "hello"

    def test_lrm_rlm_removed(self):
        result = sanitize_untrusted_text("a\u200eb\u200fc", 100)
        assert result == "abc"

    def test_length_cap(self):
        result = sanitize_untrusted_text("a" * 1000, 50)
        assert len(result) == 50
        assert result == "a" * 50

    def test_length_cap_zero(self):
        assert sanitize_untrusted_text("hello", 0) == ""

    def test_null_byte_removed(self):
        assert sanitize_untrusted_text("a\x00b", 100) == "ab"

    def test_japanese_preserved(self):
        text = "日本語のテキスト"
        assert sanitize_untrusted_text(text, 100) == text

    def test_combined_attack(self):
        # Mix of zero-width, control char, null byte, and normal text
        payload = "hello\u200b\x01world\x00\ufeff!"
        assert sanitize_untrusted_text(payload, 100) == "helloworld!"


class TestWrapUntrusted:
    def test_wraps_in_delimiter(self):
        result = wrap_untrusted("hello")
        assert result == f"{UNTRUSTED_OPEN}\nhello\n{UNTRUSTED_CLOSE}"

    def test_empty_content(self):
        result = wrap_untrusted("")
        assert result == f"{UNTRUSTED_OPEN}\n\n{UNTRUSTED_CLOSE}"

    def test_multiline_content(self):
        result = wrap_untrusted("line1\nline2")
        assert UNTRUSTED_OPEN in result
        assert UNTRUSTED_CLOSE in result
        assert "line1\nline2" in result
