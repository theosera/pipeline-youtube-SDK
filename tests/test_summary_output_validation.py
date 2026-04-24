"""Tests for #2: Stage 02 LLM output structural validation."""

from __future__ import annotations

import pytest

from pipeline_youtube.stages.summary import (
    SummaryOutputError,
    _validate_summary_output,
)

VALID = (
    "## 全体サマリ\nsummary body.\n\n"
    "## 要点タイムライン\n\n"
    "### [00:00 ~ 00:30] intro\n本文\n\n"
    "### [01:00 ~ 02:00] middle\n本文\n"
)


class TestValidatorAccepts:
    def test_minimal_valid(self):
        out = _validate_summary_output(VALID)
        assert "## 全体サマリ" in out
        assert "### [00:00 ~ 00:30]" in out

    def test_accepts_fullwidth_tilde(self):
        body = VALID.replace("~ 00:30", "〜 00:30", 1)
        _validate_summary_output(body)


class TestValidatorRejects:
    def test_missing_summary_section(self):
        body = "## 要点タイムライン\n\n### [00:00 ~ 00:30] h\n本文\n"
        with pytest.raises(SummaryOutputError, match="missing"):
            _validate_summary_output(body)

    def test_missing_timeline_section(self):
        body = "## 全体サマリ\n\n本文\n\n### [00:00 ~ 00:30] h\n本文\n"
        with pytest.raises(SummaryOutputError, match="missing"):
            _validate_summary_output(body)

    def test_no_range_heading(self):
        body = "## 全体サマリ\n\nbody\n\n## 要点タイムライン\n\nno ranges here\n"
        with pytest.raises(SummaryOutputError, match="no `### "):
            _validate_summary_output(body)

    def test_too_long(self):
        body = VALID + "x" * 50_001
        with pytest.raises(SummaryOutputError, match="exceeds"):
            _validate_summary_output(body)


class TestValidatorStripsUnsafe:
    def test_strips_script_tag(self):
        body = VALID + "\n<script>alert(1)</script>\n"
        out = _validate_summary_output(body)
        assert "<script>" not in out

    def test_strips_templater_tokens(self):
        body = VALID + "\n<% tp.user.name %>\n"
        out = _validate_summary_output(body)
        assert "<%" not in out
        assert "%>" not in out

    def test_drops_unexpected_embeds(self):
        """Stage 02 never legitimately uses `![[...]]`; any occurrence is stripped."""
        body = VALID + "\n![[evil.webp]]\n"
        out = _validate_summary_output(body)
        assert "![[evil.webp]]" not in out
        assert "dropped embed" in out

    def test_preserves_flashcards(self):
        body = VALID + "\n## フラッシュカード\n#flashcards\nQ: what?\nA: this.\n"
        out = _validate_summary_output(body)
        assert "#flashcards" in out
        assert "Q: what?" in out
