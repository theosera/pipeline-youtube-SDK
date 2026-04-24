"""Tests for WS3: 3-layer summary output + one_liner persistence."""

from __future__ import annotations

from pipeline_youtube.stages.summary import _extract_one_liner


class TestExtractOneLiner:
    def test_extracts_leading_marker(self):
        body = "ONE_LINER: 本日の核心\n\n## 全体サマリ\n本文"
        value, rest = _extract_one_liner(body)
        assert value == "本日の核心"
        assert rest.startswith("## 全体サマリ")

    def test_quoted_value_stripped(self):
        body = 'ONE_LINER: "本日の核心"\n\n## 全体サマリ'
        value, rest = _extract_one_liner(body)
        assert value == "本日の核心"

    def test_missing_marker_returns_none(self):
        body = "## 全体サマリ\n本文\n"
        value, rest = _extract_one_liner(body)
        assert value is None
        assert rest == body

    def test_marker_must_be_first_content(self):
        body = "## 全体サマリ\nONE_LINER: 来ないよ\n"
        value, rest = _extract_one_liner(body)
        assert value is None
        assert rest == body

    def test_value_capped(self):
        body = "ONE_LINER: " + "あ" * 200 + "\n\nbody"
        value, _rest = _extract_one_liner(body)
        assert value is not None
        assert len(value) <= 60

    def test_empty_value_returns_none(self):
        body = "ONE_LINER: \n\n## 全体サマリ"
        value, _rest = _extract_one_liner(body)
        assert value is None
