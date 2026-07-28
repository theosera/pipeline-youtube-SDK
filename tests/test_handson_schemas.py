"""Verifies handson/schemas.py: fmt_hms rendering and the parse_* validators.

Covers the long-video invariant (H:MM:SS past 99:59), prose-tolerant JSON
extraction, defensive caps, and the HandsonParseError contract.
"""

from __future__ import annotations

import json

import pytest

from pipeline_youtube.handson.schemas import (
    HandsonParseError,
    SegmentLabel,
    fmt_hms,
    parse_moc_output,
    parse_segments,
    parse_step_body,
    parse_step_plan,
)


class TestFmtHms:
    def test_under_one_hour_renders_mmss(self):
        assert fmt_hms(65) == "01:05"
        assert fmt_hms(0) == "00:00"
        assert fmt_hms(3599) == "59:59"

    def test_over_one_hour_renders_h_mm_ss(self):
        assert fmt_hms(3600) == "1:00:00"
        assert fmt_hms(3661) == "1:01:01"

    def test_long_video_past_9959_does_not_wrap(self):
        # 2h32m05s — the legacy MM:SS renderers would emit "152:05".
        assert fmt_hms(9125) == "2:32:05"

    def test_negative_clamps_to_zero(self):
        assert fmt_hms(-5) == "00:00"


class TestParseSegments:
    def test_valid_payload(self):
        raw = json.dumps(
            {
                "segments": [
                    {"start_sec": 0, "end_sec": 1800, "label": "lecture", "summary": "本編"},
                    {"start_sec": 1800, "end_sec": 2100, "label": "qa", "summary": "質疑"},
                    {"start_sec": 2100, "end_sec": 2400, "label": "tips", "summary": "小ネタ"},
                ]
            },
            ensure_ascii=False,
        )
        segments = parse_segments(raw)
        assert [s.label for s in segments] == [
            SegmentLabel.LECTURE,
            SegmentLabel.QA,
            SegmentLabel.TIPS,
        ]
        assert segments[1].start_sec == 1800

    def test_prose_around_json_is_tolerated(self):
        raw = (
            '前置きです。\n{"segments": [{"start_sec": 0, "end_sec": 60, "label": "qa"}]}\n後置き。'
        )
        segments = parse_segments(raw)
        assert len(segments) == 1
        assert segments[0].label is SegmentLabel.QA

    def test_unknown_label_defaults_to_lecture(self):
        raw = json.dumps({"segments": [{"start_sec": 0, "end_sec": 60, "label": "monologue"}]})
        assert parse_segments(raw)[0].label is SegmentLabel.LECTURE

    def test_items_without_integer_bounds_are_dropped(self):
        raw = json.dumps(
            {
                "segments": [
                    {"start_sec": "abc", "end_sec": 60, "label": "lecture"},
                    {"end_sec": 60, "label": "lecture"},
                    {"start_sec": 10, "end_sec": 60, "label": "lecture"},
                    "not-a-dict",
                ]
            }
        )
        segments = parse_segments(raw)
        assert len(segments) == 1
        assert segments[0].start_sec == 10

    def test_float_string_seconds_are_coerced(self):
        raw = json.dumps({"segments": [{"start_sec": "12.9", "end_sec": 60.7, "label": "qa"}]})
        seg = parse_segments(raw)[0]
        assert seg.start_sec == 12
        assert seg.end_sec == 60

    def test_non_list_segments_raises(self):
        with pytest.raises(HandsonParseError, match="segments must be a list"):
            parse_segments(json.dumps({"segments": "nope"}))

    def test_garbage_raises_handson_parse_error(self):
        with pytest.raises(HandsonParseError):
            parse_segments("これは JSON ではありません")


class TestParseStepPlan:
    def test_valid_payload(self):
        raw = json.dumps(
            {
                "steps": [
                    {
                        "index": 1,
                        "label": "環境構築",
                        "start_sec": 0,
                        "end_sec": 900,
                        "goal": "動く状態にする",
                        "insight_ids": ["q001"],
                    }
                ],
                "unassigned_insight_ids": ["p001"],
            },
            ensure_ascii=False,
        )
        plan = parse_step_plan(raw)
        assert plan.steps[0].label == "環境構築"
        assert plan.steps[0].insight_ids == ("q001",)
        assert plan.unassigned_insight_ids == ("p001",)

    def test_non_dict_items_skipped(self):
        raw = json.dumps(
            {"steps": ["x", {"index": 1, "label": "a", "start_sec": 0, "end_sec": 60}]}
        )
        assert len(parse_step_plan(raw).steps) == 1

    def test_non_list_steps_raises(self):
        with pytest.raises(HandsonParseError, match="steps must be a list"):
            parse_step_plan(json.dumps({"steps": {"index": 1}}))


class TestParseStepBody:
    def test_valid_payload(self):
        label, body = parse_step_body(
            json.dumps({"label": "実装", "body_markdown": "## ゴール\nx\n\n## 手順\n1. y"})
        )
        assert label == "実装"
        assert "## 手順" in body

    def test_body_is_capped(self):
        label, body = parse_step_body(json.dumps({"label": "a", "body_markdown": "x" * 100_000}))
        assert len(body) == 50_000


class TestParseMocOutput:
    def test_valid_payload(self):
        out = parse_moc_output(
            json.dumps({"title": "T", "moc_markdown": "# T", "summary_markdown": "## Q&A から"})
        )
        assert out.title == "T"
        assert out.summary_markdown.startswith("## Q&A")

    def test_missing_keys_default_to_empty(self):
        out = parse_moc_output(json.dumps({"title": "T"}))
        assert out.moc_markdown == ""
        assert out.summary_markdown == ""
