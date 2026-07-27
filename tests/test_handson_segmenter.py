"""Verifies handson/segmenter.py: deterministic normalization, snapping,
long-video preservation, the repair→fallback ladder, and insight numbering.
"""

from __future__ import annotations

import json

import pytest

from pipeline_youtube.handson import segmenter as seg_mod
from pipeline_youtube.handson.schemas import Segment, SegmentLabel
from pipeline_youtube.handson.segmenter import (
    build_insights,
    classify_segments,
    normalize_segments,
    slice_transcript_text,
)
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.providers.base import LLMError, LLMResponse
from pipeline_youtube.services.cache import Cache
from pipeline_youtube.transcript.base import TranscriptSnippet

_NO_CACHE = Cache(None, enabled=False)


def _video() -> VideoMeta:
    return VideoMeta(
        video_id="vid001",
        title="Long Talk",
        url="https://www.youtube.com/watch?v=vid001",
        duration=9000,
        channel="Test",
        upload_date="20260601",
        playlist_title=None,
    )


def _snippets(total_sec: int = 9000, step: int = 30) -> list[TranscriptSnippet]:
    return [
        TranscriptSnippet(f"テキスト{i}", float(i * step), float(step))
        for i in range(total_sec // step)
    ]


def _seg(
    start: int, end: int, label: SegmentLabel = SegmentLabel.LECTURE, summary: str = ""
) -> Segment:
    return Segment(start, end, label, summary)


def _response(text: str) -> LLMResponse:
    return LLMResponse(text=text, model="sonnet", input_tokens=10, output_tokens=10)


class TestNormalizeSegments:
    def test_gap_is_closed_by_extending_previous(self):
        out = normalize_segments(
            [_seg(0, 300), _seg(600, 900, SegmentLabel.QA)], duration=900, chunk_starts=[]
        )
        assert [(s.start_sec, s.end_sec) for s in out] == [(0, 600), (600, 900)]

    def test_overlap_is_trimmed_from_current(self):
        out = normalize_segments(
            [_seg(0, 600), _seg(300, 900, SegmentLabel.QA)], duration=900, chunk_starts=[]
        )
        assert [(s.start_sec, s.end_sec) for s in out] == [(0, 600), (600, 900)]

    def test_first_start_and_last_end_are_forced(self):
        out = normalize_segments([_seg(120, 700)], duration=900, chunk_starts=[])
        assert out[0].start_sec == 0
        assert out[-1].end_sec == 900

    def test_boundary_snaps_within_45s(self):
        out = normalize_segments(
            [_seg(0, 644), _seg(644, 1200, SegmentLabel.QA)],
            duration=1200,
            chunk_starts=[0, 600, 1200],
        )
        assert out[0].end_sec == 600
        assert out[1].start_sec == 600

    def test_boundary_beyond_45s_is_kept(self):
        out = normalize_segments(
            [_seg(0, 646), _seg(646, 1200, SegmentLabel.QA)],
            duration=1200,
            chunk_starts=[0, 1200],
        )
        assert out[0].end_sec == 646

    def test_tiny_fragment_merges_into_previous(self):
        out = normalize_segments(
            [_seg(0, 600), _seg(600, 610, SegmentLabel.TIPS), _seg(610, 900)],
            duration=900,
            chunk_starts=[],
        )
        # 10s TIPS fragment absorbed by the previous LECTURE span.
        assert [(s.start_sec, s.end_sec, s.label) for s in out] == [
            (0, 610, SegmentLabel.LECTURE),
            (610, 900, SegmentLabel.LECTURE),
        ]

    def test_tiny_first_fragment_merges_into_next(self):
        out = normalize_segments(
            [_seg(0, 10, SegmentLabel.TIPS), _seg(10, 900)], duration=900, chunk_starts=[]
        )
        assert len(out) == 1
        assert (out[0].start_sec, out[0].end_sec) == (0, 900)
        assert out[0].label is SegmentLabel.LECTURE

    def test_long_video_boundaries_past_9959_survive(self):
        # 2h30m video with a QA session starting at 2h — beyond the legacy
        # 99:59 MM:SS regex ceiling. Integer seconds must pass unharmed.
        out = normalize_segments(
            [_seg(0, 7200), _seg(7200, 9000, SegmentLabel.QA)],
            duration=9000,
            chunk_starts=[0, 7200],
        )
        assert [(s.start_sec, s.end_sec) for s in out] == [(0, 7200), (7200, 9000)]
        assert out[1].label is SegmentLabel.QA

    def test_empty_input_returns_empty(self):
        assert normalize_segments([], duration=900, chunk_starts=[]) == []

    def test_out_of_range_only_returns_empty(self):
        assert normalize_segments([_seg(950, 990)], duration=900, chunk_starts=[]) == []


class TestClassifySegments:
    def test_valid_response_is_normalized(self, monkeypatch):
        payload = json.dumps(
            {
                "segments": [
                    {"start_sec": 0, "end_sec": 3600, "label": "lecture", "summary": "本編"},
                    {"start_sec": 3600, "end_sec": 3900, "label": "qa", "summary": "質疑"},
                    {"start_sec": 3900, "end_sec": 9000, "label": "lecture", "summary": "続き"},
                ]
            },
            ensure_ascii=False,
        )
        calls = {"n": 0}

        def fake_invoke(**kw):
            calls["n"] += 1
            return _response(payload)

        monkeypatch.setattr(seg_mod, "invoke_claude", fake_invoke)
        segments, responses, fallback = classify_segments(
            _video(), _snippets(), duration=9000, model="sonnet", cache=_NO_CACHE
        )
        assert fallback is None
        assert calls["n"] == 1
        assert len(responses) == 1
        assert [s.label for s in segments] == [
            SegmentLabel.LECTURE,
            SegmentLabel.QA,
            SegmentLabel.LECTURE,
        ]

    def test_two_garbage_responses_fall_back_to_all_lecture(self, monkeypatch):
        calls = {"n": 0}

        def fake_invoke(**kw):
            calls["n"] += 1
            return _response("garbage not json")

        monkeypatch.setattr(seg_mod, "invoke_claude", fake_invoke)
        segments, responses, fallback = classify_segments(
            _video(), _snippets(), duration=9000, model="sonnet", cache=_NO_CACHE
        )
        assert calls["n"] == 2  # initial + one repair
        assert fallback is not None and "segment_parse_failed" in fallback
        assert len(segments) == 1
        assert segments[0].label is SegmentLabel.LECTURE
        assert (segments[0].start_sec, segments[0].end_sec) == (0, 9000)

    def test_repair_prompt_carries_the_defect(self, monkeypatch):
        prompts: list[str] = []
        payload = json.dumps({"segments": [{"start_sec": 0, "end_sec": 9000, "label": "lecture"}]})

        def fake_invoke(**kw):
            prompts.append(kw["prompt"])
            if len(prompts) == 1:
                return _response("not json at all")
            return _response(payload)

        monkeypatch.setattr(seg_mod, "invoke_claude", fake_invoke)
        segments, _, fallback = classify_segments(
            _video(), _snippets(), duration=9000, model="sonnet", cache=_NO_CACHE
        )
        assert fallback is None
        assert len(prompts) == 2
        assert "前回の出力は JSON として解釈できませんでした" in prompts[1]

    def test_llm_error_falls_back(self, monkeypatch):
        def fake_invoke(**kw):
            raise LLMError("boom")

        monkeypatch.setattr(seg_mod, "invoke_claude", fake_invoke)
        segments, responses, fallback = classify_segments(
            _video(), _snippets(), duration=9000, model="sonnet", cache=_NO_CACHE
        )
        assert fallback is not None and fallback.startswith("segment_call_failed")
        assert segments[0].label is SegmentLabel.LECTURE
        assert responses == []

    def test_empty_transcript_falls_back_without_llm_call(self, monkeypatch):
        def fake_invoke(**kw):  # pragma: no cover - must not be reached
            pytest.fail("invoke_claude must not be called for empty transcripts")

        monkeypatch.setattr(seg_mod, "invoke_claude", fake_invoke)
        segments, responses, fallback = classify_segments(
            _video(), [], duration=9000, model="sonnet", cache=_NO_CACHE
        )
        assert fallback == "empty_transcript_chunks"
        assert len(segments) == 1


class TestBuildInsights:
    def test_ids_are_numbered_per_label_in_timeline_order(self):
        segments = [
            _seg(0, 600),
            _seg(600, 900, SegmentLabel.QA, "最初の質疑"),
            _seg(900, 1500),
            _seg(1500, 1600, SegmentLabel.TIPS, "小ネタ"),
            _seg(1600, 2200, SegmentLabel.QA, "二度目の質疑"),
        ]
        insights = build_insights(segments, _snippets(2400))
        assert [(i.insight_id, i.label) for i in insights] == [
            ("q001", SegmentLabel.QA),
            ("p001", SegmentLabel.TIPS),
            ("q002", SegmentLabel.QA),
        ]

    def test_quote_comes_from_the_segment_slice(self):
        segments = [_seg(0, 60), _seg(60, 120, SegmentLabel.QA, "質疑")]
        insights = build_insights(segments, _snippets(120))
        assert "テキスト2" in insights[0].quote  # snippet starting at 60s
        assert "テキスト0" not in insights[0].quote

    def test_slice_text_is_range_bounded(self):
        text = slice_transcript_text(_snippets(120), 30, 90)
        assert "テキスト1" in text and "テキスト2" in text
        assert "テキスト0" not in text and "テキスト3" not in text
