"""Tests for pipeline_youtube.domain.results.VideoRunResult."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline_youtube.domain.results import VideoRunResult
from pipeline_youtube.domain.video import VideoMeta


def _make_video(**kwargs) -> VideoMeta:
    defaults = dict(
        video_id="dQw4w9WgXcQ",
        title="Test Video",
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        duration=300,
        channel="Test Channel",
        upload_date="20240101",
        playlist_title="Test Playlist",
    )
    defaults.update(kwargs)
    return VideoMeta(**defaults)


def _make_result(**kwargs) -> VideoRunResult:
    """Create a VideoRunResult with required video field and optional overrides."""
    video = kwargs.pop("video", _make_video())
    return VideoRunResult(video=video, **kwargs)


class TestVideoRunResultConstruction:
    def test_only_required_field(self):
        video = _make_video()
        result = VideoRunResult(video=video)
        assert result.video is video

    def test_defaults_are_none(self):
        result = _make_result()
        assert result.learning_md_path is None
        assert result.learning_md_body is None
        assert result.error is None
        assert result.transcript_cost_usd is None
        assert result.transcript_model is None
        assert result.summary_cost_usd is None
        assert result.summary_model is None
        assert result.learning_cost_usd is None
        assert result.learning_model is None

    def test_confirmed_terms_default_empty_tuple(self):
        result = _make_result()
        assert result.confirmed_terms == ()

    def test_all_fields_set(self):
        video = _make_video()
        path = Path("/tmp/out.md")
        result = VideoRunResult(
            video=video,
            learning_md_path=path,
            learning_md_body="# Learning",
            error=None,
            transcript_cost_usd=0.01,
            transcript_model="whisper-1",
            summary_cost_usd=0.02,
            summary_model="claude-haiku",
            learning_cost_usd=0.03,
            learning_model="claude-sonnet",
            confirmed_terms=("AI", "LLM"),
        )
        assert result.learning_md_path == path
        assert result.learning_md_body == "# Learning"
        assert result.transcript_cost_usd == 0.01
        assert result.transcript_model == "whisper-1"
        assert result.summary_cost_usd == 0.02
        assert result.summary_model == "claude-haiku"
        assert result.learning_cost_usd == 0.03
        assert result.learning_model == "claude-sonnet"
        assert result.confirmed_terms == ("AI", "LLM")

    def test_mutable_fields_can_be_reassigned(self):
        # VideoRunResult is NOT frozen — fields should be mutable
        result = _make_result()
        result.error = "something went wrong"
        assert result.error == "something went wrong"
        result.learning_md_body = "# Body"
        assert result.learning_md_body == "# Body"


class TestVideoRunResultOkProperty:
    def test_ok_true_when_no_error_and_body_present(self):
        result = _make_result(learning_md_body="# Content")
        assert result.ok is True

    def test_ok_false_when_error_set(self):
        result = _make_result(error="stage failed", learning_md_body="# Content")
        assert result.ok is False

    def test_ok_false_when_body_is_none(self):
        result = _make_result(learning_md_body=None)
        assert result.ok is False

    def test_ok_false_when_error_and_no_body(self):
        result = _make_result(error="failed", learning_md_body=None)
        assert result.ok is False

    def test_ok_false_when_body_none_no_error(self):
        # Both conditions must hold: no error AND body present
        result = _make_result(error=None, learning_md_body=None)
        assert result.ok is False

    def test_ok_true_with_empty_string_body(self):
        # Empty string is not None — ok checks `is not None`, not truthiness
        result = _make_result(error=None, learning_md_body="")
        assert result.ok is True

    def test_ok_becomes_true_after_mutation(self):
        result = _make_result(error=None, learning_md_body=None)
        assert result.ok is False
        result.learning_md_body = "# Content"
        assert result.ok is True

    def test_ok_becomes_false_after_error_set(self):
        result = _make_result(error=None, learning_md_body="# Content")
        assert result.ok is True
        result.error = "downstream failure"
        assert result.ok is False


class TestVideoRunResultCostFields:
    def test_zero_cost_stored(self):
        result = _make_result(transcript_cost_usd=0.0, summary_cost_usd=0.0, learning_cost_usd=0.0)
        assert result.transcript_cost_usd == 0.0
        assert result.summary_cost_usd == 0.0
        assert result.learning_cost_usd == 0.0

    def test_none_cost_distinct_from_zero(self):
        result_none = _make_result(transcript_cost_usd=None)
        result_zero = _make_result(transcript_cost_usd=0.0)
        assert result_none.transcript_cost_usd is None
        assert result_zero.transcript_cost_usd == 0.0
        assert result_none.transcript_cost_usd != result_zero.transcript_cost_usd

    def test_negative_cost_stored(self):
        # Domain doesn't enforce positivity; just stores the value
        result = _make_result(transcript_cost_usd=-0.001)
        assert result.transcript_cost_usd == pytest.approx(-0.001)


class TestVideoRunResultConfirmedTerms:
    def test_non_empty_tuple(self):
        result = _make_result(confirmed_terms=("GPT", "LLM", "Transformer"))
        assert result.confirmed_terms == ("GPT", "LLM", "Transformer")
        assert len(result.confirmed_terms) == 3

    def test_empty_tuple(self):
        result = _make_result(confirmed_terms=())
        assert result.confirmed_terms == ()
