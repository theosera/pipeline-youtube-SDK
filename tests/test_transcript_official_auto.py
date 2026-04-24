"""Tests for tier 1 (official.py) and tier 2 (auto.py) fetchers.

youtube-transcript-api is mocked so these tests run offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from youtube_transcript_api._errors import (
    IpBlocked,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from pipeline_youtube.transcript.auto import fetch_auto
from pipeline_youtube.transcript.base import TranscriptNotAvailable, TranscriptSource
from pipeline_youtube.transcript.official import fetch_official


@dataclass
class _FakeSnippet:
    text: str
    start: float
    duration: float


def _make_fake_transcript(snippets: list[_FakeSnippet], language_code: str = "ja") -> MagicMock:
    t = MagicMock()
    t.language_code = language_code
    t.fetch.return_value = snippets
    return t


def _make_fake_transcript_list(
    manual: MagicMock | None = None,
    generated: MagicMock | None = None,
) -> MagicMock:
    tl = MagicMock()
    if manual is None:
        tl.find_manually_created_transcript.side_effect = NoTranscriptFound("vid", ["ja"], [])
    else:
        tl.find_manually_created_transcript.return_value = manual

    if generated is None:
        tl.find_generated_transcript.side_effect = NoTranscriptFound("vid", ["ja"], [])
    else:
        tl.find_generated_transcript.return_value = generated

    return tl


class TestFetchOfficial:
    def test_success(self):
        snippets = [
            _FakeSnippet("hello", 0.0, 1.5),
            _FakeSnippet("world", 2.0, 1.5),
        ]
        fake_t = _make_fake_transcript(snippets, language_code="ja")
        tl = _make_fake_transcript_list(manual=fake_t)

        with patch("pipeline_youtube.transcript.official._get_api") as gapi:
            gapi.return_value.list.return_value = tl
            result = fetch_official("vid123", ["ja", "en"])

        assert result.source == TranscriptSource.OFFICIAL
        assert result.language == "ja"
        assert len(result.snippets) == 2
        assert result.snippets[0].text == "hello"
        assert result.snippets[0].start == 0.0

    def test_no_manual_raises_not_available(self):
        tl = _make_fake_transcript_list(manual=None)
        with patch("pipeline_youtube.transcript.official._get_api") as gapi:
            gapi.return_value.list.return_value = tl
            with pytest.raises(TranscriptNotAvailable, match="no_manual"):
                fetch_official("vid123", ["ja"])

    def test_transcripts_disabled(self):
        with patch("pipeline_youtube.transcript.official._get_api") as gapi:
            gapi.return_value.list.side_effect = TranscriptsDisabled("vid")
            with pytest.raises(TranscriptNotAvailable, match="transcripts_disabled"):
                fetch_official("vid", ["ja"])

    def test_video_unavailable(self):
        with patch("pipeline_youtube.transcript.official._get_api") as gapi:
            gapi.return_value.list.side_effect = VideoUnavailable("vid")
            with pytest.raises(TranscriptNotAvailable, match="video_unavailable"):
                fetch_official("vid", ["ja"])

    def test_ip_blocked(self):
        with patch("pipeline_youtube.transcript.official._get_api") as gapi:
            gapi.return_value.list.side_effect = IpBlocked("vid")
            with pytest.raises(TranscriptNotAvailable, match="ip_blocked"):
                fetch_official("vid", ["ja"])

    def test_empty_video_id(self):
        with pytest.raises(TranscriptNotAvailable, match="empty"):
            fetch_official("", ["ja"])

    def test_empty_snippets_raises(self):
        fake_t = _make_fake_transcript([], language_code="ja")
        tl = _make_fake_transcript_list(manual=fake_t)
        with patch("pipeline_youtube.transcript.official._get_api") as gapi:
            gapi.return_value.list.return_value = tl
            with pytest.raises(TranscriptNotAvailable, match="empty_transcript"):
                fetch_official("vid", ["ja"])

    def test_unexpected_exception_wrapped(self):
        with patch("pipeline_youtube.transcript.official._get_api") as gapi:
            gapi.return_value.list.side_effect = RuntimeError("boom")
            with pytest.raises(TranscriptNotAvailable, match="unexpected"):
                fetch_official("vid", ["ja"])


class TestFetchAuto:
    def test_success(self):
        snippets = [
            _FakeSnippet("auto text", 0.0, 2.0),
            _FakeSnippet("more", 2.0, 2.0),
        ]
        fake_t = _make_fake_transcript(snippets, language_code="ja")
        tl = _make_fake_transcript_list(generated=fake_t)

        # auto.py imports `_get_api` from official, so the binding in
        # auto's namespace must be patched (not official's).
        with patch("pipeline_youtube.transcript.auto._get_api") as gapi:
            gapi.return_value.list.return_value = tl
            result = fetch_auto("vid123", ["ja"])

        assert result.source == TranscriptSource.AUTO
        assert result.language == "ja"
        assert len(result.snippets) == 2

    def test_no_generated_raises(self):
        tl = _make_fake_transcript_list(generated=None)
        with patch("pipeline_youtube.transcript.auto._get_api") as gapi:
            gapi.return_value.list.return_value = tl
            with pytest.raises(TranscriptNotAvailable, match="no_auto"):
                fetch_auto("vid", ["ja"])
