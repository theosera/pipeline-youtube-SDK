"""Tests for transcript/whisper_fallback.py.

All tests mock the whisper and yt-dlp imports — no actual audio
download or model inference happens.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pipeline_youtube.transcript.base import TranscriptNotAvailable, TranscriptSource
from pipeline_youtube.transcript.whisper_fallback import (
    _noop_lock,
    _segments_to_snippets,
    fetch_whisper,
)


class TestSegmentsToSnippets:
    def test_basic_conversion(self):
        segments = [
            {"start": 0.0, "end": 3.5, "text": "Hello world"},
            {"start": 3.5, "end": 7.0, "text": "Second segment"},
        ]
        snippets = _segments_to_snippets(segments)
        assert len(snippets) == 2
        assert snippets[0].text == "Hello world"
        assert snippets[0].start == 0.0
        assert snippets[0].duration == 3.5
        assert snippets[1].start == 3.5

    def test_empty_text_filtered(self):
        segments = [
            {"start": 0.0, "end": 1.0, "text": ""},
            {"start": 1.0, "end": 2.0, "text": "   "},
            {"start": 2.0, "end": 3.0, "text": "Valid"},
        ]
        snippets = _segments_to_snippets(segments)
        assert len(snippets) == 1
        assert snippets[0].text == "Valid"

    def test_empty_segments(self):
        assert _segments_to_snippets([]) == []

    def test_missing_fields_use_defaults(self):
        segments = [{"text": "Only text"}]
        snippets = _segments_to_snippets(segments)
        assert len(snippets) == 1
        assert snippets[0].start == 0.0
        assert snippets[0].duration == 0.0


class TestNoopLock:
    def test_context_manager(self):
        with _noop_lock() as lock:
            assert lock is not None


class TestFetchWhisper:
    @patch("pipeline_youtube.transcript.whisper_fallback.whisper", create=True)
    def test_whisper_not_installed_raises(self, _mock_whisper):
        """When whisper import fails, TranscriptNotAvailable is raised."""
        with (
            patch.dict("sys.modules", {"whisper": None}),
            patch(
                "pipeline_youtube.transcript.whisper_fallback.fetch_whisper",
                side_effect=TranscriptNotAvailable("whisper_not_installed"),
            ),
            pytest.raises(TranscriptNotAvailable, match="whisper_not_installed"),
        ):
            fetch_whisper("test_id", ["ja"])

    @patch("pipeline_youtube.transcript.whisper_fallback._download_audio")
    @patch("pipeline_youtube.transcript.whisper_fallback._run_whisper")
    def test_successful_transcription(self, mock_run, mock_download, tmp_path):
        """Happy path: download audio, run whisper, return result."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_text("fake audio")
        mock_download.return_value = audio_file
        mock_run.return_value = [
            {"start": 0.0, "end": 5.0, "text": "テスト"},
            {"start": 5.0, "end": 10.0, "text": "成功"},
        ]

        # Mock whisper import check
        with patch.dict("sys.modules", {"whisper": MagicMock()}):
            result = fetch_whisper("test_id", ["ja"])

        assert result.source == TranscriptSource.WHISPER
        assert result.language == "ja"
        assert len(result.snippets) == 2
        assert result.snippets[0].text == "テスト"

    @patch("pipeline_youtube.transcript.whisper_fallback._download_audio")
    @patch("pipeline_youtube.transcript.whisper_fallback._run_whisper")
    def test_no_segments_raises(self, mock_run, mock_download, tmp_path):
        """When whisper produces no segments, TranscriptNotAvailable is raised."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_text("fake audio")
        mock_download.return_value = audio_file
        mock_run.return_value = []

        with (
            patch.dict("sys.modules", {"whisper": MagicMock()}),
            pytest.raises(TranscriptNotAvailable, match="whisper_produced_no_segments"),
        ):
            fetch_whisper("test_id", ["ja"])

    @patch("pipeline_youtube.transcript.whisper_fallback._download_audio")
    def test_download_failure_raises(self, mock_download):
        """When audio download fails, TranscriptNotAvailable is raised."""
        mock_download.side_effect = TranscriptNotAvailable("audio_download_failed: 404")

        with (
            patch.dict("sys.modules", {"whisper": MagicMock()}),
            pytest.raises(TranscriptNotAvailable, match="audio_download_failed"),
        ):
            fetch_whisper("test_id", ["en"])

    @patch("pipeline_youtube.transcript.whisper_fallback._download_audio")
    @patch("pipeline_youtube.transcript.whisper_fallback._run_whisper")
    def test_cleanup_audio_file(self, mock_run, mock_download, tmp_path):
        """Audio file is deleted after transcription, even on success."""
        audio_file = tmp_path / "whisper_test.mp3"
        audio_file.write_text("fake")
        mock_download.return_value = audio_file
        mock_run.return_value = [{"start": 0, "end": 1, "text": "ok"}]

        with patch.dict("sys.modules", {"whisper": MagicMock()}):
            fetch_whisper("test", ["ja"])

        assert not audio_file.exists()

    @patch("pipeline_youtube.transcript.whisper_fallback._download_audio")
    @patch("pipeline_youtube.transcript.whisper_fallback._run_whisper")
    def test_empty_languages_auto_detects(self, mock_run, mock_download, tmp_path):
        """When languages list is empty, language hint is None (auto-detect)."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_text("fake")
        mock_download.return_value = audio_file
        mock_run.return_value = [{"start": 0, "end": 1, "text": "test"}]

        with patch.dict("sys.modules", {"whisper": MagicMock()}):
            result = fetch_whisper("test_id", [])

        assert result.language is None
        mock_run.assert_called_once_with(audio_file, model_name="small", language=None)
