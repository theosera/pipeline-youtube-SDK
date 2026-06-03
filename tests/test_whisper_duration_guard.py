"""Whisper long-form guard: skip multi-hour audio before it blocks the queue.

CPU transcription of day-long VODs runs for hours and, since Whisper
concurrency is bounded (default 1), one such video stalls every other.
``_guard_audio_duration`` probes the duration from yt-dlp metadata *before*
download and raises ``TranscriptNotAvailable("audio_too_long: ...")`` so the
fallback chain ends gracefully without occupying a Whisper slot.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from pipeline_youtube.transcript import whisper_fallback as wf
from pipeline_youtube.transcript.base import TranscriptNotAvailable


def _install_fake_ytdlp(monkeypatch: pytest.MonkeyPatch, info: object) -> None:
    """Register a fake ``yt_dlp`` whose extract_info returns ``info``."""

    class _FakeYDL:
        def __init__(self, opts: dict[str, Any]) -> None:
            self._opts = opts

        def __enter__(self) -> _FakeYDL:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def extract_info(self, url: str, download: bool = True) -> object:
            assert download is False  # probe must never download the audio
            return info

    fake = types.ModuleType("yt_dlp")
    fake.YoutubeDL = _FakeYDL  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yt_dlp", fake)


class TestConfigure:
    def test_default_is_two_hours(self):
        assert wf.DEFAULT_WHISPER_MAX_AUDIO_SECONDS == 2 * 60 * 60
        assert wf._max_audio_seconds == wf.DEFAULT_WHISPER_MAX_AUDIO_SECONDS

    def test_configure_sets_global(self):
        try:
            wf.configure_whisper_max_audio_seconds(60)
            assert wf._max_audio_seconds == 60
        finally:
            wf.configure_whisper_max_audio_seconds(wf.DEFAULT_WHISPER_MAX_AUDIO_SECONDS)


class TestProbeDuration:
    def test_reads_duration_from_metadata(self, monkeypatch):
        _install_fake_ytdlp(monkeypatch, {"duration": 1234.5})
        assert wf._probe_duration_seconds("vid") == pytest.approx(1234.5)

    def test_missing_duration_returns_none(self, monkeypatch):
        _install_fake_ytdlp(monkeypatch, {"title": "live, no duration"})
        assert wf._probe_duration_seconds("vid") is None

    def test_non_dict_info_returns_none(self, monkeypatch):
        _install_fake_ytdlp(monkeypatch, None)
        assert wf._probe_duration_seconds("vid") is None

    def test_extract_failure_returns_none(self, monkeypatch):
        fake = types.ModuleType("yt_dlp")

        class _Boom:
            def __init__(self, opts):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def extract_info(self, url, download=True):
                raise RuntimeError("network down")

        fake.YoutubeDL = _Boom  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "yt_dlp", fake)
        assert wf._probe_duration_seconds("vid") is None


class TestGuard:
    def test_raises_when_over_limit(self, monkeypatch):
        monkeypatch.setattr(wf, "_probe_duration_seconds", lambda _vid: 5 * 3600.0)
        monkeypatch.setattr(wf, "_max_audio_seconds", 2 * 3600)
        with pytest.raises(TranscriptNotAvailable) as exc:
            wf._guard_audio_duration("vid")
        assert "audio_too_long" in str(exc.value)

    def test_passes_when_under_limit(self, monkeypatch):
        monkeypatch.setattr(wf, "_probe_duration_seconds", lambda _vid: 600.0)
        monkeypatch.setattr(wf, "_max_audio_seconds", 2 * 3600)
        wf._guard_audio_duration("vid")  # no raise

    def test_passes_when_duration_unknown(self, monkeypatch):
        monkeypatch.setattr(wf, "_probe_duration_seconds", lambda _vid: None)
        monkeypatch.setattr(wf, "_max_audio_seconds", 2 * 3600)
        wf._guard_audio_duration("vid")  # no raise

    def test_disabled_when_limit_non_positive(self, monkeypatch):
        # A non-positive cap must short-circuit without even probing.
        called = {"n": 0}

        def _probe(_vid: str) -> float:
            called["n"] += 1
            return 10 * 3600.0

        monkeypatch.setattr(wf, "_probe_duration_seconds", _probe)
        monkeypatch.setattr(wf, "_max_audio_seconds", 0)
        wf._guard_audio_duration("vid")  # no raise
        assert called["n"] == 0
