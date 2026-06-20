"""Tests for the upfront transcript cache warm-up (Phase 3 C)."""

from __future__ import annotations

import pytest

from pipeline_youtube.cache import configure_cache
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.services.cache import Cache
from pipeline_youtube.stages import scripts as scripts_mod
from pipeline_youtube.transcript.base import (
    TranscriptNotAvailable,
    TranscriptSource,
    build_result,
)


def _videos(n: int) -> list[VideoMeta]:
    return [
        VideoMeta(
            video_id=f"vid{i:05d}",
            title=f"v{i}",
            url=f"https://www.youtube.com/watch?v=vid{i:05d}",
            duration=60,
            channel="ch",
            upload_date=None,
            playlist_title=None,
        )
        for i in range(n)
    ]


def _official_ok(video_id: str, languages: list[str]):
    from pipeline_youtube.transcript.base import TranscriptSnippet

    return build_result(
        video_id=video_id,
        source=TranscriptSource.OFFICIAL,
        language="ja",
        snippets=[TranscriptSnippet("字幕", 0.0, 1.0)],
    )


def _not_available(video_id: str, languages: list[str]):
    raise TranscriptNotAvailable("none")


class TestWarmTranscriptCache:
    @pytest.fixture(autouse=True)
    def _no_live_innertube(self, monkeypatch):
        # warm_transcript_cache defaults to use_innertube=True, so it would call
        # the real fetch_innertube (live YouTube HTTP) ahead of the mocked
        # official/auto tiers. With fake video IDs this makes network requests
        # that 20s-timeout or flake on offline/restricted CI runners. Stub the
        # InnerTube tier offline so the warm-up tests stay hermetic.
        monkeypatch.setattr(scripts_mod, "fetch_innertube", _not_available)

    def test_noop_when_cache_disabled(self, monkeypatch):
        # Default cache is disabled; warming must do nothing (and not fetch).
        calls = {"n": 0}

        def counting(video_id, languages):
            calls["n"] += 1
            return _official_ok(video_id, languages)

        monkeypatch.setattr(scripts_mod, "fetch_official", counting)
        monkeypatch.setattr(scripts_mod, "fetch_auto", _not_available)

        warmed = scripts_mod.warm_transcript_cache(_videos(3), cache=Cache(None, enabled=False))
        assert warmed == 0
        assert calls["n"] == 0

    def test_warms_and_populates_cache(self, tmp_path, monkeypatch):
        cache = configure_cache(tmp_path, enabled=True)
        monkeypatch.setattr(scripts_mod, "fetch_official", _official_ok)
        monkeypatch.setattr(scripts_mod, "fetch_auto", _not_available)

        videos = _videos(5)
        warmed = scripts_mod.warm_transcript_cache(videos, concurrency=4, cache=cache)

        assert warmed == 5
        # Every video's official tier is now cached.
        for v in videos:
            assert cache.get_transcript(v.video_id, "official", "ja") is not None

    def test_videos_without_captions_are_not_counted(self, tmp_path, monkeypatch):
        cache = configure_cache(tmp_path, enabled=True)
        monkeypatch.setattr(scripts_mod, "fetch_official", _not_available)
        monkeypatch.setattr(scripts_mod, "fetch_auto", _not_available)

        warmed = scripts_mod.warm_transcript_cache(_videos(3), cache=cache)
        assert warmed == 0

    def test_per_video_errors_are_swallowed(self, tmp_path, monkeypatch):
        cache = configure_cache(tmp_path, enabled=True)

        def boom(video_id, languages):
            raise RuntimeError("network down")

        monkeypatch.setattr(scripts_mod, "fetch_official", boom)
        monkeypatch.setattr(scripts_mod, "fetch_auto", boom)

        # Must not raise even though every fetch errors hard.
        warmed = scripts_mod.warm_transcript_cache(_videos(2), cache=cache)
        assert warmed == 0

    def test_empty_list_returns_zero(self):
        assert scripts_mod.warm_transcript_cache([], cache=Cache(None, enabled=False)) == 0
