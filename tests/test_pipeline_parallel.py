"""Tests for WS4: Stage 03 download runs concurrently with Stage 02 LLM call."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.services.cache import Cache
from pipeline_youtube.stages.capture import VideoPrefetch, prefetch_video_download

# Capture-path tests below stub the backend and don't exercise persistent
# caching, so they thread a disabled (no-op) cache.
_NO_CACHE = Cache(None, enabled=False)


def _video() -> VideoMeta:
    return VideoMeta(
        video_id="abc1234567",
        title="test",
        url="https://www.youtube.com/watch?v=abc1234567",
        duration=60,
        channel="ch",
        upload_date=None,
        playlist_title=None,
    )


class TestPrefetchHandle:
    def test_wait_returns_none_on_success(self, tmp_path: Path):
        def fake_download(url: str, dest: Path, resolution: str = "480", **kw: Any) -> None:
            dest.write_bytes(b"fake mp4")

        with patch("pipeline_youtube.stages.capture._download_video", fake_download):
            handle = prefetch_video_download(_video())
            assert isinstance(handle, VideoPrefetch)
            assert handle.wait(timeout=5.0) is None
            assert handle.path.exists()
            handle.path.unlink(missing_ok=True)

    def test_wait_returns_exception_on_failure(self):
        def fake_download(url: str, dest: Path, resolution: str = "480", **kw: Any) -> None:
            raise RuntimeError("boom")

        with patch("pipeline_youtube.stages.capture._download_video", fake_download):
            handle = prefetch_video_download(_video())
            err = handle.wait(timeout=5.0)
            assert isinstance(err, RuntimeError)
            assert "boom" in str(err)


class TestParallelOverlap:
    @pytest.mark.asyncio
    async def test_download_overlaps_with_llm(self, tmp_path: Path, monkeypatch):
        """If download and LLM each take 0.5s, sequential is ~1s, parallel is ~0.5s."""

        def slow_download(url: str, dest: Path, resolution: str = "480", **kw: Any) -> None:
            time.sleep(0.5)
            dest.write_bytes(b"x")

        # Start prefetch; simulate Stage 02 by sleeping the same amount
        with patch("pipeline_youtube.stages.capture._download_video", slow_download):
            t0 = time.monotonic()
            handle = prefetch_video_download(_video())
            time.sleep(0.5)  # simulate Stage 02 LLM latency
            err = handle.wait(timeout=5.0)
            elapsed = time.monotonic() - t0

        assert err is None
        # Allow some overhead but must be well under sequential 1.0s
        assert elapsed < 0.9, f"expected overlap, got elapsed={elapsed:.2f}s"
        handle.path.unlink(missing_ok=True)


class TestPrefetchedPathConsumed:
    def test_capture_skips_download_when_prefetch_present(self, tmp_path: Path, monkeypatch):
        from pipeline_youtube.stages import capture as cap_mod

        # Prepare a fake summary md with one range and a fake prefetched video
        summary_md = tmp_path / "02.md"
        summary_md.write_text(
            "---\n---\n\n## 要点タイムライン\n### [00:00 ~ 00:05] heading\n本文\n",
            encoding="utf-8",
        )
        capture_md = tmp_path / "03.md"
        capture_md.write_text("---\n---\n", encoding="utf-8")

        fake_video = tmp_path / "fake.mp4"
        fake_video.write_bytes(b"x")

        called: dict[str, int] = {"download": 0, "extract": 0}

        def never_download(*args: Any, **kwargs: Any) -> None:
            called["download"] += 1

        def fake_extractor(video_path: Path, output_path: Path, **kwargs: Any) -> None:
            called["extract"] += 1
            output_path.write_bytes(b"img")

        monkeypatch.setattr(cap_mod, "_download_video", never_download)
        monkeypatch.setattr(cap_mod, "_dispatch_extractor", lambda _strategy: fake_extractor)
        monkeypatch.setattr(
            cap_mod,
            "_resolve_capture_format",
            lambda _req, _backend: cap_mod._FormatChoice(ext="webp", strategy="direct"),
        )
        monkeypatch.setattr(cap_mod, "get_vault_root", lambda: tmp_path)
        monkeypatch.setattr(cap_mod, "ensure_safe_path", lambda p: p)

        result = cap_mod.run_stage_capture(
            _video(),
            summary_md,
            capture_md,
            prefetched_video_path=fake_video,
            cache=_NO_CACHE,
        )

        assert called["download"] == 0
        assert called["extract"] == 1
        assert result.outcomes and result.outcomes[0].success
        # No network download happened, so the flag must report False.
        assert result.video_downloaded is False

    def test_capture_fails_closed_when_local_media_source_missing(
        self, tmp_path: Path, monkeypatch
    ):
        """--local-media (allow_download=False) must never fall back to the cache
        or a YouTube download."""
        from pipeline_youtube.stages import capture as cap_mod

        summary_md = tmp_path / "02.md"
        summary_md.write_text(
            "---\n---\n\n## 要点タイムライン\n### [00:00 ~ 00:05] heading\n本文\n",
            encoding="utf-8",
        )
        capture_md = tmp_path / "03.md"
        capture_md.write_text("---\n---\n", encoding="utf-8")
        missing_video = tmp_path / "missing.mp4"

        called: dict[str, int] = {"download": 0}

        def never_download(*args: Any, **kwargs: Any) -> None:
            called["download"] += 1

        monkeypatch.setattr(cap_mod, "_download_video", never_download)
        monkeypatch.setattr(
            cap_mod,
            "_resolve_capture_format",
            lambda _req, _backend: cap_mod._FormatChoice(ext="webp", strategy="direct"),
        )
        monkeypatch.setattr(cap_mod, "get_vault_root", lambda: tmp_path)
        monkeypatch.setattr(cap_mod, "ensure_safe_path", lambda p: p)

        result = cap_mod.run_stage_capture(
            _video(),
            summary_md,
            capture_md,
            prefetched_video_path=missing_video,
            allow_download=False,
            cache=_NO_CACHE,
        )

        assert called["download"] == 0
        assert result.error is not None
        assert result.error.startswith("local_media_file_missing")
        assert result.outcomes == []


class TestPrefetchSkippedOnCacheHit:
    """Regression: an unconditional prefetch re-downloads the mp4 on every
    rerun and overwrites the persistent video cache, defeating it. The
    prefetch must be skipped when the video is already cached so that
    `run_stage_capture` reuses the cached copy.
    """

    def _drive_process_video(self, tmp_path: Path, monkeypatch, *, cached: bool) -> int:
        from datetime import datetime

        from pipeline_youtube import video_processing as main_mod
        from pipeline_youtube.providers.base import LLMResponse
        from pipeline_youtube.stages.capture import CaptureResult, SummaryRange
        from pipeline_youtube.transcript.base import (
            TranscriptSnippet,
            TranscriptSource,
            build_result,
        )

        prefetch_calls = {"n": 0}

        # Stub _process_video collaborators so only the prefetch decision matters.
        paths = {k: tmp_path / f"{k}.md" for k in ("scripts", "summary", "capture", "learning")}
        monkeypatch.setattr(main_mod, "compute_note_paths", lambda video, run_time: paths)
        monkeypatch.setattr(main_mod, "create_placeholder_notes", lambda *a, **kw: None)
        monkeypatch.setattr(
            main_mod,
            "run_stage_scripts",
            lambda video, path, *, dry_run, include_code_blocks=False, media_path=None, correct_model=None, known_terms=None, use_innertube=True, cache=None: (
                build_result(
                    video_id=video.video_id,
                    source=TranscriptSource.OFFICIAL,
                    language="ja",
                    snippets=[TranscriptSnippet("字幕", 0.0, 30.0)],
                )
            ),
        )
        monkeypatch.setattr(main_mod, "record_transcript_stat", lambda *a, **kw: None)
        monkeypatch.setattr(
            main_mod,
            "run_stage_summary",
            lambda *a, **kw: LLMResponse(
                text="ok",
                model="sonnet",
                input_tokens=1,
                output_tokens=1,
                cache_creation_tokens=0,
                cache_read_tokens=0,
                total_cost_usd=0.0,
                duration_ms=1,
            ),
        )
        monkeypatch.setattr(
            main_mod,
            "run_stage_capture",
            lambda *a, **kw: CaptureResult(
                ranges=[SummaryRange(0, 30, "x")], capture_format="webp"
            ),
        )

        def fake_prefetch(video, backend=None):
            prefetch_calls["n"] += 1
            return None

        monkeypatch.setattr(main_mod, "prefetch_video_download", fake_prefetch)

        class _FakeCache:
            def get_video(self, video_id: str, fmt: str):
                return (tmp_path / "cached.mp4") if cached else None

        main_mod._process_video(
            _video(),
            datetime(2026, 1, 1),
            dry_run=False,
            capture_format="auto",
            models={"stage_02": "sonnet", "stage_04": "sonnet"},
            stop_after_capture=True,  # short-circuit before Stage 04
            cache=_FakeCache(),  # only get_video is exercised (stages are stubbed)
        )
        return prefetch_calls["n"]

    def test_prefetch_skipped_when_video_cached(self, tmp_path: Path, monkeypatch):
        assert self._drive_process_video(tmp_path, monkeypatch, cached=True) == 0

    def test_prefetch_runs_on_cache_miss(self, tmp_path: Path, monkeypatch):
        assert self._drive_process_video(tmp_path, monkeypatch, cached=False) == 1
