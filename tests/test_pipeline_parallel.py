"""Tests for WS4: Stage 03 download runs concurrently with Stage 02 LLM call."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.stages.capture import VideoPrefetch, prefetch_video_download


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
        )

        assert called["download"] == 0
        assert called["extract"] == 1
        assert result.outcomes and result.outcomes[0].success
