"""Verifies stages/capture.capture_step_clips: directly-constructed ranges
(long-video safe), range-positional naming, per-range failure isolation,
cache reuse, dry-run, and the never-raise download degradation.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.services.cache import Cache
from pipeline_youtube.stages.capture import (
    CaptureResult,
    SummaryRange,
    capture_step_clips,
)

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


def _ranges() -> list[SummaryRange]:
    # Includes a range starting past 99:59 (7500s = 2:05:00): constructed
    # directly, so the MM:SS regex ceiling never applies.
    return [
        SummaryRange(0, 600, "step1"),
        SummaryRange(3900, 7200, "step2"),
        SummaryRange(7500, 9000, "step3"),
    ]


class _FakeBackend:
    """CaptureBackend stand-in: libwebp available, records ffmpeg calls."""

    def __init__(self, fail_call_indices: set[int] | None = None):
        self.ffmpeg_calls: list[list[str]] = []
        self.download_calls: list[str] = []
        self._fail = fail_call_indices or set()

    def ffmpeg_encoders(self) -> set[str]:
        return {"libwebp"}

    def has_gif2webp(self) -> bool:
        return False

    def ffmpeg(self, args: list[str], timeout: int = 180) -> None:
        call_index = len(self.ffmpeg_calls)
        self.ffmpeg_calls.append(list(args))
        if call_index in self._fail:
            raise subprocess.CalledProcessError(1, ["ffmpeg"], stderr=b"boom")
        # The output path is the last argument; create the file like ffmpeg.
        Path(args[-1]).write_bytes(b"webp")

    def download_video(self, url: str, dest: Path, resolution: str = "480") -> None:
        self.download_calls.append(url)
        dest.write_bytes(b"mp4")


class _FailingDownloadBackend(_FakeBackend):
    def download_video(self, url: str, dest: Path, resolution: str = "480") -> None:
        raise RuntimeError("network down")


class _ExplodingBackend(_FakeBackend):
    """Any use is a test failure (dry-run must not touch the backend)."""

    def ffmpeg_encoders(self) -> set[str]:  # pragma: no cover - must not run
        pytest.fail("backend must not be consulted in dry-run")

    def download_video(self, url: str, dest: Path, resolution: str = "480") -> None:
        pytest.fail("backend must not download in dry-run")  # pragma: no cover


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    (tmp_path / ".obsidian").mkdir()
    return tmp_path


class TestCaptureStepClips:
    def test_dry_run_returns_ranges_without_backend_use(self, vault: Path):
        result = capture_step_clips(
            _video(),
            _ranges(),
            assets_subfolder="2026-07-27-1200 Long Talk",
            dry_run=True,
            backend=_ExplodingBackend(),
            cache=_NO_CACHE,
            vault_root=vault,
        )
        assert result.ranges == _ranges()
        assert result.outcomes == []

    def test_empty_ranges_short_circuit(self, vault: Path):
        result = capture_step_clips(
            _video(),
            [],
            assets_subfolder="f",
            backend=_FakeBackend(),
            cache=_NO_CACHE,
            vault_root=vault,
        )
        assert result.error == "no_ranges"

    def test_names_are_range_positional_under_subfolder(self, vault: Path):
        backend = _FakeBackend()
        result = capture_step_clips(
            _video(),
            _ranges(),
            assets_subfolder="2026-07-27-1200 Long Talk",
            backend=backend,
            cache=_NO_CACHE,
            vault_root=vault,
        )
        assert result.error is None
        assert result.video_downloaded is True
        names = [o.image_path.name for o in result.outcomes if o.image_path]
        assert names == ["pyt_vid001_h00.webp", "pyt_vid001_h01.webp", "pyt_vid001_h02.webp"]
        parent = result.outcomes[0].image_path.parent  # type: ignore[union-attr]
        assert parent.name == "2026-07-27-1200 Long Talk"
        assert "pipeline-youtube" in str(parent.parent)
        # Extraction was centered: the 3rd range center is (7500+9000)/2.
        third_call = backend.ffmpeg_calls[2]
        assert third_call[0] == "-ss"
        assert float(third_call[1]) == pytest.approx(8250 - 3.5 / 2.0, abs=0.1)

    def test_single_range_failure_is_isolated(self, vault: Path):
        backend = _FakeBackend(fail_call_indices={1})
        result = capture_step_clips(
            _video(),
            _ranges(),
            assets_subfolder="f",
            backend=backend,
            cache=_NO_CACHE,
            vault_root=vault,
        )
        assert result.success_count == 2
        assert result.failure_count == 1
        assert result.outcomes[1].image_path is None
        assert result.outcomes[1].error is not None
        assert "ffmpeg_exit_1" in result.outcomes[1].error
        # Index alignment survives the failure: the 3rd range keeps h02.
        assert result.outcomes[2].image_path is not None
        assert result.outcomes[2].image_path.name == "pyt_vid001_h02.webp"

    def test_download_failure_degrades_without_raising(self, vault: Path):
        result = capture_step_clips(
            _video(),
            _ranges(),
            assets_subfolder="f",
            backend=_FailingDownloadBackend(),
            cache=_NO_CACHE,
            vault_root=vault,
        )
        assert result.error is not None
        assert result.error.startswith("download_failed")
        assert result.outcomes == []

    def test_cache_hit_skips_download(self, vault: Path, tmp_path: Path):
        cache_root = tmp_path / "cache"
        cache = Cache(cache_root, enabled=True)
        seed = tmp_path / "seed.mp4"
        seed.write_bytes(b"mp4")
        cache.put_video("vid001", "480", seed)

        backend = _FakeBackend()
        result = capture_step_clips(
            _video(),
            _ranges(),
            assets_subfolder="f",
            backend=backend,
            cache=cache,
            vault_root=vault,
        )
        assert backend.download_calls == []
        assert result.video_downloaded is False
        assert result.success_count == 3
        # The cached working copy must survive for the next run.
        assert cache.get_video("vid001", "480") is not None


def test_capture_result_reused_type() -> None:
    """capture_step_clips returns the existing CaptureResult contract."""
    assert CaptureResult(ranges=[]).success_count == 0
