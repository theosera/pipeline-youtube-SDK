"""Tmp ownership across overlapping processes, threads and capture modes (SDK #168; ported from upstream #205)."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.services.cache import Cache
from pipeline_youtube.stages import capture
from pipeline_youtube.stages.capture_backend import DockerCaptureBackend, HostCaptureBackend


@pytest.fixture(autouse=True)
def isolated_project(tmp_path: Path, monkeypatch):
    # Exercise the real path builder without writing to the checkout's tmp.
    monkeypatch.setattr(capture, "__file__", str(tmp_path / "pipeline_youtube/stages/capture.py"))


@pytest.fixture
def video() -> VideoMeta:
    return VideoMeta(
        video_id="abc123abc12",
        title="test",
        url="https://www.youtube.com/watch?v=abc123abc12",
        duration=60,
        channel="test",
        upload_date=None,
        playlist_title=None,
    )


def _path_for(video: VideoMeta, *, pid: int = 100, tid: int = 200) -> Path:
    # Keep identity patches out of executor/backend internals.
    with patch("os.getpid", return_value=pid), patch("threading.get_ident", return_value=tid):
        return capture._tmp_video_path(video)


def test_filename_contains_process_and_thread_ids(video: VideoMeta, tmp_path: Path):
    path = _path_for(video, pid=1234, tid=5678)
    assert path == tmp_path / "tmp/abc123abc12-1234-5678.mp4"


def test_different_processes_get_different_destinations(video: VideoMeta):
    assert _path_for(video, pid=100) != _path_for(video, pid=101)


def test_different_threads_get_different_destinations(video: VideoMeta):
    assert _path_for(video, tid=200) != _path_for(video, tid=201)


def test_same_thread_reuses_same_path(video: VideoMeta):
    assert capture._tmp_video_path(video) == capture._tmp_video_path(video)


def test_live_threads_get_different_destinations(video: VideoMeta):
    barrier = threading.Barrier(2, timeout=5)

    def work() -> Path:
        path = capture._tmp_video_path(video)
        barrier.wait()  # Both workers must still be alive (thread IDs can be reused).
        return path

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(work) for _ in range(2)]
        paths = [future.result(timeout=10) for future in futures]
    assert paths[0] != paths[1]


@pytest.mark.parametrize("other_identity", [{"pid": 101}, {"tid": 201}], ids=["process", "thread"])
def test_download_unlink_preserves_other_worker_files(video: VideoMeta, other_identity):
    dest = _path_for(video)
    other = _path_for(video, **other_identity)
    for suffix in (".mp4", ".mkv", ".webm"):
        dest.with_suffix(suffix).write_bytes(b"old own download")
        other.with_suffix(suffix).write_bytes(b"other worker")

    def download(_urls):
        # Real _download_video unlinks the MP4; the real host backend clears
        # same-stem alternative containers before the fake network boundary.
        for suffix in (".mp4", ".mkv", ".webm"):
            assert not dest.with_suffix(suffix).exists()
            assert other.with_suffix(suffix).read_bytes() == b"other worker"
        dest.write_bytes(b"new own download")

    with patch("yt_dlp.YoutubeDL") as ydl:
        ydl.return_value.__enter__.return_value.download.side_effect = download
        capture._download_video(video.watch_url, dest, backend=HostCaptureBackend())

    assert dest.read_bytes() == b"new own download"
    for suffix in (".mp4", ".mkv", ".webm"):
        assert other.with_suffix(suffix).read_bytes() == b"other worker"


@pytest.mark.parametrize("mode", ["stage03", "hands_on"])
@pytest.mark.parametrize("other_identity", [{"pid": 101}, {"tid": 201}], ids=["process", "thread"])
def test_capture_cleanup_preserves_other_worker_video(
    video: VideoMeta, tmp_path: Path, monkeypatch, mode: str, other_identity
):
    other = _path_for(video, **other_identity)
    other.write_bytes(b"other worker")
    own = _path_for(video)
    (tmp_path / ".obsidian").mkdir()
    summary = tmp_path / "02.md"
    summary.write_text("### [00:00 ~ 00:05] test\n", encoding="utf-8")
    note = tmp_path / "03.md"
    note.write_text("---\n---\n", encoding="utf-8")

    def download(_url: str, dest: Path, **_kwargs):
        assert dest == own
        dest.write_bytes(b"own download")

    def extract(source: Path, output: Path, **_kwargs):
        assert source == own
        assert source.read_bytes() == b"own download"
        output.write_bytes(b"RIFF" + (8).to_bytes(4, "little") + b"WEBPVP8L")

    backend = Mock()
    backend.download_video.side_effect = download
    monkeypatch.setattr(capture, "_dispatch_extractor", lambda _strategy: extract)
    monkeypatch.setattr(
        capture, "_resolve_capture_format", lambda *_args: capture._FormatChoice("webp", "direct")
    )
    with patch("os.getpid", return_value=100), patch("threading.get_ident", return_value=200):
        if mode == "stage03":
            result = capture.run_stage_capture(
                video,
                summary,
                note,
                backend=backend,
                vault_root=tmp_path,
                cache=Cache(None, enabled=False),
            )
        else:
            result = capture.capture_step_clips(
                video,
                [capture.SummaryRange(0, 5, "test")],
                assets_subfolder="hands-on",
                backend=backend,
                cache=Cache(None, enabled=False),
                vault_root=tmp_path,
            )

    assert result.error is None
    assert result.success_count == 1
    assert not own.exists()
    assert other.read_bytes() == b"other worker"


def test_prefetch_uses_callers_path_before_and_after_worker_finishes(video: VideoMeta):
    expected = capture._tmp_video_path(video)
    caller_tid = threading.get_ident()
    downloads: list[tuple[int, Path]] = []

    def download(_url: str, dest: Path, *_args, **_kwargs):
        downloads.append((threading.get_ident(), dest))
        dest.write_bytes(b"prefetched")

    with patch.object(capture, "_download_video", side_effect=download):
        handle = capture.prefetch_video_download(video)
        assert handle.path == expected == capture._tmp_video_path(video)
        assert handle.wait(timeout=5) is None

    assert downloads == [(downloads[0][0], expected)]
    assert downloads[0][0] != caller_tid
    assert capture._tmp_video_path(video) == expected
    assert expected.read_bytes() == b"prefetched"


def test_generated_path_translates_to_docker_work_mount(video: VideoMeta, tmp_path: Path):
    path = capture._tmp_video_path(video)
    backend = DockerCaptureBackend(tmp_dir=tmp_path / "tmp", assets_dir=tmp_path / "assets")
    assert path.parent == backend.tmp_dir
    assert backend._host_to_container(path) == f"/work/{path.name}"

    def download(*_args: object, **_kwargs: object) -> None:
        path.write_bytes(b"downloaded")

    with patch(
        "pipeline_youtube.stages.capture_backend.subprocess.run", side_effect=download
    ) as run:
        backend.download_video(video.watch_url, path, resolution="480")
    command = run.call_args.args[0]
    assert command[command.index("-o") + 1] == f"/work/{path.stem}.%(ext)s"

    output = tmp_path / "assets/clip.webp"
    with patch("pipeline_youtube.stages.capture_backend.subprocess.run") as run:
        backend.ffmpeg(["-i", str(path), str(output)], timeout=10)
    command = run.call_args.args[0]
    assert command[command.index("-i") + 1] == f"/work/{path.name}"
    assert command[-1] == "/assets/clip.webp"


def test_sweep_removes_stale_generated_path(video: VideoMeta):
    path = capture._tmp_video_path(video)
    path.write_bytes(b"stale download")
    past = time.time() - 48 * 3600
    os.utime(path, (past, past))
    assert capture.sweep_stale_tmp(path.parent) == 1
    assert not path.exists()


@pytest.fixture
def capture_notes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    (tmp_path / ".obsidian").mkdir()
    summary = tmp_path / "02.md"
    summary.write_text("### [00:00 ~ 00:05] test\n", encoding="utf-8")
    note = tmp_path / "03.md"
    note.write_text("---\n---\n", encoding="utf-8")
    monkeypatch.setattr(
        capture, "_resolve_capture_format", lambda *_args: capture._FormatChoice("webp", "direct")
    )
    return summary, note


@pytest.mark.parametrize("mode", ["stage03", "hands_on"])
@pytest.mark.parametrize("capture_fails", [False, True], ids=["success", "capture_failure"])
def test_generated_download_is_cached_and_hit_survives_cleanup(
    video: VideoMeta,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capture_notes: tuple[Path, Path],
    mode: str,
    capture_fails: bool,
) -> None:
    cache = Cache(tmp_path / "cache")
    own = capture._tmp_video_path(video)
    other = _path_for(video)
    other.write_bytes(b"other worker")
    extracted: list[Path] = []

    def download(_url: str, dest: Path, **_kwargs: object) -> None:
        assert dest == own
        dest.write_bytes(b"downloaded video")

    def extract(source: Path, output: Path, **_kwargs: object) -> None:
        extracted.append(source)
        assert source.read_bytes() == b"downloaded video"
        if capture_fails:
            raise RuntimeError("synthetic capture failure")
        output.write_bytes(b"RIFF" + (8).to_bytes(4, "little") + b"WEBPVP8L")

    backend = Mock()
    backend.download_video.side_effect = download
    monkeypatch.setattr(capture, "_dispatch_extractor", lambda _strategy: extract)

    def run_capture() -> capture.CaptureResult:
        if mode == "stage03":
            return capture.run_stage_capture(
                video,
                *capture_notes,
                cache=cache,
                vault_root=tmp_path,
                resolution="720",
                backend=backend,
            )
        return capture.capture_step_clips(
            video,
            [capture.SummaryRange(0, 5, "test")],
            assets_subfolder="hands-on",
            cache=cache,
            vault_root=tmp_path,
            resolution="720",
            backend=backend,
        )

    with (
        patch.object(cache, "get_video", wraps=cache.get_video) as get_video,
        patch.object(cache, "put_video", wraps=cache.put_video) as put_video,
    ):
        miss = run_capture()
        get_video.assert_called_once_with(video.video_id, "720")
        put_video.assert_called_once_with(video.video_id, "720", own)
        assert miss.video_downloaded
        assert not own.exists()
        cached = cache.get_video(video.video_id, "720")
        assert cached == tmp_path / "cache/video" / video.video_id / "720"
        assert cached.read_bytes() == b"downloaded video"
        get_video.reset_mock()

        # A new caller identity still borrows the same persistent cache entry.
        with patch("os.getpid", return_value=100), patch("threading.get_ident", return_value=200):
            hit = run_capture()
        get_video.assert_called_once_with(video.video_id, "720")
        put_video.assert_called_once_with(video.video_id, "720", own)

    assert not hit.video_downloaded
    assert backend.download_video.call_count == 1
    assert extracted == [own, cached]
    assert cached.read_bytes() == b"downloaded video"
    assert other.read_bytes() == b"other worker"
    assert cache.get_video(video.video_id, "480") is None
    for result in (miss, hit):
        assert result.error is None
        assert result.success_count == (0 if capture_fails else 1)
        if capture_fails:
            assert result.outcomes[0].error == "RuntimeError: synthetic capture failure"


@pytest.mark.parametrize("delete_video", [True, False], ids=["owned_prefetch", "borrowed_video"])
@pytest.mark.parametrize("capture_fails", [False, True], ids=["success", "capture_failure"])
def test_generated_prefetch_preserves_cache_and_borrowed_video(
    video: VideoMeta,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capture_notes: tuple[Path, Path],
    delete_video: bool,
    capture_fails: bool,
) -> None:
    cache = Cache(tmp_path / "cache")
    source = capture._tmp_video_path(video)
    source.write_bytes(b"prefetched video")
    backend = Mock()

    def extract(video_path: Path, output: Path, **_kwargs: object) -> None:
        assert video_path == source
        assert video_path.read_bytes() == b"prefetched video"
        if capture_fails:
            raise RuntimeError("synthetic capture failure")
        output.write_bytes(b"RIFF" + (8).to_bytes(4, "little") + b"WEBPVP8L")

    monkeypatch.setattr(capture, "_dispatch_extractor", lambda _strategy: extract)
    with (
        patch.object(cache, "get_video", wraps=cache.get_video) as get_video,
        patch.object(cache, "put_video", wraps=cache.put_video) as put_video,
    ):
        result = capture.run_stage_capture(
            video,
            *capture_notes,
            prefetched_video_path=source,
            delete_video=delete_video,
            allow_download=delete_video,
            cache=cache,
            vault_root=tmp_path,
            backend=backend,
        )
        get_video.assert_not_called()
        if delete_video:
            put_video.assert_called_once_with(video.video_id, "480", source)
        else:
            put_video.assert_not_called()

    backend.download_video.assert_not_called()
    assert not result.video_downloaded
    assert result.error is None
    assert result.success_count == (0 if capture_fails else 1)
    if capture_fails:
        assert result.outcomes[0].error == "RuntimeError: synthetic capture failure"
    cached = cache.get_video(video.video_id, "480")
    if delete_video:
        assert not source.exists()
        assert cached is not None
        assert cached.read_bytes() == b"prefetched video"
    else:
        assert source.read_bytes() == b"prefetched video"
        assert cached is None
