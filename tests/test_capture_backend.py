"""Tests for CaptureBackend protocol implementations (host + docker).

Real Docker invocations are out of scope — CI can't run privileged
containers reliably — so the Docker backend tests mock `subprocess.run`
and assert the command shape: hardening flags, bind mounts, path
translation.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline_youtube.stages.capture_backend import (
    DEFAULT_DOCKER_IMAGE,
    CaptureBackendError,
    DockerBackendNotReady,
    DockerCaptureBackend,
    HostCaptureBackend,
    _adopt_ytdlp_output,
    _clear_ytdlp_outputs,
    _host_ffmpeg_encoders,
)

# =====================================================
# _adopt_ytdlp_output (extension fallback after yt-dlp)
# =====================================================


class TestAdoptYtdlpOutput:
    def test_noop_when_dest_exists(self, tmp_path: Path):
        dest = tmp_path / "abc123abc12.mp4"
        dest.write_bytes(b"dest")
        leftover = tmp_path / "abc123abc12.webm"
        leftover.write_bytes(b"webm")
        _adopt_ytdlp_output(dest)
        assert dest.read_bytes() == b"dest"
        assert leftover.exists()

    def test_adopts_same_stem_container(self, tmp_path: Path):
        dest = tmp_path / "abc123abc12.mp4"
        webm = tmp_path / "abc123abc12.webm"
        webm.write_bytes(b"real-download")
        _adopt_ytdlp_output(dest)
        assert dest.read_bytes() == b"real-download"
        assert not webm.exists()

    def test_ignores_dash_fragment_leftover(self, tmp_path: Path):
        """Interrupted DASH mux leaves `{id}.f137.mp4`; glob `{id}.*` matches it.

        Lexicographic sort picked the fragment (`.f` < `.w`) and ffmpeg then
        captured from a partial file. The real mux is `{id}.webm`.
        """
        dest = tmp_path / "abc123abc12.mp4"
        fragment = tmp_path / "abc123abc12.f137.mp4"
        fragment.write_bytes(b"partial-dash")
        webm = tmp_path / "abc123abc12.webm"
        webm.write_bytes(b"real-download")
        _adopt_ytdlp_output(dest)
        assert dest.read_bytes() == b"real-download"
        assert fragment.exists()
        assert fragment.read_bytes() == b"partial-dash"

    def test_ignores_part_temp(self, tmp_path: Path):
        dest = tmp_path / "abc123abc12.mp4"
        part = tmp_path / "abc123abc12.mp4.part"
        part.write_bytes(b"incomplete")
        mkv = tmp_path / "abc123abc12.mkv"
        mkv.write_bytes(b"real-download")
        _adopt_ytdlp_output(dest)
        assert dest.read_bytes() == b"real-download"
        assert part.exists()

    def test_fragment_only_is_not_adopted(self, tmp_path: Path):
        dest = tmp_path / "abc123abc12.mp4"
        fragment = tmp_path / "abc123abc12.f137.mp4"
        fragment.write_bytes(b"partial-dash")
        with pytest.raises(FileNotFoundError, match="produced no file"):
            _adopt_ytdlp_output(dest)
        assert fragment.exists()
        assert not dest.exists()

    def test_newer_same_stem_container_wins(self, tmp_path: Path):
        dest = tmp_path / "abc123abc12.mp4"
        old = tmp_path / "abc123abc12.mkv"
        old.write_bytes(b"old-leftover")
        past = time.time() - 3600
        os.utime(old, (past, past))
        new = tmp_path / "abc123abc12.webm"
        new.write_bytes(b"this-download")
        _adopt_ytdlp_output(dest)
        assert dest.read_bytes() == b"this-download"
        assert old.exists()

    def test_newer_fragment_is_still_ignored(self, tmp_path: Path):
        """mtime alone must not decide: a fragment newer than the mux still loses."""
        dest = tmp_path / "abc123abc12.mp4"
        webm = tmp_path / "abc123abc12.webm"
        webm.write_bytes(b"real-download")
        past = time.time() - 3600
        os.utime(webm, (past, past))
        fragment = tmp_path / "abc123abc12.f137.mp4"
        fragment.write_bytes(b"partial-dash")
        _adopt_ytdlp_output(dest)
        assert dest.read_bytes() == b"real-download"
        assert fragment.exists()

    def test_ignores_same_stem_non_media(self, tmp_path: Path):
        """A same-stem thumbnail (`{id}.webp`) is not a video container."""
        dest = tmp_path / "abc123abc12.mp4"
        mkv = tmp_path / "abc123abc12.mkv"
        mkv.write_bytes(b"real-download")
        past = time.time() - 3600
        os.utime(mkv, (past, past))
        thumb = tmp_path / "abc123abc12.webp"
        thumb.write_bytes(b"thumbnail")
        _adopt_ytdlp_output(dest)
        assert dest.read_bytes() == b"real-download"
        assert thumb.exists()


class TestDownloadVideoSkipsLeftoverFragment:
    """Both backends adopt yt-dlp's output through `_adopt_ytdlp_output`.

    The lexicographic pick lived in each `download_video`, so the helper tests
    alone do not show that the call sites stopped choosing the fragment.
    """

    def test_host(self, tmp_path: Path):
        dest = tmp_path / "abc123abc12.mp4"
        fragment = tmp_path / "abc123abc12.f137.mp4"
        fragment.write_bytes(b"partial-dash")

        class FakeYDL:
            def __init__(self, opts):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def download(self, urls):
                (tmp_path / "abc123abc12.webm").write_bytes(b"real-download")

        with patch.dict("sys.modules", {"yt_dlp": MagicMock(YoutubeDL=FakeYDL)}):
            HostCaptureBackend().download_video(
                "https://www.youtube.com/watch?v=abc123abc12", dest, resolution="480"
            )
        assert dest.read_bytes() == b"real-download"
        assert fragment.read_bytes() == b"partial-dash"

    def test_docker(self, docker_backend):
        dest = docker_backend.tmp_dir / "abc123abc12.mp4"
        fragment = docker_backend.tmp_dir / "abc123abc12.f137.mp4"
        fragment.write_bytes(b"partial-dash")

        def fake_run(*args, **kwargs):
            (docker_backend.tmp_dir / "abc123abc12.webm").write_bytes(b"real-download")
            return MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=fake_run):
            docker_backend.download_video(
                "https://www.youtube.com/watch?v=abc123abc12", dest, resolution="480"
            )
        assert dest.read_bytes() == b"real-download"
        assert fragment.read_bytes() == b"partial-dash"


class TestClearYtdlpOutputs:
    def test_removes_complete_containers_only(self, tmp_path: Path):
        stem = "abc123abc12"
        containers = [tmp_path / f"{stem}{ext}" for ext in (".mp4", ".mkv", ".webm", ".m4v")]
        kept = [
            tmp_path / f"{stem}.f137.mp4",  # DASH fragment: yt-dlp resumes from it
            tmp_path / f"{stem}.mp4.part",
            tmp_path / f"{stem}.webp",
            tmp_path / "zzz999zzz99.mkv",  # another video
        ]
        for f in containers + kept:
            f.write_bytes(b"x")
        _clear_ytdlp_outputs(tmp_path / f"{stem}.mp4")
        assert [f for f in containers if f.exists()] == []
        assert all(f.exists() for f in kept)


def _future_mtime(path: Path) -> None:
    ahead = time.time() + 3600
    os.utime(path, (ahead, ahead))


class TestDownloadVideoClearsStaleContainers:
    """A leftover complete container must not beat this run's output.

    yt-dlp may reuse an old complete file instead of writing a new one, so a
    leftover can carry the newest mtime. Clearing before the download leaves
    only this run's container for `_adopt_ytdlp_output` to pick.
    """

    def test_host(self, tmp_path: Path):
        dest = tmp_path / "abc123abc12.mp4"
        stale = tmp_path / "abc123abc12.mkv"
        stale.write_bytes(b"stale-download")
        _future_mtime(stale)

        class FakeYDL:
            def __init__(self, opts):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def download(self, urls):
                (tmp_path / "abc123abc12.webm").write_bytes(b"this-download")

        with patch.dict("sys.modules", {"yt_dlp": MagicMock(YoutubeDL=FakeYDL)}):
            HostCaptureBackend().download_video(
                "https://www.youtube.com/watch?v=abc123abc12", dest, resolution="480"
            )
        assert dest.read_bytes() == b"this-download"
        assert not stale.exists()

    def test_docker(self, docker_backend):
        dest = docker_backend.tmp_dir / "abc123abc12.mp4"
        stale = docker_backend.tmp_dir / "abc123abc12.mkv"
        stale.write_bytes(b"stale-download")
        _future_mtime(stale)

        def fake_run(*args, **kwargs):
            (docker_backend.tmp_dir / "abc123abc12.webm").write_bytes(b"this-download")
            return MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=fake_run):
            docker_backend.download_video(
                "https://www.youtube.com/watch?v=abc123abc12", dest, resolution="480"
            )
        assert dest.read_bytes() == b"this-download"
        assert not stale.exists()


# =====================================================
# HostCaptureBackend
# =====================================================


class TestHostBackend:
    def test_name(self):
        assert HostCaptureBackend().name == "host"

    def test_ffmpeg_shells_out_with_timeout(self):
        with patch("subprocess.run") as run:
            HostCaptureBackend().ffmpeg(["-version"], timeout=5)
            args, kwargs = run.call_args
            cmd = args[0]
            assert cmd[0] == "ffmpeg"
            assert cmd[1] == "-version"
            assert kwargs["timeout"] == 5
            assert kwargs["check"] is True

    def test_gif2webp_shells_out(self):
        with patch("subprocess.run") as run:
            HostCaptureBackend().gif2webp(["-version"], timeout=5)
            cmd = run.call_args.args[0]
            assert cmd[0] == "gif2webp"

    def test_ffmpeg_encoders_caches(self):
        """Calling twice should hit subprocess only once (lru_cache)."""
        _host_ffmpeg_encoders.cache_clear()
        fake = MagicMock(returncode=0, stdout="V..... libwebp webp\nV..... gif gif\n")
        with patch("subprocess.run", return_value=fake) as run:
            a = _host_ffmpeg_encoders()
            b = _host_ffmpeg_encoders()
            assert a == b
            assert "libwebp" in a
            assert run.call_count == 1
        _host_ffmpeg_encoders.cache_clear()


# =====================================================
# DockerCaptureBackend
# =====================================================


@pytest.fixture
def docker_backend(tmp_path: Path):
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    return DockerCaptureBackend(tmp_dir=tmp_dir, assets_dir=assets_dir)


class TestDockerBackendPreflight:
    def test_missing_docker_cli(self, docker_backend):
        with (
            patch("shutil.which", return_value=None),
            pytest.raises(DockerBackendNotReady, match="docker CLI not found"),
        ):
            docker_backend.preflight()

    def test_missing_image(self, docker_backend):
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(
                    returncode=1, cmd="docker image inspect", stderr=b"No such image"
                ),
            ),
            pytest.raises(DockerBackendNotReady, match="image .* not found"),
        ):
            docker_backend.preflight()

    def test_daemon_unreachable(self, docker_backend):
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="docker", timeout=15),
            ),
            pytest.raises(DockerBackendNotReady, match="daemon unreachable"),
        ):
            docker_backend.preflight()

    def test_preflight_happy_path(self, docker_backend):
        fake = MagicMock(returncode=0)
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch("subprocess.run", return_value=fake),
        ):
            docker_backend.preflight()  # must not raise


class TestDockerBackendCommandShape:
    def _extract_cmd(self, run_mock):
        return run_mock.call_args.args[0]

    def test_base_args_contain_hardening_flags(self, docker_backend):
        with patch("subprocess.run") as run:
            docker_backend.ffmpeg(["-version"], timeout=5)
            cmd = self._extract_cmd(run)

        assert "docker" in cmd[0]
        assert "run" in cmd
        assert "--rm" in cmd
        assert "--read-only" in cmd
        assert "--cap-drop=ALL" in cmd
        assert "--security-opt=no-new-privileges:true" in cmd
        # --user is derived from caller UID/GID (see _caller_uid_gid).
        assert any(a.startswith("--user=") for a in cmd)
        # tmpfs is required for ffmpeg scratch space
        assert any("--tmpfs=/tmp" in a for a in cmd)

    def test_user_flag_uses_caller_uid_gid(self, docker_backend):
        """--user must mirror the host process UID/GID, not a hard-coded 1000."""
        with (
            patch(
                "pipeline_youtube.stages.capture_backend.os.getuid",
                return_value=4242,
                create=True,
            ),
            patch(
                "pipeline_youtube.stages.capture_backend.os.getgid",
                return_value=5353,
                create=True,
            ),
            patch("subprocess.run") as run,
        ):
            docker_backend.ffmpeg(["-version"], timeout=5)
            cmd = self._extract_cmd(run)
        assert "--user=4242:5353" in cmd

    def test_ffmpeg_network_is_none(self, docker_backend):
        with patch("subprocess.run") as run:
            docker_backend.ffmpeg(["-version"], timeout=5)
            cmd = self._extract_cmd(run)
        assert "--network=none" in cmd
        assert "--network=bridge" not in cmd

    def test_yt_dlp_network_is_bridge(self, docker_backend):
        """yt-dlp needs HTTPS to youtube.com, ffmpeg never does."""
        dest = docker_backend.tmp_dir / "vid.mp4"

        def fake_run(*args, **kwargs):
            # Simulate yt-dlp writing the output so the post-call
            # existence check doesn't raise.
            dest.write_bytes(b"\x00")
            return MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=fake_run) as run:
            docker_backend.download_video(
                "https://www.youtube.com/watch?v=x", dest, resolution="480"
            )
            cmd = self._extract_cmd(run)
        assert "--network=bridge" in cmd
        assert "yt-dlp" in cmd

    def test_bind_mounts_point_at_configured_dirs(self, docker_backend):
        with patch("subprocess.run") as run:
            docker_backend.ffmpeg(["-version"], timeout=5)
            cmd = self._extract_cmd(run)
        mounts = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-v"]
        tmp = str(docker_backend.tmp_dir.resolve())
        assets = str(docker_backend.assets_dir.resolve())
        assert any(m.startswith(tmp + ":") for m in mounts)
        assert any(m.startswith(assets + ":") for m in mounts)

    def test_image_is_last_before_args(self, docker_backend):
        with patch("subprocess.run") as run:
            docker_backend.ffmpeg(["-ss", "1.0", "-y", "/tmp/out.webp"], timeout=5)
            cmd = self._extract_cmd(run)
        # Image name must appear before the binary name
        img_idx = cmd.index(DEFAULT_DOCKER_IMAGE)
        assert cmd[img_idx + 1] == "ffmpeg"


class TestDockerBackendPathTranslation:
    def test_tmp_path_translated_to_work_mount(self, docker_backend):
        host_path = docker_backend.tmp_dir / "foo.mp4"
        assert docker_backend._host_to_container(host_path) == "/work/foo.mp4"

    def test_assets_path_translated(self, docker_backend):
        host_path = docker_backend.assets_dir / "pyt_x_00.webp"
        assert docker_backend._host_to_container(host_path) == "/assets/pyt_x_00.webp"

    def test_unmounted_path_rejected(self, docker_backend, tmp_path):
        stray = tmp_path / "outside" / "oops.txt"
        stray.parent.mkdir()
        stray.write_text("x")
        with pytest.raises(CaptureBackendError, match="not under tmp"):
            docker_backend._host_to_container(stray)

    def test_ffmpeg_args_get_paths_rewritten(self, docker_backend):
        vid = docker_backend.tmp_dir / "v.mp4"
        vid.write_bytes(b"x")
        out = docker_backend.assets_dir / "pyt_x_00.webp"
        with patch("subprocess.run") as run:
            docker_backend.ffmpeg(["-ss", "1.0", "-i", str(vid), "-y", str(out)], timeout=5)
            cmd = run.call_args.args[0]
        # Paths must have been translated
        assert "/work/v.mp4" in cmd
        assert "/assets/pyt_x_00.webp" in cmd
        assert str(vid) not in cmd
        assert str(out) not in cmd

    def test_flag_args_pass_through(self, docker_backend):
        with patch("subprocess.run") as run:
            docker_backend.ffmpeg(["-loop", "0", "-an", "-y"], timeout=5)
            cmd = run.call_args.args[0]
        for flag in ("-loop", "0", "-an", "-y"):
            assert flag in cmd

    def test_relative_paths_not_translated(self, docker_backend):
        """Only absolute paths should be candidates for rewriting.

        Relative paths like ``./foo.mp4`` must pass through unchanged;
        production never sends them (capture.py always builds absolute
        paths) but the heuristic should not silently rewrite anyway.
        """
        with patch("subprocess.run") as run:
            docker_backend.ffmpeg(
                ["-i", "./foo.mp4", "-vf", "fps=5,scale=-2:480", "-y", "out.gif"],
                timeout=5,
            )
            cmd = run.call_args.args[0]
        assert "./foo.mp4" in cmd
        assert "out.gif" in cmd
        assert "fps=5,scale=-2:480" in cmd


class TestDockerBackendErrorPropagation:
    def test_yt_dlp_failure_wrapped(self, docker_backend):
        dest = docker_backend.tmp_dir / "v.mp4"
        with (
            patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(
                    returncode=42,
                    cmd="docker run ... yt-dlp ...",
                    stderr=b"ERROR: Video unavailable",
                ),
            ),
            pytest.raises(CaptureBackendError, match="yt-dlp .* exited 42"),
        ):
            docker_backend.download_video(
                "https://www.youtube.com/watch?v=x", dest, resolution="480"
            )

    def test_has_gif2webp_always_true(self, docker_backend):
        """Image always ships gif2webp, so no runtime probe needed."""
        assert docker_backend.has_gif2webp() is True
