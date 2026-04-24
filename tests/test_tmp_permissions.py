"""Tests for #7: tmp directory + downloaded file have owner-only permissions."""

from __future__ import annotations

import stat
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline_youtube.stages.capture import _tmp_video_path
from pipeline_youtube.stages.capture_backend import HostCaptureBackend


def _video(video_id: str = "abc1234567"):
    from pipeline_youtube.playlist import VideoMeta

    return VideoMeta(
        video_id=video_id,
        title="t",
        url=f"https://www.youtube.com/watch?v={video_id}",
        duration=60,
        channel="c",
        upload_date=None,
        playlist_title=None,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX perms")
class TestTmpDirPermissions:
    def test_tmp_dir_is_700(self):
        path = _tmp_video_path(_video())
        mode = stat.S_IMODE(path.parent.stat().st_mode)
        assert mode == 0o700, f"expected 0o700, got {oct(mode)}"

    def test_host_backend_chmods_download_to_600(self, tmp_path: Path):
        """HostCaptureBackend must tighten downloaded file perms to 0o600."""
        dest = tmp_path / "v.mp4"

        def fake_download(urls):
            dest.write_bytes(b"x")  # mimic yt-dlp writing the file

        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_ydl.return_value.__enter__.return_value.download = fake_download
            HostCaptureBackend().download_video(
                "https://youtube.com/watch?v=x", dest, resolution="480"
            )

        mode = stat.S_IMODE(dest.stat().st_mode)
        assert mode == 0o600, f"expected 0o600, got {oct(mode)}"
