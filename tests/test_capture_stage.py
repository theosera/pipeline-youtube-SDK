"""Tests for stage 03 (capture) with yt-dlp and ffmpeg mocked."""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pipeline_youtube import config
from pipeline_youtube.pipeline import create_placeholder_notes
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.stages import capture as capture_stage
from pipeline_youtube.stages.capture import (
    CaptureResult,
    _capture_image_name,
    _FormatChoice,
    parse_summary_ranges,
    run_stage_capture,
)

# =====================================================
# Pure-function tests (no filesystem / subprocess)
# =====================================================


SAMPLE_SUMMARY = """## 全体サマリ
動画全体の要約。

## 要点タイムライン

### [00:00 ~ 01:03] ハーネスエンジニアリングとは
AI の能力を最大限引き出す環境整備。

### [01:03 ~ 02:50] 問題①コンテキストフア
長時間タスクで文脈が埋まると AI が焦る。

### [02:50 ~ 03:26] 解決策①コンテキストリセット
新エージェントにハンドオフする。

### [12:50 ~ 15:10] まとめと今後の展望
モデル特性に合わせた継続最適化。
"""


class TestParseSummaryRanges:
    def test_parses_all_h3_ranges(self):
        ranges = parse_summary_ranges(SAMPLE_SUMMARY)
        assert len(ranges) == 4
        assert ranges[0].start_sec == 0
        assert ranges[0].end_sec == 63
        assert ranges[0].heading == "ハーネスエンジニアリングとは"
        assert ranges[3].start_sec == 770  # 12:50
        assert ranges[3].end_sec == 910  # 15:10

    def test_center_and_mmss(self):
        rng = parse_summary_ranges(SAMPLE_SUMMARY)[1]
        assert rng.start_mmss == "01:03"
        assert rng.end_mmss == "02:50"
        assert rng.center_sec == (63 + 170) / 2.0

    def test_tolerates_fullwidth_tilde(self):
        md = "### [00:10 〜 01:20] タイトル\n\n本文\n"
        ranges = parse_summary_ranges(md)
        assert len(ranges) == 1
        assert ranges[0].start_sec == 10
        assert ranges[0].end_sec == 80

    def test_tolerates_wave_dash(self):
        md = "### [00:10 ~ 01:20] タイトル\n\n本文\n"
        ranges = parse_summary_ranges(md)
        assert len(ranges) == 1

    def test_rejects_end_before_start(self):
        md = "### [05:00 ~ 03:00] bad range\n\n本文\n"
        assert parse_summary_ranges(md) == []

    def test_empty_input(self):
        assert parse_summary_ranges("") == []

    def test_ignores_non_range_h3(self):
        md = "### プロローグ\n本文\n### [00:00 ~ 01:00] 正しいレンジ\n本文\n"
        ranges = parse_summary_ranges(md)
        assert len(ranges) == 1
        assert ranges[0].heading == "正しいレンジ"


class TestCaptureImageName:
    def test_index_zero_zero_padded(self):
        """idx 0 is `pyt_<id>_00.webp` (zero-padded, contiguous from 0)."""
        assert _capture_image_name("abc123", 0) == "pyt_abc123_00.webp"

    def test_index_one(self):
        assert _capture_image_name("abc123", 1) == "pyt_abc123_01.webp"

    def test_index_ten_preserves_padding(self):
        assert _capture_image_name("abc123", 10) == "pyt_abc123_10.webp"

    def test_custom_extension(self):
        assert _capture_image_name("abc123", 0, "gif") == "pyt_abc123_00.gif"

    def test_video_id_with_underscore(self):
        # YouTube video IDs can contain `-` and `_`
        assert _capture_image_name("_h3decBW12Q", 3) == "pyt__h3decBW12Q_03.webp"

    def test_does_not_include_video_title(self):
        """Filename must NOT match `${notename}` — no note title in it."""
        name = _capture_image_name("abc123", 0)
        assert "note" not in name.lower()
        assert name.startswith("pyt_")


# =====================================================
# End-to-end (yt-dlp + ffmpeg mocked)
# =====================================================


@pytest.fixture
def vault(tmp_path: Path):
    config.set_vault_root(tmp_path)
    config.set_dry_run(False)
    yield config.get_vault_root()
    config.reset_vault_root()


def _video():
    return VideoMeta(
        video_id="_h3decBW12Q",
        title="Anthropicが公開したハーネス設計、全部解説します",
        url="https://www.youtube.com/watch?v=_h3decBW12Q",
        duration=945,
        channel="AI Channel",
        upload_date="20260414",
        playlist_title="Harness Engineering",
    )


def _setup_case(vault: Path, summary_md_content: str = SAMPLE_SUMMARY):
    """Create placeholders + write summary md with the given content."""
    video = _video()
    run_time = datetime(2026, 4, 14, 21, 41)
    paths = create_placeholder_notes(video, run_time, dry_run=False)

    summary_path = paths["summary"]
    existing = summary_path.read_text(encoding="utf-8")
    summary_path.write_text(existing + "\n" + summary_md_content, encoding="utf-8")

    return video, paths


def _fake_successful_ffmpeg(*args, **kwargs):
    """Mock ffmpeg that creates the output file."""
    # subprocess.run signature: run(cmd, ...)
    cmd = args[0] if args else kwargs.get("args")
    # Last arg is the output path
    output_path = Path(cmd[-1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"\x52\x49\x46\x46")  # RIFF header stub
    return MagicMock(returncode=0, stdout=b"", stderr=b"")


def _fake_failing_ffmpeg(*args, **kwargs):
    cmd = args[0] if args else kwargs.get("args")
    raise subprocess.CalledProcessError(
        returncode=1,
        cmd=cmd,
        stderr=b"ffmpeg: simulated failure",
    )


class TestRunStageCapture:
    def test_happy_path_creates_webps_and_appends_md(self, vault, monkeypatch):
        video, paths = _setup_case(vault)

        # Mock yt-dlp download to create an empty mp4
        def fake_download(url, dest, resolution="480", *, backend=None):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"\x00\x00\x00\x20ftypmp42")  # mp4 magic

        monkeypatch.setattr(capture_stage, "_download_video", fake_download)
        # Pin format to WebP so test is deterministic regardless of host ffmpeg capabilities
        monkeypatch.setattr(
            capture_stage,
            "_resolve_capture_format",
            lambda _fmt, _backend: _FormatChoice(ext="webp", strategy="direct"),
        )
        monkeypatch.setattr(subprocess, "run", _fake_successful_ffmpeg)
        # Pin format to WebP so test is deterministic regardless of host ffmpeg capabilities
        monkeypatch.setattr(
            capture_stage,
            "_resolve_capture_format",
            lambda _fmt, _backend: _FormatChoice(ext="webp", strategy="direct"),
        )

        result = run_stage_capture(
            video,
            summary_md_path=paths["summary"],
            capture_md_path=paths["capture"],
        )

        assert isinstance(result, CaptureResult)
        assert result.error is None
        assert result.video_downloaded is True
        assert len(result.ranges) == 4
        assert result.success_count == 4
        assert result.failure_count == 0
        assert len(result.image_paths) == 4

        # Verify pipeline-youtube naming: pyt_<id>_00.webp, pyt_<id>_01.webp, ...
        names = [p.name for p in result.image_paths]
        assert names[0] == "pyt__h3decBW12Q_00.webp"
        assert names[1] == "pyt__h3decBW12Q_01.webp"
        assert names[2] == "pyt__h3decBW12Q_02.webp"
        assert names[3] == "pyt__h3decBW12Q_03.webp"

        # All WebPs live in the dedicated pipeline-youtube subfolder
        for p in result.image_paths:
            assert p.exists()
            assert "pipeline-youtube" in p.parent.parts

        # 03_Capture md contains range + embed blocks
        capture_body = paths["capture"].read_text(encoding="utf-8")
        assert "[00:00 ~ 01:03]" in capture_body
        assert "![[pyt__h3decBW12Q_00.webp]]" in capture_body
        assert "[12:50 ~ 15:10]" in capture_body
        assert "![[pyt__h3decBW12Q_03.webp]]" in capture_body

    def test_dry_run_skips_download_and_write(self, vault, monkeypatch):
        video, paths = _setup_case(vault)
        pre = paths["capture"].read_text(encoding="utf-8")

        def fail_download(*a, **kw):
            raise AssertionError("download must not be called in dry_run")

        def fail_ffmpeg(*a, **kw):
            raise AssertionError("ffmpeg must not be called in dry_run")

        monkeypatch.setattr(capture_stage, "_download_video", fail_download)
        monkeypatch.setattr(subprocess, "run", fail_ffmpeg)

        result = run_stage_capture(
            video,
            summary_md_path=paths["summary"],
            capture_md_path=paths["capture"],
            dry_run=True,
        )

        assert result.video_downloaded is False
        assert result.outcomes == []
        assert len(result.ranges) == 4
        assert paths["capture"].read_text(encoding="utf-8") == pre

    def test_no_summary_file(self, vault, monkeypatch):
        video, paths = _setup_case(vault)
        paths["summary"].unlink()

        result = run_stage_capture(
            video,
            summary_md_path=paths["summary"],
            capture_md_path=paths["capture"],
        )
        assert result.error == "summary_md_not_found"
        assert result.ranges == []

    def test_no_ranges_in_summary(self, vault, monkeypatch):
        video, paths = _setup_case(
            vault, summary_md_content="## 全体サマリ\n\n本文のみ、h3無し。\n"
        )

        result = run_stage_capture(
            video,
            summary_md_path=paths["summary"],
            capture_md_path=paths["capture"],
        )
        assert result.error == "no_ranges_parsed"
        assert result.ranges == []

    def test_download_failure_returns_error(self, vault, monkeypatch):
        video, paths = _setup_case(vault)

        def boom_download(*a, **kw):
            raise RuntimeError("network down")

        monkeypatch.setattr(capture_stage, "_download_video", boom_download)

        result = run_stage_capture(
            video,
            summary_md_path=paths["summary"],
            capture_md_path=paths["capture"],
        )
        assert result.error is not None
        assert "download_failed" in result.error
        assert "RuntimeError" in result.error
        assert result.video_downloaded is False

    def test_partial_ffmpeg_failure_numbering_contiguous(self, vault, monkeypatch):
        """Range 1 fails → successful names remain contiguous: _00, _01, _02."""
        video, paths = _setup_case(vault)

        def fake_download(url, dest, resolution="480", *, backend=None):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"stub")

        monkeypatch.setattr(capture_stage, "_download_video", fake_download)
        # Pin format to WebP so test is deterministic regardless of host ffmpeg capabilities
        monkeypatch.setattr(
            capture_stage,
            "_resolve_capture_format",
            lambda _fmt, _backend: _FormatChoice(ext="webp", strategy="direct"),
        )

        call_count = {"n": 0}

        def flaky_ffmpeg(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:  # second range fails
                raise subprocess.CalledProcessError(returncode=1, cmd=args[0], stderr=b"oops")
            return _fake_successful_ffmpeg(*args, **kwargs)

        monkeypatch.setattr(subprocess, "run", flaky_ffmpeg)

        result = run_stage_capture(
            video,
            summary_md_path=paths["summary"],
            capture_md_path=paths["capture"],
        )

        assert result.success_count == 3
        assert result.failure_count == 1
        names = [p.name for p in result.image_paths]
        assert names == [
            "pyt__h3decBW12Q_00.webp",
            "pyt__h3decBW12Q_01.webp",
            "pyt__h3decBW12Q_02.webp",
        ]

        body = paths["capture"].read_text(encoding="utf-8")
        assert "![[pyt__h3decBW12Q_00.webp]]" in body
        assert "<!-- capture failed:" in body

    def test_temp_video_deleted_after_run(self, vault, monkeypatch):
        video, paths = _setup_case(vault)
        recorded_paths: list[Path] = []

        def fake_download(url, dest, resolution="480", *, backend=None):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"stub")
            recorded_paths.append(dest)

        monkeypatch.setattr(capture_stage, "_download_video", fake_download)
        # Pin format to WebP so test is deterministic regardless of host ffmpeg capabilities
        monkeypatch.setattr(
            capture_stage,
            "_resolve_capture_format",
            lambda _fmt, _backend: _FormatChoice(ext="webp", strategy="direct"),
        )
        monkeypatch.setattr(subprocess, "run", _fake_successful_ffmpeg)
        # Pin format to WebP so test is deterministic regardless of host ffmpeg capabilities
        monkeypatch.setattr(
            capture_stage,
            "_resolve_capture_format",
            lambda _fmt, _backend: _FormatChoice(ext="webp", strategy="direct"),
        )

        run_stage_capture(
            video,
            summary_md_path=paths["summary"],
            capture_md_path=paths["capture"],
        )

        assert len(recorded_paths) == 1
        assert not recorded_paths[0].exists(), "temp video should be deleted"
