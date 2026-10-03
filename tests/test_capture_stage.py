"""Tests for stage 03 (capture) with yt-dlp and ffmpeg mocked."""

from __future__ import annotations

import errno
import os
import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pipeline_youtube import config
from pipeline_youtube.pipeline import create_placeholder_notes
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.services.cache import Cache
from pipeline_youtube.stages import capture as capture_stage
from pipeline_youtube.stages.capture import (
    CaptureResult,
    _capture_image_name,
    _FormatChoice,
    parse_summary_ranges,
    run_stage_capture,
)

# These tests stub the capture backend and don't exercise persistent caching,
# so they thread a disabled (no-op) cache.
_NO_CACHE = Cache(None, enabled=False)

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
    config.set_dry_run(False)
    yield tmp_path


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
    paths = create_placeholder_notes(video, run_time, dry_run=False, vault_root=vault)

    summary_path = paths["summary"]
    existing = summary_path.read_text(encoding="utf-8")
    summary_path.write_text(existing + "\n" + summary_md_content, encoding="utf-8")

    return video, paths


def _webp(payload: bytes = b"VP8L") -> bytes:
    """The smallest file Stage 03's check takes for a whole WebP (#190)."""
    body = b"WEBP" + payload
    return b"RIFF" + len(body).to_bytes(4, "little") + body


def _gif(payload: bytes = b"") -> bytes:
    """The smallest file Stage 03's check takes for a whole GIF (#190)."""
    return b"GIF89a" + payload + b";"


def _image_for(path: Path, payload: bytes = b"VP8L") -> bytes:
    return _gif(payload) if path.suffix == ".gif" else _webp(payload)


def _fake_successful_ffmpeg(*args, **kwargs):
    """Mock ffmpeg that creates the output file."""
    # subprocess.run signature: run(cmd, ...)
    cmd = args[0] if args else kwargs.get("args")
    # Last arg is the output path
    output_path = Path(cmd[-1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_image_for(output_path))
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
            cache=_NO_CACHE,
            vault_root=vault,
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

        # ...nested under a per-playlist subfolder whose name matches the
        # playlist's 01~05 note folder, so deleting the playlist's folders
        # has a matching assets subfolder to delete.
        playlist_folder = paths["capture"].parent.name
        for p in result.image_paths:
            assert p.parent.name == playlist_folder
            assert p.parent.parent.name == "pipeline-youtube"

        # 03_Capture md contains range + path-qualified embed blocks. The
        # embed includes the per-playlist subfolder so duplicate basenames
        # across playlists/reruns stay unambiguous in Obsidian.
        capture_body = paths["capture"].read_text(encoding="utf-8")
        assert "[00:00 ~ 01:03]" in capture_body
        assert f"![[{playlist_folder}/pyt__h3decBW12Q_00.webp]]" in capture_body
        assert "[12:50 ~ 15:10]" in capture_body
        assert f"![[{playlist_folder}/pyt__h3decBW12Q_03.webp]]" in capture_body

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
            cache=_NO_CACHE,
            vault_root=vault,
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
            cache=_NO_CACHE,
            vault_root=vault,
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
            cache=_NO_CACHE,
            vault_root=vault,
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
            cache=_NO_CACHE,
            vault_root=vault,
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
            cache=_NO_CACHE,
            vault_root=vault,
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
        playlist_folder = paths["capture"].parent.name
        assert f"![[{playlist_folder}/pyt__h3decBW12Q_00.webp]]" in body
        assert "<!-- capture failed:" in body

    def test_same_folder_rerun_preserves_prior_captures(self, vault, monkeypatch):
        """Same-minute / --force-video rerun must not clobber prior WebPs.

        Notes get ``Title-2.md`` via ``resolve_unique_path``, but captures used
        to overwrite ``pyt_{id}_NN.webp`` with ffmpeg ``-y``. The earlier
        note's embeds would then silently show the rerun's frames.
        """
        video, paths = _setup_case(vault)

        def fake_download(url, dest, resolution="480", *, backend=None):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"stub")

        monkeypatch.setattr(capture_stage, "_download_video", fake_download)
        monkeypatch.setattr(
            capture_stage,
            "_resolve_capture_format",
            lambda _fmt, _backend: _FormatChoice(ext="webp", strategy="direct"),
        )
        monkeypatch.setattr(subprocess, "run", _fake_successful_ffmpeg)

        first = run_stage_capture(
            video,
            summary_md_path=paths["summary"],
            capture_md_path=paths["capture"],
            cache=_NO_CACHE,
            vault_root=vault,
        )
        assert first.success_count == 4
        prior_paths = list(first.image_paths)
        prior_bytes = {p: p.read_bytes() for p in prior_paths}
        playlist_folder = paths["capture"].parent.name
        prior_body = paths["capture"].read_text(encoding="utf-8")

        # Second placeholder in the same playlist folder (same run_time).
        rerun_paths = create_placeholder_notes(
            video,
            datetime(2026, 4, 14, 21, 41),
            dry_run=False,
            vault_root=vault,
        )
        assert rerun_paths["capture"] != paths["capture"]
        assert rerun_paths["capture"].parent == paths["capture"].parent
        assert rerun_paths["capture"].name.endswith("-2.md")
        # Stage 03 reads ranges from the paired summary note.
        rerun_summary = rerun_paths["summary"]
        rerun_summary.write_text(
            rerun_summary.read_text(encoding="utf-8") + "\n" + SAMPLE_SUMMARY,
            encoding="utf-8",
        )

        # Distinct marker bytes so an overwrite would be detectable.
        def fake_rerun_ffmpeg(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args")
            output_path = Path(cmd[-1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(_webp(b"RERUN"))
            return MagicMock(returncode=0, stdout=b"", stderr=b"")

        monkeypatch.setattr(subprocess, "run", fake_rerun_ffmpeg)

        second = run_stage_capture(
            video,
            summary_md_path=rerun_paths["summary"],
            capture_md_path=rerun_paths["capture"],
            cache=_NO_CACHE,
            vault_root=vault,
        )
        assert second.success_count == 4

        for p in prior_paths:
            assert p.exists()
            assert p.read_bytes() == prior_bytes[p], f"prior capture overwritten: {p.name}"

        rerun_names = [p.name for p in second.image_paths]
        assert rerun_names == [
            "pyt__h3decBW12Q_00-2.webp",
            "pyt__h3decBW12Q_01-2.webp",
            "pyt__h3decBW12Q_02-2.webp",
            "pyt__h3decBW12Q_03-2.webp",
        ]
        for p in second.image_paths:
            assert p.read_bytes() == _webp(b"RERUN")
            assert p.parent.name == playlist_folder

        assert paths["capture"].read_text(encoding="utf-8") == prior_body
        rerun_body = rerun_paths["capture"].read_text(encoding="utf-8")
        assert f"![[{playlist_folder}/pyt__h3decBW12Q_00-2.webp]]" in rerun_body
        assert f"![[{playlist_folder}/pyt__h3decBW12Q_00.webp]]" not in rerun_body

    def test_rerun_suffix_keeps_gif_extension(self, vault, monkeypatch):
        """A GIF rerun gets ``_NN-2.gif`` and leaves the prior ``_NN.gif`` alone.

        The collision check re-splits ``pyt_{id}_NN.{ext}`` into stem and
        extension for ``resolve_unique_path``. Every other test here pins WebP,
        so an extension hard-coded to ``.webp`` in that split would pass them
        all while GIF captures (ffmpeg without libwebp) were misnamed and
        checked for collisions against the wrong files.
        """
        video, paths = _setup_case(vault)

        def fake_download(url, dest, resolution="480", *, backend=None):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"stub")

        monkeypatch.setattr(capture_stage, "_download_video", fake_download)
        monkeypatch.setattr(
            capture_stage,
            "_resolve_capture_format",
            lambda _fmt, _backend: _FormatChoice(ext="gif", strategy="native_gif"),
        )
        monkeypatch.setattr(subprocess, "run", _fake_successful_ffmpeg)

        # A prior run's captures already sit in this playlist's assets folder.
        playlist_folder = paths["capture"].parent.name
        assets_dir = vault / capture_stage.ASSETS_REL_PATH / playlist_folder
        assets_dir.mkdir(parents=True)
        prior = [assets_dir / f"pyt__h3decBW12Q_{i:02d}.gif" for i in range(4)]
        for p in prior:
            p.write_bytes(b"PRIOR")

        result = run_stage_capture(
            video,
            summary_md_path=paths["summary"],
            capture_md_path=paths["capture"],
            cache=_NO_CACHE,
            vault_root=vault,
        )

        assert [p.name for p in result.image_paths] == [
            "pyt__h3decBW12Q_00-2.gif",
            "pyt__h3decBW12Q_01-2.gif",
            "pyt__h3decBW12Q_02-2.gif",
            "pyt__h3decBW12Q_03-2.gif",
        ]
        for p in prior:
            assert p.read_bytes() == b"PRIOR", f"prior capture overwritten: {p.name}"
        body = paths["capture"].read_text(encoding="utf-8")
        assert f"![[{playlist_folder}/pyt__h3decBW12Q_00-2.gif]]" in body

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
            cache=_NO_CACHE,
            vault_root=vault,
        )

        assert len(recorded_paths) == 1
        assert not recorded_paths[0].exists(), "temp video should be deleted"


# =====================================================
# #190: make in a staging directory, check, publish without replacing
# =====================================================


def _pin(monkeypatch, ext: str = "webp") -> None:
    def fake_download(url, dest, resolution="480", *, backend=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"stub")

    monkeypatch.setattr(capture_stage, "_download_video", fake_download)
    strategy = "native_gif" if ext == "gif" else "direct"
    monkeypatch.setattr(
        capture_stage,
        "_resolve_capture_format",
        lambda _fmt, _backend: _FormatChoice(ext=ext, strategy=strategy),
    )


def _assets_dir(paths, vault: Path) -> Path:
    return vault / capture_stage.ASSETS_REL_PATH / paths["capture"].parent.name


class TestStagedCapture:
    def test_a_killed_ffmpeg_leaves_no_partial_file_and_no_name_shift(self, vault, monkeypatch):
        """The first range's ffmpeg is killed by the subprocess timeout after
        creating its output (0 bytes, as ffmpeg does at start). Before #190 that
        file stayed at pyt_{id}_00.webp and the next image became _00-2.webp."""
        video, paths = _setup_case(vault)
        _pin(monkeypatch)
        outputs: list[Path] = []

        def killed_first(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args")
            output_path = Path(cmd[-1])
            outputs.append(output_path)
            if len(outputs) == 1:
                output_path.write_bytes(b"")
                raise subprocess.TimeoutExpired(cmd, 0.1)
            return _fake_successful_ffmpeg(*args, **kwargs)

        monkeypatch.setattr(subprocess, "run", killed_first)
        result = run_stage_capture(
            video,
            summary_md_path=paths["summary"],
            capture_md_path=paths["capture"],
            cache=_NO_CACHE,
            vault_root=vault,
        )

        assert result.outcomes[0].image_path is None
        assert "TimeoutExpired" in (result.outcomes[0].error or "")
        assert [p.name for p in result.image_paths] == [
            "pyt__h3decBW12Q_00.webp",
            "pyt__h3decBW12Q_01.webp",
            "pyt__h3decBW12Q_02.webp",
        ]
        assets = _assets_dir(paths, vault)
        assert sorted(p.name for p in assets.iterdir()) == [p.name for p in result.image_paths]
        # ffmpeg only ever wrote inside this run's staging directory, which is gone.
        staging = {p.parent for p in outputs}
        assert len(staging) == 1
        (staging_dir,) = staging
        assert staging_dir.parent == assets
        assert staging_dir.name.startswith(".pyt-capture-")
        assert not staging_dir.exists()

    @pytest.mark.parametrize(
        ("ext", "written", "reason"),
        [
            ("webp", b"", "output_empty"),
            ("webp", _webp()[:-2], "output_truncated"),
            ("webp", _gif(), "output_not_webp"),
            ("gif", _gif()[:-1], "output_truncated"),
            ("gif", _webp(), "output_not_gif"),
        ],
    )
    def test_an_unusable_output_with_exit_zero_is_a_failure(
        self, vault, monkeypatch, ext, written, reason
    ):
        video, paths = _setup_case(vault)
        _pin(monkeypatch, ext)
        calls = {"final": 0}

        def first_output_unusable(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args")
            output_path = Path(cmd[-1])
            if output_path.suffix == f".{ext}":
                calls["final"] += 1
                if calls["final"] == 1:
                    output_path.write_bytes(written)
                    return MagicMock(returncode=0, stdout=b"", stderr=b"")
            return _fake_successful_ffmpeg(*args, **kwargs)

        monkeypatch.setattr(subprocess, "run", first_output_unusable)
        result = run_stage_capture(
            video,
            summary_md_path=paths["summary"],
            capture_md_path=paths["capture"],
            cache=_NO_CACHE,
            vault_root=vault,
        )

        assert result.outcomes[0].image_path is None
        assert result.outcomes[0].error == f"CaptureCheckError: {reason}"
        assert result.success_count == 3
        names = sorted(p.name for p in _assets_dir(paths, vault).iterdir())
        assert names == [f"pyt__h3decBW12Q_{i:02d}.{ext}" for i in range(3)]
        assert "<!-- capture failed: CaptureCheckError" in paths["capture"].read_text(
            encoding="utf-8"
        )

    def test_a_name_taken_after_the_check_is_skipped_not_replaced(self, vault, monkeypatch):
        """Another run takes pyt_{id}_00.webp between the free-name look and the
        write. Publishing links, which fails on an existing name, so this run's
        image goes to -2 and the other file keeps its bytes."""
        video, paths = _setup_case(vault)
        _pin(monkeypatch)
        monkeypatch.setattr(subprocess, "run", _fake_successful_ffmpeg)
        real_link = os.link
        taken: list[Path] = []

        def link_after_someone_else(src, dst, *a, **kw):
            # The publish names files relative to directory descriptors.
            if not taken:
                other = _assets_dir(paths, vault) / dst
                other.write_bytes(b"OTHER RUN")
                taken.append(other)
            return real_link(src, dst, *a, **kw)

        monkeypatch.setattr(capture_stage.os, "link", link_after_someone_else)
        result = run_stage_capture(
            video,
            summary_md_path=paths["summary"],
            capture_md_path=paths["capture"],
            cache=_NO_CACHE,
            vault_root=vault,
        )

        assert taken, "publishing did not link: a rename or a copy can replace a name"
        assert taken[0].name == "pyt__h3decBW12Q_00.webp"
        assert taken[0].read_bytes() == b"OTHER RUN"
        assert result.image_paths[0].name == "pyt__h3decBW12Q_00-2.webp"
        assert result.image_paths[0].read_bytes() == _webp()

    def test_no_hard_links_fails_the_range_without_a_fallback(self, vault, monkeypatch):
        video, paths = _setup_case(vault)
        _pin(monkeypatch)
        monkeypatch.setattr(subprocess, "run", _fake_successful_ffmpeg)

        def no_links(src, dst, *a, **kw):
            raise PermissionError(errno.EPERM, "hard links are not supported here")

        monkeypatch.setattr(capture_stage.os, "link", no_links)
        result = run_stage_capture(
            video,
            summary_md_path=paths["summary"],
            capture_md_path=paths["capture"],
            cache=_NO_CACHE,
            vault_root=vault,
        )

        assert result.success_count == 0
        # A fixed reason: the OSError's own message names both paths.
        assert all(o.error == "CaptureCheckError: publish_failed: EPERM" for o in result.outcomes)
        assert list(_assets_dir(paths, vault).iterdir()) == []

    @pytest.mark.parametrize("swap", ["a link to a file outside", "another file"])
    def test_a_staged_name_swapped_after_the_check_is_not_published(
        self, vault, monkeypatch, tmp_path, swap
    ):
        """Between the check and the link, the staged name is replaced (the
        staging directory is in the assets folder, which a Docker capture
        container can write). The publish must link the file it checked or
        nothing: before, os.link followed the link and published the outside
        file as pyt_{id}_00.webp."""
        video, paths = _setup_case(vault)
        _pin(monkeypatch)
        monkeypatch.setattr(subprocess, "run", _fake_successful_ffmpeg)
        outside = tmp_path / "outside-secret.txt"
        outside.write_bytes(b"NOT AN IMAGE")
        real_check = capture_stage._check_capture
        swapped: list[str] = []

        def check_then_swap(dir_fd, name, ext):
            checked = real_check(dir_fd, name, ext)
            if not swapped:
                os.rename(name, f"{name}.checked", src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
                if swap == "another file":
                    with open(
                        os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=dir_fd),
                        "wb",
                    ) as f:
                        f.write(_webp(b"SWAP"))
                else:
                    os.symlink(outside, name, dir_fd=dir_fd)
                swapped.append(name)
            return checked

        monkeypatch.setattr(capture_stage, "_check_capture", check_then_swap)
        # The publish stats the new name right after the link. If the link had
        # followed the swapped-in symbolic link, the outside file would be
        # linked into the assets folder at that moment, open to a container,
        # even though it is unlinked again right after.
        real_stat = os.stat
        outside_links: list[int] = []

        def stat_and_count(path, *a, **kw):
            outside_links.append(os.lstat(outside).st_nlink)
            return real_stat(path, *a, **kw)

        monkeypatch.setattr(capture_stage.os, "stat", stat_and_count)
        result = run_stage_capture(
            video,
            summary_md_path=paths["summary"],
            capture_md_path=paths["capture"],
            cache=_NO_CACHE,
            vault_root=vault,
        )

        assert swapped, "the swap never ran"
        assert result.outcomes[0].image_path is None
        assert result.outcomes[0].error == "CaptureCheckError: output_replaced"
        assert [p.name for p in result.image_paths] == [
            f"pyt__h3decBW12Q_{i:02d}.webp" for i in range(3)
        ]
        for p in _assets_dir(paths, vault).iterdir():
            assert not p.is_symlink()
            assert p.read_bytes() == _webp(), p.name
        assert outside.read_bytes() == b"NOT AN IMAGE"
        assert outside.stat().st_nlink == 1
        assert outside_links and max(outside_links) == 1, outside_links

    @pytest.mark.parametrize(
        ("make", "reason"),
        [
            ("fifo", "output_not_a_regular_file"),
            ("hard link to another file", "output_linked"),
            ("symlink", "output_unreadable: ELOOP"),
        ],
    )
    def test_a_staged_output_that_is_no_plain_file_is_refused(
        self, vault, monkeypatch, tmp_path, make, reason
    ):
        """What sits at the staged name is not a file ffmpeg wrote: a FIFO (the
        check must not block on it), a hard link to a file elsewhere, or a
        symbolic link."""
        video, paths = _setup_case(vault)
        _pin(monkeypatch)
        outside = tmp_path / "outside.webp"
        outside.write_bytes(_webp())
        calls = {"n": 0}

        def odd_first_output(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args")
            calls["n"] += 1
            if calls["n"] == 1:
                out = Path(cmd[-1])
                if make == "fifo":
                    os.mkfifo(out)
                elif make == "symlink":
                    out.symlink_to(outside)
                else:
                    os.link(outside, out)
                return MagicMock(returncode=0, stdout=b"", stderr=b"")
            return _fake_successful_ffmpeg(*args, **kwargs)

        monkeypatch.setattr(subprocess, "run", odd_first_output)
        result = run_stage_capture(
            video,
            summary_md_path=paths["summary"],
            capture_md_path=paths["capture"],
            cache=_NO_CACHE,
            vault_root=vault,
        )

        assert result.outcomes[0].error == f"CaptureCheckError: {reason}"
        assert result.success_count == 3
        assert outside.stat().st_nlink == 1

    def test_a_webp_longer_than_its_riff_size_is_refused(self, vault, monkeypatch):
        video, paths = _setup_case(vault)
        _pin(monkeypatch)
        calls = {"n": 0}

        def overlong_first(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args")
            calls["n"] += 1
            if calls["n"] == 1:
                Path(cmd[-1]).write_bytes(_webp() + b"\x00\x00")
                return MagicMock(returncode=0, stdout=b"", stderr=b"")
            return _fake_successful_ffmpeg(*args, **kwargs)

        monkeypatch.setattr(subprocess, "run", overlong_first)
        result = run_stage_capture(
            video,
            summary_md_path=paths["summary"],
            capture_md_path=paths["capture"],
            cache=_NO_CACHE,
            vault_root=vault,
        )
        assert result.outcomes[0].error == "CaptureCheckError: output_overlong"

    def test_publishing_stops_after_the_last_candidate_name(self, vault, monkeypatch):
        video, paths = _setup_case(vault, summary_md_content="### [00:10 ~ 00:20] one\n")
        _pin(monkeypatch)
        monkeypatch.setattr(subprocess, "run", _fake_successful_ffmpeg)
        monkeypatch.setattr(capture_stage, "_PUBLISH_MAX_CANDIDATES", 2)
        assets = _assets_dir(paths, vault)
        assets.mkdir(parents=True)
        for name in ("pyt__h3decBW12Q_00.webp", "pyt__h3decBW12Q_00-2.webp"):
            (assets / name).write_bytes(b"PRIOR")
        result = run_stage_capture(
            video,
            summary_md_path=paths["summary"],
            capture_md_path=paths["capture"],
            cache=_NO_CACHE,
            vault_root=vault,
        )
        assert result.outcomes[0].error == "CaptureCheckError: no_free_name"
        assert sorted(p.name for p in assets.iterdir()) == [
            "pyt__h3decBW12Q_00-2.webp",
            "pyt__h3decBW12Q_00.webp",
        ]

    def test_each_range_has_its_own_staged_file(self, vault, monkeypatch):
        """A published image is a hard link to its staged file, so if two ranges
        shared a staged name, the second ffmpeg -y would rewrite the first
        image. Each output here carries its own bytes."""
        video, paths = _setup_case(vault)
        _pin(monkeypatch)
        calls = {"n": 0}

        def numbered(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args")
            calls["n"] += 1
            out = Path(cmd[-1])
            with open(out, "wb") as f:  # truncates in place, as ffmpeg -y does
                f.write(_webp(f"IMG{calls['n']}".encode()))
            return MagicMock(returncode=0, stdout=b"", stderr=b"")

        monkeypatch.setattr(subprocess, "run", numbered)
        result = run_stage_capture(
            video,
            summary_md_path=paths["summary"],
            capture_md_path=paths["capture"],
            cache=_NO_CACHE,
            vault_root=vault,
        )
        assert [p.read_bytes() for p in result.image_paths] == [
            _webp(f"IMG{i}".encode()) for i in range(1, 5)
        ]

    def test_a_staging_directory_that_cannot_be_made_names_the_error_class(
        self, vault, monkeypatch
    ):
        video, paths = _setup_case(vault)
        _pin(monkeypatch)

        def no_tmp(*a, **kw):
            raise PermissionError(errno.EACCES, "denied")

        monkeypatch.setattr(capture_stage.tempfile, "mkdtemp", no_tmp)
        result = run_stage_capture(
            video,
            summary_md_path=paths["summary"],
            capture_md_path=paths["capture"],
            cache=_NO_CACHE,
            vault_root=vault,
        )
        assert result.error == "staging_dir_failed: PermissionError"
        assert result.outcomes == []

    def test_a_staging_directory_that_cannot_be_removed_does_not_fail_the_stage(
        self, vault, monkeypatch
    ):
        """Removing the staging directory is best effort: the images are already
        published. Here its entries cannot be unlinked, and the stage still
        returns its result."""
        video, paths = _setup_case(vault)
        _pin(monkeypatch)
        staging: list[Path] = []

        def lock_the_staging_dir(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args")
            result = _fake_successful_ffmpeg(*args, **kwargs)
            staging.append(Path(cmd[-1]).parent)
            Path(cmd[-1]).parent.chmod(0o500)
            return result

        monkeypatch.setattr(subprocess, "run", lock_the_staging_dir)
        try:
            result = run_stage_capture(
                video,
                summary_md_path=paths["summary"],
                capture_md_path=paths["capture"],
                cache=_NO_CACHE,
                vault_root=vault,
            )
        finally:
            for d in staging:
                if d.exists():
                    d.chmod(0o700)
        if os.geteuid() == 0:
            pytest.skip("root removes the entries anyway")
        assert result.error is None
        assert result.success_count == 1
        assert staging and staging[0].exists(), "the cleanup could not run, so the directory stays"

    @pytest.mark.parametrize("linked", ["the staging directory", "the assets folder"])
    def test_a_directory_that_is_a_link_stops_the_stage(self, vault, monkeypatch, tmp_path, linked):
        """Both directories are opened once without following a link: one that
        is a link (swapped in, or set up so) would make the check and the
        publish act on another folder."""
        video, paths = _setup_case(vault)
        _pin(monkeypatch)
        monkeypatch.setattr(subprocess, "run", _fake_successful_ffmpeg)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        if linked == "the assets folder":
            assets = _assets_dir(paths, vault)
            assets.parent.mkdir(parents=True, exist_ok=True)
            assets.symlink_to(elsewhere)
        else:
            real_mkdtemp = capture_stage.tempfile.mkdtemp

            def linked_mkdtemp(*a, **kw):
                made = Path(real_mkdtemp(*a, **kw))
                made.rmdir()
                made.symlink_to(elsewhere)
                return str(made)

            monkeypatch.setattr(capture_stage.tempfile, "mkdtemp", linked_mkdtemp)
        result = run_stage_capture(
            video,
            summary_md_path=paths["summary"],
            capture_md_path=paths["capture"],
            cache=_NO_CACHE,
            vault_root=vault,
        )
        # The errno differs by system (ENOTDIR on macOS, ELOOP elsewhere).
        assert (result.error or "").startswith("staging_dir_failed: "), result.error
        assert result.outcomes == []
        assert list(elsewhere.iterdir()) == [] or linked == "the assets folder"


@pytest.mark.parametrize("cache_state", ["hit", "miss", "disabled"])
@pytest.mark.parametrize("failure", [None, "capture", "staging"])
def test_staged_capture_preserves_cache_and_cleans_working_video(
    vault, monkeypatch, cache_state, failure
):
    """SDK cache ownership survives both successful publication and early failures."""
    video, paths = _setup_case(vault, summary_md_content="### [00:10 ~ 00:20] one\n")
    _pin(monkeypatch)
    cache = Cache(vault / "cache", enabled=cache_state != "disabled")
    working = vault / "working.mp4"
    monkeypatch.setattr(capture_stage, "_tmp_video_path", lambda _video: working)
    if cache_state == "hit":
        seed = vault / "seed.mp4"
        seed.write_bytes(b"SOURCE VIDEO")
        cache.put_video(video.video_id, "480", seed)

    downloads: list[Path] = []
    sources: list[Path] = []

    def download(url, dest, resolution="480", *, backend=None):
        downloads.append(dest)
        dest.write_bytes(b"SOURCE VIDEO")

    def extract(source, output, **kwargs):
        assert source.read_bytes() == b"SOURCE VIDEO"
        sources.append(source)
        output.write_bytes(b"" if failure == "capture" else _webp())

    def no_staging(*args, **kwargs):
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr(capture_stage, "_download_video", download)
    monkeypatch.setattr(capture_stage, "_dispatch_extractor", lambda _strategy: extract)
    if failure == "staging":
        monkeypatch.setattr(capture_stage.tempfile, "mkdtemp", no_staging)

    result = run_stage_capture(
        video,
        summary_md_path=paths["summary"],
        capture_md_path=paths["capture"],
        cache=cache,
        vault_root=vault,
    )

    assert downloads == ([] if cache_state == "hit" else [working])
    assert result.video_downloaded is (cache_state != "hit")
    assert not working.exists()
    persistent = cache.get_video(video.video_id, "480")
    if cache_state == "disabled":
        assert persistent is None
        assert not (vault / "cache").exists()
    else:
        assert persistent is not None
        assert persistent.read_bytes() == b"SOURCE VIDEO"
    if failure == "staging":
        assert result.error == "staging_dir_failed: PermissionError"
        assert sources == []
    else:
        assert sources == [persistent if cache_state == "hit" else working]
        assert result.success_count == (0 if failure else 1)
        if failure == "capture":
            assert result.outcomes[0].error == "CaptureCheckError: output_empty"
    assert not list(_assets_dir(paths, vault).glob(".pyt-capture-*"))
