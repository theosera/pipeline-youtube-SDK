"""Tests for M1 (_assert_not_flaglike) and L1 (sweep_stale_tmp)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from pipeline_youtube.stages.capture import _assert_not_flaglike, sweep_stale_tmp


class TestAssertNotFlaglike:
    def test_absolute_path_ok(self):
        _assert_not_flaglike(Path("/tmp/foo.mp4"))

    def test_dotslash_path_ok(self):
        _assert_not_flaglike(Path("./foo.mp4"))

    def test_dash_prefix_rejected(self):
        with pytest.raises(ValueError, match="flag-like"):
            _assert_not_flaglike(Path("-foo.mp4"))

    def test_dash_relative_rejected(self):
        with pytest.raises(ValueError, match="flag-like"):
            _assert_not_flaglike(Path("-AbCd12345e.mp4"))


class TestSweepStaleTmp:
    def test_missing_dir_noop(self, tmp_path: Path):
        assert sweep_stale_tmp(tmp_path / "missing") == 0

    def test_keeps_recent_files(self, tmp_path: Path):
        recent = tmp_path / "new.mp4"
        recent.write_bytes(b"x")
        assert sweep_stale_tmp(tmp_path, older_than_hours=24.0) == 0
        assert recent.exists()

    def test_removes_stale_video(self, tmp_path: Path):
        stale = tmp_path / "old.mp4"
        stale.write_bytes(b"x")
        past = time.time() - 48 * 3600
        os.utime(stale, (past, past))
        assert sweep_stale_tmp(tmp_path, older_than_hours=24.0) == 1
        assert not stale.exists()

    def test_ignores_unknown_extensions(self, tmp_path: Path):
        keep = tmp_path / "old.log"
        keep.write_bytes(b"x")
        past = time.time() - 48 * 3600
        os.utime(keep, (past, past))
        assert sweep_stale_tmp(tmp_path, older_than_hours=24.0) == 0
        assert keep.exists()

    def test_removes_multiple_extensions(self, tmp_path: Path):
        past = time.time() - 48 * 3600
        for name in ("a.mp4", "b.webm", "c.m4a", "d.mkv"):
            p = tmp_path / name
            p.write_bytes(b"x")
            os.utime(p, (past, past))
        assert sweep_stale_tmp(tmp_path, older_than_hours=24.0) == 4
