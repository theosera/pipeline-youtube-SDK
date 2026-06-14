"""Tests for the opt-in sub-agent orchestration helpers (parallel.py)."""

from __future__ import annotations

import sys

import pytest

from pipeline_youtube.parallel import (
    build_synthesis_argv,
    build_worker_argv,
    parse_video_range,
    split_into_shards,
    strip_cli_option,
)


class TestSplitIntoShards:
    def test_23_videos_3_shards_remainder_on_last(self) -> None:
        # The canonical example: 23 → 1-8, 9-16, 17-23 (last shard gets the rest).
        assert split_into_shards(23, 3) == [(0, 8), (8, 16), (16, 23)]

    def test_even_split(self) -> None:
        assert split_into_shards(9, 3) == [(0, 3), (3, 6), (6, 9)]

    def test_last_shard_smaller(self) -> None:
        assert split_into_shards(7, 3) == [(0, 3), (3, 6), (6, 7)]

    def test_small_playlist_drops_empty_trailing_shards(self) -> None:
        # 2 videos with 3 requested agents → 2 workers, not one idle worker.
        assert split_into_shards(2, 3) == [(0, 1), (1, 2)]

    def test_single_video(self) -> None:
        assert split_into_shards(1, 3) == [(0, 1)]

    def test_zero_videos(self) -> None:
        assert split_into_shards(0, 3) == []

    def test_shard_count_one_is_whole_list(self) -> None:
        assert split_into_shards(10, 1) == [(0, 10)]

    def test_shards_are_contiguous_and_cover_everything(self) -> None:
        shards = split_into_shards(23, 3)
        assert shards[0][0] == 0
        assert shards[-1][1] == 23
        for (_, end), (nxt_start, _) in zip(shards, shards[1:], strict=False):
            assert end == nxt_start

    def test_invalid_shard_count_raises(self) -> None:
        with pytest.raises(ValueError, match="shard_count"):
            split_into_shards(10, 0)


class TestParseVideoRange:
    def test_basic(self) -> None:
        assert parse_video_range("8:16") == (8, 16)

    def test_zero_start(self) -> None:
        assert parse_video_range("0:8") == (0, 8)

    @pytest.mark.parametrize("spec", ["8", "8:16:24", "a:b", "", "-1:4", "10:4"])
    def test_invalid(self, spec: str) -> None:
        with pytest.raises(ValueError):
            parse_video_range(spec)


class TestStripCliOption:
    def test_removes_space_separated(self) -> None:
        argv = ["url", "--sub-agents", "3", "--concurrency", "2"]
        assert strip_cli_option(argv, "--sub-agents") == ["url", "--concurrency", "2"]

    def test_removes_equals_form(self) -> None:
        argv = ["url", "--sub-agents=3", "--dry-run"]
        assert strip_cli_option(argv, "--sub-agents") == ["url", "--dry-run"]

    def test_no_op_when_absent(self) -> None:
        argv = ["url", "--concurrency", "2"]
        assert strip_cli_option(argv, "--sub-agents") == argv

    def test_does_not_strip_following_unrelated_flags(self) -> None:
        argv = ["--sub-agents", "3", "url", "--model", "opus"]
        assert strip_cli_option(argv, "--sub-agents") == ["url", "--model", "opus"]


class TestBuildArgv:
    def test_worker_argv_appends_shard_flags(self) -> None:
        base = ["https://yt/playlist", "--concurrency", "2"]
        argv = build_worker_argv(
            base, run_timestamp="2026-06-14T13:00:00", start=8, end=16, code_bearing=True
        )
        assert argv[:3] == [sys.executable, "-m", "pipeline_youtube.main"]
        assert argv[3:5] == ["https://yt/playlist", "--concurrency"]
        assert argv[-8:] == [
            "--sub-agents",
            "1",
            "--run-timestamp",
            "2026-06-14T13:00:00",
            "--code-bearing",
            "--video-range",
            "8:16",
            "--skip-synthesis",
        ]
        # Recursion guard: exactly one --sub-agents, pinned to 1.
        assert argv.count("--sub-agents") == 1

    def test_worker_argv_pins_no_code_bearing(self) -> None:
        argv = build_worker_argv(
            ["url"], run_timestamp="2026-06-14T13:00:00", start=0, end=8, code_bearing=False
        )
        assert "--no-code-bearing" in argv
        assert "--code-bearing" not in argv

    def test_synthesis_argv_is_synthesis_only(self) -> None:
        base = ["https://yt/playlist", "--model", "opus"]
        argv = build_synthesis_argv(base, run_timestamp="2026-06-14T13:00:00", code_bearing=True)
        assert argv[-1] == "--synthesis-only"
        assert "--skip-synthesis" not in argv
        assert "--code-bearing" in argv
        assert argv[argv.index("--sub-agents") + 1] == "1"
        assert "--run-timestamp" in argv
