"""Verifies the --handson entry-layer wiring: flag exclusivity in
cli_validation, RunMode decision + run_handson plan flag, and the
single-video guard in build_plan.
"""

from __future__ import annotations

from pathlib import Path

import click
import pytest

from pipeline_youtube.cli_config import CliConfig
from pipeline_youtube.cli_types import CliRequest, ResolvedInput, RunMode, Runtime
from pipeline_youtube.cli_validation import validate_request
from pipeline_youtube.execution_plan import _decide_mode, build_plan
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.services.cache import Cache

_NO_CACHE = Cache(None, enabled=False)


def _request(**overrides) -> CliRequest:
    defaults = {
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "dry_run": False,
        "concurrency": 3,
        "sub_agents": 1,
        "video_range": None,
        "run_timestamp": None,
        "code_bearing_override": None,
        "transcript_concurrency": None,
        "llm_concurrency": None,
        "download_concurrency": None,
        "cache_dir": None,
        "no_cache": False,
        "cache_llm_synthesis": False,
        "skip_synthesis": False,
        "synthesis_only": False,
        "folder_name": None,
        "eval_loop": 0,
        "force_video": (),
        "capture_format": "auto",
        "model": "sonnet",
        "min_playlist_size": 3,
        "max_chapters": None,
        "config_path": None,
        "stop_after_capture": False,
        "resume_reviewed": False,
        "capture_backend": None,
        "synthesis_timeout": None,
        "synthesis_profile": None,
        "provider": None,
        "hybrid": False,
        "handson": False,
        "local_media": None,
    }
    defaults.update(overrides)
    return CliRequest(**defaults)


def _video(video_id: str = "dQw4w9WgXcQ") -> VideoMeta:
    return VideoMeta(
        video_id=video_id,
        title="Talk",
        url=f"https://www.youtube.com/watch?v={video_id}",
        duration=9000,
        channel="Test",
        upload_date="20260601",
        playlist_title=None,
    )


def _runtime(tmp_path: Path) -> Runtime:
    return Runtime(
        cfg=CliConfig(vault_root=tmp_path, models={}, filler_words=()),
        vault_root=tmp_path,
        filler_words=(),
        project_root=tmp_path,
        logs_dir=tmp_path,
        models={},
        cache=_NO_CACHE,
        capture_backend=None,
        synthesis_timeout=None,
        synthesis_profile="auto",
    )


def _resolved(n_videos: int = 1) -> ResolvedInput:
    return ResolvedInput(
        videos=[_video(f"vid{i:08d}xyz"[:11]) for i in range(n_videos)],
        media_map={},
        playlist_title="Talk",
        code_bearing=False,
    )


class TestHandsonExclusivity:
    @pytest.mark.parametrize(
        ("flag_name", "overrides"),
        [
            ("--sub-agents > 1", {"sub_agents": 3}),
            ("--video-range", {"video_range": "0:1"}),
            ("--synthesis-only", {"synthesis_only": True}),
            ("--resume-reviewed", {"resume_reviewed": True}),
            ("--stop-after-capture", {"stop_after_capture": True}),
            ("--skip-synthesis", {"skip_synthesis": True}),
            ("--local-media", {"local_media": Path("/tmp"), "url": None}),
        ],
    )
    def test_conflicting_flag_is_rejected(self, flag_name: str, overrides: dict):
        request = _request(handson=True, **overrides)
        with pytest.raises(click.UsageError, match="--handson cannot be combined"):
            validate_request(request)

    def test_plain_handson_passes_validation(self):
        validate_request(_request(handson=True))

    def test_handson_off_keeps_existing_behavior(self):
        validate_request(_request())


class TestHandsonMode:
    def test_decide_mode_returns_handson(self):
        assert _decide_mode(_request(handson=True)) is RunMode.HANDSON

    def test_sub_agent_precedence_stays_above_handson(self):
        # cli_validation rejects the combo; precedence is a robustness net.
        assert _decide_mode(_request(handson=True, sub_agents=3)) is RunMode.SUB_AGENT_PARENT

    def test_build_plan_sets_run_handson(self, tmp_path: Path):
        plan = build_plan(_request(handson=True), _runtime(tmp_path), _resolved(1))
        assert plan.mode is RunMode.HANDSON
        assert plan.run_handson is True

    def test_normal_plan_keeps_run_handson_false(self, tmp_path: Path):
        plan = build_plan(_request(), _runtime(tmp_path), _resolved(3))
        assert plan.run_handson is False

    def test_resume_reviewed_disables_checkpoint(self, tmp_path: Path):
        # Phase 3 must re-run Stage 04 from reviewed notes; a same-day leftover
        # 04 from an earlier full run must not checkpoint-skip past the review gate.
        plan = build_plan(_request(resume_reviewed=True), _runtime(tmp_path), _resolved(3))
        assert plan.filter_reviewed_only is True
        assert plan.allow_checkpoint is False

    def test_normal_plan_keeps_checkpoint_enabled(self, tmp_path: Path):
        plan = build_plan(_request(), _runtime(tmp_path), _resolved(3))
        assert plan.allow_checkpoint is True


class TestSingleVideoGuard:
    def test_playlist_input_is_rejected(self, tmp_path: Path):
        with pytest.raises(click.UsageError, match="exactly one video"):
            build_plan(_request(handson=True), _runtime(tmp_path), _resolved(3))

    def test_single_video_passes(self, tmp_path: Path):
        plan = build_plan(_request(handson=True), _runtime(tmp_path), _resolved(1))
        assert plan.run_handson is True

    def test_guard_does_not_apply_to_normal_mode(self, tmp_path: Path):
        plan = build_plan(_request(), _runtime(tmp_path), _resolved(5))
        assert plan.mode is RunMode.NORMAL
