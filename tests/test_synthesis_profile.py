"""Tests for Stage 05 Agent Teams profile selection and dispatch."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from pipeline_youtube import config
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.providers.base import LLMResponse as ClaudeResponse
from pipeline_youtube.stages.synthesis import (
    SynthesisProfile,
    _select_profile,
    run_stage_synthesis,
)
from pipeline_youtube.synthesis import agents as agents_mod


@pytest.fixture
def vault(tmp_path: Path):
    config.set_vault_root(tmp_path)
    config.set_dry_run(False)
    yield config.get_vault_root()
    config.reset_vault_root()


def _video(i: int) -> VideoMeta:
    return VideoMeta(
        video_id=f"vid{i:03d}",
        title=f"Video {i}",
        url=f"https://www.youtube.com/watch?v=vid{i:03d}",
        duration=1000 + i,
        channel="Test",
        upload_date="20260415",
        playlist_title="Test Playlist",
    )


def _fake(text: str) -> ClaudeResponse:
    return ClaudeResponse(
        text=text,
        model="sonnet",
        input_tokens=3,
        output_tokens=500,
        cache_creation_tokens=24000,
        cache_read_tokens=15000,
        total_cost_usd=0.1,
        duration_ms=20000,
    )


ALPHA_OUT = json.dumps(
    {
        "topics": [
            {
                "topic_id": "t001",
                "label": "concept-a",
                "source_videos": ["vid001", "vid002", "vid003"],
                "duplication_count": 3,
                "category": "core",
                "summary": "s",
                "excerpts": [],
            },
            {
                "topic_id": "t002",
                "label": "concept-b",
                "source_videos": ["vid001", "vid002"],
                "duplication_count": 2,
                "category": "supporting",
                "summary": "s",
            },
        ]
    },
    ensure_ascii=False,
)

BETA_OUT = json.dumps(
    {
        "chapters": [
            {
                "index": 1,
                "label": "Chapter A",
                "category": "core",
                "topic_ids": ["t001", "t002"],
                "source_videos": ["vid001", "vid002", "vid003"],
                "rationale": "r",
            }
        ]
    },
    ensure_ascii=False,
)

LEADER_OUT = json.dumps(
    {
        "moc": {"title": "T", "body_markdown": "# T\n\n## 章構成\n- [[01_Chapter A]]"},
        "chapters": [
            {
                "chapter_index": 1,
                "label": "Chapter A",
                "category": "core",
                "source_video_ids": ["vid001", "vid002", "vid003"],
                "body_markdown": "body",
            }
        ],
    },
    ensure_ascii=False,
)

REVIEWER_OK = json.dumps({"needs_revision": False, "fixes": []}, ensure_ascii=False)

REVIEWER_FIX = json.dumps(
    {
        "needs_revision": True,
        "summary": "missing citations",
        "fixes": [
            {
                "target": "chapter:1",
                "reason": "citation missing",
                "patch_hint": "add [[...#^MM-SS]] to each item",
            }
        ],
    },
    ensure_ascii=False,
)


# =====================================================
# _select_profile
# =====================================================


class TestSelectProfile:
    def test_small_playlist_auto_selects_standard(self):
        assert _select_profile(5, None) is SynthesisProfile.STANDARD
        assert _select_profile(5, "auto") is SynthesisProfile.STANDARD

    def test_boundary_fifteen_is_standard(self):
        assert _select_profile(15, None) is SynthesisProfile.STANDARD

    def test_boundary_sixteen_crosses_to_parallel(self):
        # Guard against off-by-one on _AUTO_STANDARD_MAX_VIDEOS (15).
        assert _select_profile(16, None) is SynthesisProfile.PARALLEL

    def test_mid_range_auto_selects_parallel(self):
        assert _select_profile(20, None) is SynthesisProfile.PARALLEL
        assert _select_profile(30, None) is SynthesisProfile.PARALLEL

    def test_boundary_thirty_one_crosses_to_parallel_full(self):
        # Guard against off-by-one on _AUTO_PARALLEL_MAX_VIDEOS (30).
        assert _select_profile(31, None) is SynthesisProfile.PARALLEL_FULL

    def test_large_auto_selects_parallel_full(self):
        assert _select_profile(50, None) is SynthesisProfile.PARALLEL_FULL

    def test_explicit_override_small_with_full(self):
        assert _select_profile(3, "full") is SynthesisProfile.FULL

    def test_explicit_override_large_with_standard(self):
        assert _select_profile(100, "standard") is SynthesisProfile.STANDARD

    def test_invalid_profile_name_raises(self):
        with pytest.raises(ValueError):
            _select_profile(5, "nonsense")

    def test_flags_reflect_components(self):
        assert SynthesisProfile.STANDARD.uses_parallel_alpha is False
        assert SynthesisProfile.STANDARD.uses_reviewer is False
        assert SynthesisProfile.PARALLEL.uses_parallel_alpha is True
        assert SynthesisProfile.PARALLEL.uses_reviewer is False
        assert SynthesisProfile.FULL.uses_parallel_alpha is False
        assert SynthesisProfile.FULL.uses_reviewer is True
        assert SynthesisProfile.PARALLEL_FULL.uses_parallel_alpha is True
        assert SynthesisProfile.PARALLEL_FULL.uses_reviewer is True


# =====================================================
# run_stage_synthesis with explicit profiles
# =====================================================


def _queue_invoke(monkeypatch, responses: list[str]):
    queue = list(responses)

    def fake_invoke(**kw):
        return _fake(queue.pop(0))

    monkeypatch.setattr(agents_mod, "invoke_claude", fake_invoke)


class TestRunStageWithProfile:
    def test_standard_profile_runs_three_agents(self, vault, monkeypatch):
        _queue_invoke(monkeypatch, [ALPHA_OUT, BETA_OUT, LEADER_OUT])
        videos = [_video(i) for i in range(1, 4)]
        bodies = [f"body{i}" for i in range(1, 4)]
        result = run_stage_synthesis(
            videos,
            bodies,
            run_time=datetime(2026, 4, 15),
            playlist_title="Test Playlist",
            profile="standard",
        )
        assert result.error is None
        assert result.profile is SynthesisProfile.STANDARD
        assert len(result.agent_results) == 3
        assert result.reviewer_feedback is None

    def test_full_profile_invokes_reviewer_no_fixes(self, vault, monkeypatch):
        _queue_invoke(monkeypatch, [ALPHA_OUT, BETA_OUT, LEADER_OUT, REVIEWER_OK])
        videos = [_video(i) for i in range(1, 4)]
        bodies = [f"body{i}" for i in range(1, 4)]
        result = run_stage_synthesis(
            videos,
            bodies,
            run_time=datetime(2026, 4, 15),
            playlist_title="Test Playlist",
            profile="full",
        )
        assert result.error is None
        assert result.profile is SynthesisProfile.FULL
        assert result.reviewer_feedback is not None
        assert result.reviewer_feedback.needs_revision is False
        # α + β + Leader + Reviewer = 4
        assert len(result.agent_results) == 4

    def test_full_profile_rerender_on_revision(self, vault, monkeypatch):
        # α, β, Leader, Reviewer(needs_revision), Leader(retry)
        _queue_invoke(
            monkeypatch,
            [ALPHA_OUT, BETA_OUT, LEADER_OUT, REVIEWER_FIX, LEADER_OUT],
        )
        videos = [_video(i) for i in range(1, 4)]
        bodies = [f"body{i}" for i in range(1, 4)]
        result = run_stage_synthesis(
            videos,
            bodies,
            run_time=datetime(2026, 4, 15),
            playlist_title="Test Playlist",
            profile="full",
        )
        assert result.error is None
        assert result.reviewer_feedback is not None
        assert result.reviewer_feedback.needs_revision is True
        # α + β + Leader + Reviewer + Leader-retry = 5
        assert len(result.agent_results) == 5

    def test_parallel_profile_batches_alpha(self, vault, monkeypatch):
        # With 12 videos and batch_size=10, expect 2 α calls.
        # Stage 05 expects ≥3 videos, and parallel α triggers when profile
        # contains "parallel". Use explicit profile to bypass auto-pick.
        alpha_batch_1 = json.dumps(
            {
                "topics": [
                    {
                        "topic_id": "t001",
                        "label": "x",
                        "source_videos": ["vid001", "vid002"],
                        "duplication_count": 2,
                        "category": "supporting",
                        "summary": "s",
                    }
                ]
            },
            ensure_ascii=False,
        )
        alpha_batch_2 = json.dumps(
            {
                "topics": [
                    {
                        "topic_id": "t001",
                        "label": "y",
                        "source_videos": ["vid011", "vid012"],
                        "duplication_count": 2,
                        "category": "supporting",
                        "summary": "s",
                    }
                ]
            },
            ensure_ascii=False,
        )
        beta_full = json.dumps(
            {
                "chapters": [
                    {
                        "index": 1,
                        "label": "chap-x",
                        "category": "supporting",
                        "topic_ids": ["t001", "t002"],
                        "source_videos": ["vid001"],
                        "rationale": "r",
                    }
                ]
            },
            ensure_ascii=False,
        )
        responses = [alpha_batch_1, alpha_batch_2, beta_full, LEADER_OUT]
        _queue_invoke(monkeypatch, responses)

        videos = [_video(i) for i in range(1, 13)]
        bodies = [f"body{i}" for i in range(1, 13)]
        result = run_stage_synthesis(
            videos,
            bodies,
            run_time=datetime(2026, 4, 15),
            playlist_title="Test Playlist",
            profile="parallel",
        )
        assert result.error is None
        assert result.profile is SynthesisProfile.PARALLEL
        # 2 α batches + β + Leader = 4
        assert len(result.agent_results) == 4
        # Merged topics should have 2 distinct labels
        assert len(result.topics) == 2

    def test_invalid_profile_returns_error(self, vault):
        videos = [_video(i) for i in range(1, 4)]
        bodies = [f"body{i}" for i in range(1, 4)]
        result = run_stage_synthesis(
            videos,
            bodies,
            run_time=datetime(2026, 4, 15),
            playlist_title="Test Playlist",
            profile="not-a-profile",
        )
        assert result.error is not None
        assert "invalid profile" in result.error

    def test_auto_profile_still_works(self, vault, monkeypatch):
        _queue_invoke(monkeypatch, [ALPHA_OUT, BETA_OUT, LEADER_OUT])
        videos = [_video(i) for i in range(1, 4)]
        bodies = [f"body{i}" for i in range(1, 4)]
        result = run_stage_synthesis(
            videos,
            bodies,
            run_time=datetime(2026, 4, 15),
            playlist_title="Test Playlist",
            profile="auto",
        )
        assert result.error is None
        assert result.profile is SynthesisProfile.STANDARD

    def test_meta_json_records_profile(self, vault, monkeypatch):
        _queue_invoke(monkeypatch, [ALPHA_OUT, BETA_OUT, LEADER_OUT])
        videos = [_video(i) for i in range(1, 4)]
        bodies = [f"body{i}" for i in range(1, 4)]
        result = run_stage_synthesis(
            videos,
            bodies,
            run_time=datetime(2026, 4, 15),
            playlist_title="Test Playlist",
            profile="standard",
        )
        assert result.meta_path is not None
        meta = json.loads(result.meta_path.read_text(encoding="utf-8"))
        assert meta["profile"] == "standard"
        assert meta["reviewer_status"] == "skipped"

    def test_meta_records_reviewer_ok(self, vault, monkeypatch):
        _queue_invoke(monkeypatch, [ALPHA_OUT, BETA_OUT, LEADER_OUT, REVIEWER_OK])
        videos = [_video(i) for i in range(1, 4)]
        bodies = [f"body{i}" for i in range(1, 4)]
        result = run_stage_synthesis(
            videos,
            bodies,
            run_time=datetime(2026, 4, 15),
            playlist_title="Test Playlist",
            profile="full",
        )
        meta = json.loads(result.meta_path.read_text(encoding="utf-8"))
        assert meta["reviewer_status"] == "ok"

    def test_reviewer_transient_failure_keeps_leader_output(self, vault, monkeypatch):
        """Non-parse exceptions from call_reviewer must not abort the stage."""
        call_count = {"n": 0}

        def flaky_invoke(**kw):
            call_count["n"] += 1
            # Reviewer is the 4th call in the full profile (α, β, Leader,
            # Reviewer). Simulate a transient network / CLI failure that
            # the old narrow handler would have let escape.
            if call_count["n"] == 4:
                raise TimeoutError("claude CLI timeout")
            return _fake([ALPHA_OUT, BETA_OUT, LEADER_OUT][call_count["n"] - 1])

        monkeypatch.setattr(agents_mod, "invoke_claude", flaky_invoke)

        videos = [_video(i) for i in range(1, 4)]
        bodies = [f"body{i}" for i in range(1, 4)]
        result = run_stage_synthesis(
            videos,
            bodies,
            run_time=datetime(2026, 4, 15),
            playlist_title="Test Playlist",
            profile="full",
        )
        # Stage completes with the original Leader output; Reviewer's
        # failure is logged and recorded in meta.
        assert result.error is None
        assert result.leader_output is not None
        assert result.reviewer_feedback is None
        meta = json.loads(result.meta_path.read_text(encoding="utf-8"))
        assert meta["reviewer_status"] == "failed"
