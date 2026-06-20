"""End-to-end tests for stages/synthesis.py with all claude calls mocked."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from pipeline_youtube import config
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.providers.base import LLMResponse as ClaudeResponse
from pipeline_youtube.services.cache import Cache
from pipeline_youtube.stages.synthesis import (
    run_stage_synthesis,
)
from pipeline_youtube.synthesis import agents as agents_mod

_NO_CACHE = Cache(None, enabled=False)


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
                "label": "コンテキスト管理",
                "source_videos": ["vid001", "vid002", "vid003"],
                "duplication_count": 3,
                "category": "core",
                "summary": "s",
                "excerpts": [],
            },
            {
                "topic_id": "t002",
                "label": "Agent Teams",
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
                "label": "コンテキスト管理の基礎",
                "category": "core",
                "topic_ids": ["t001"],
                "source_videos": ["vid001", "vid002", "vid003"],
                "rationale": "r",
            },
            {
                "index": 2,
                "label": "Agent Teams 実装",
                "category": "supporting",
                "topic_ids": ["t002"],
                "source_videos": ["vid001", "vid002"],
                "rationale": "r",
            },
        ]
    },
    ensure_ascii=False,
)

LEADER_OUT = json.dumps(
    {
        "moc": {
            "title": "Test Playlist ハンズオン",
            "body_markdown": "# Test Playlist ハンズオン\n\n## 章構成\n- [[01_コンテキスト管理の基礎]] - core\n- [[02_Agent Teams 実装]] - supporting",
        },
        "chapters": [
            {
                "chapter_index": 1,
                "label": "コンテキスト管理の基礎",
                "category": "core",
                "source_video_ids": ["vid001", "vid002", "vid003"],
                "body_markdown": "> [!important]\n> コア概念です\n\n## 概念定義\n\n...",
            },
            {
                "chapter_index": 2,
                "label": "Agent Teams 実装",
                "category": "supporting",
                "source_video_ids": ["vid001", "vid002"],
                "body_markdown": "## 実装\n\n**Agent Teams** とは...",
            },
        ],
    },
    ensure_ascii=False,
)


def _mock_all_agents(monkeypatch, responses: list[str] | None = None):
    """Monkeypatch invoke_claude to return queued responses in order.

    γ was removed (replaced by deterministic Python set diff), so only
    α / β / Leader are LLM-backed.
    """
    queue = list(responses or [ALPHA_OUT, BETA_OUT, LEADER_OUT])

    def fake_invoke(**kw):
        return _fake(queue.pop(0))

    monkeypatch.setattr(agents_mod, "invoke_claude", fake_invoke)


# =====================================================
# Skip rules
# =====================================================


class TestSkipRules:
    def test_skips_playlist_with_less_than_three_videos(self, vault):
        result = run_stage_synthesis(
            [_video(1), _video(2)],
            ["body1", "body2"],
            run_time=datetime(2026, 4, 15),
            playlist_title="Small Playlist",
            cache=_NO_CACHE,
        )
        assert result.skipped is True
        assert result.skip_reason is not None
        assert "2 videos" in result.skip_reason

    def test_runs_with_exactly_three_videos(self, vault, monkeypatch):
        _mock_all_agents(monkeypatch)
        videos = [_video(i) for i in range(1, 4)]
        bodies = [f"body{i}" for i in range(1, 4)]

        result = run_stage_synthesis(
            videos,
            bodies,
            run_time=datetime(2026, 4, 15),
            playlist_title="Test Playlist",
            cache=_NO_CACHE,
        )
        assert result.skipped is False
        assert result.error is None

    def test_min_playlist_size_override_raises_threshold(self, vault):
        videos = [_video(i) for i in range(1, 4)]
        bodies = [f"body{i}" for i in range(1, 4)]
        result = run_stage_synthesis(
            videos,
            bodies,
            run_time=datetime(2026, 4, 15),
            playlist_title="Test Playlist",
            min_playlist_size=5,
            cache=_NO_CACHE,
        )
        assert result.skipped is True
        assert result.skip_reason is not None
        assert "< 5" in result.skip_reason

    def test_max_chapters_threads_through_to_beta(self, vault, monkeypatch):
        _mock_all_agents(monkeypatch)
        captured: dict = {}
        original = agents_mod.invoke_claude

        def spy(**kw):
            if "チャプターアーキテクト" in kw.get("append_system_prompt", ""):
                captured.update(kw)
            return original(**kw)

        monkeypatch.setattr(agents_mod, "invoke_claude", spy)

        videos = [_video(i) for i in range(1, 4)]
        bodies = [f"body{i}" for i in range(1, 4)]
        run_stage_synthesis(
            videos,
            bodies,
            run_time=datetime(2026, 4, 15),
            playlist_title="Test Playlist",
            max_chapters=4,
            cache=_NO_CACHE,
        )
        assert "最大 4 章" in captured["prompt"]

    def test_length_mismatch_returns_error(self, vault):
        result = run_stage_synthesis(
            [_video(1), _video(2), _video(3)],
            ["body1", "body2"],
            run_time=datetime(2026, 4, 15),
            playlist_title="x",
            cache=_NO_CACHE,
        )
        assert result.error is not None
        assert "length mismatch" in result.error


# =====================================================
# Happy path
# =====================================================


class TestHappyPath:
    def test_writes_moc_and_chapter_files(self, vault, monkeypatch):
        _mock_all_agents(monkeypatch)
        videos = [_video(i) for i in range(1, 4)]
        bodies = [f"body{i}" for i in range(1, 4)]

        result = run_stage_synthesis(
            videos,
            bodies,
            run_time=datetime(2026, 4, 15),
            playlist_title="Test Playlist",
            cache=_NO_CACHE,
        )

        assert result.error is None
        assert result.moc_path is not None
        assert result.moc_path.exists()
        assert result.moc_path.name == "00_MOC.md"

        assert len(result.chapter_paths) == 2
        chapter_names = [p.name for p in result.chapter_paths]
        assert any("01_" in n for n in chapter_names)
        assert any("02_" in n for n in chapter_names)

    def test_moc_has_frontmatter_and_body(self, vault, monkeypatch):
        _mock_all_agents(monkeypatch)
        videos = [_video(i) for i in range(1, 4)]
        bodies = [f"body{i}" for i in range(1, 4)]

        result = run_stage_synthesis(
            videos,
            bodies,
            run_time=datetime(2026, 4, 15),
            playlist_title="Test Playlist",
            cache=_NO_CACHE,
        )

        moc_content = result.moc_path.read_text(encoding="utf-8")
        assert moc_content.startswith("---\n")
        assert 'playlist: "Test Playlist"' in moc_content
        assert "synthesis" in moc_content
        assert "moc" in moc_content
        assert "# Test Playlist ハンズオン" in moc_content

    def test_chapter_has_category_in_frontmatter(self, vault, monkeypatch):
        _mock_all_agents(monkeypatch)
        videos = [_video(i) for i in range(1, 4)]
        bodies = [f"body{i}" for i in range(1, 4)]

        result = run_stage_synthesis(
            videos,
            bodies,
            run_time=datetime(2026, 4, 15),
            playlist_title="Test Playlist",
            cache=_NO_CACHE,
        )

        # First chapter is core → has > [!important] callout
        ch1 = result.chapter_paths[0].read_text(encoding="utf-8")
        assert 'category: "core"' in ch1
        assert "> [!important]" in ch1
        assert 'chapter: "1"' in ch1

    def test_meta_duplicate_score_json_written(self, vault, monkeypatch):
        _mock_all_agents(monkeypatch)
        videos = [_video(i) for i in range(1, 4)]
        bodies = [f"body{i}" for i in range(1, 4)]

        result = run_stage_synthesis(
            videos,
            bodies,
            run_time=datetime(2026, 4, 15),
            playlist_title="Test Playlist",
            cache=_NO_CACHE,
        )

        assert result.meta_path is not None
        assert result.meta_path.exists()
        assert result.meta_path.parent.name == "_meta"

        meta = json.loads(result.meta_path.read_text(encoding="utf-8"))
        assert len(meta["topics"]) == 2
        assert meta["topics"][0]["topic_id"] == "t001"
        assert meta["topics"][0]["category"] == "core"
        assert len(meta["chapters"]) == 2
        assert meta["coverage"]["missing_topic_ids"] == []

    def test_agent_results_captured(self, vault, monkeypatch):
        _mock_all_agents(monkeypatch)
        videos = [_video(i) for i in range(1, 4)]
        bodies = [f"body{i}" for i in range(1, 4)]

        result = run_stage_synthesis(
            videos,
            bodies,
            run_time=datetime(2026, 4, 15),
            playlist_title="Test Playlist",
            cache=_NO_CACHE,
        )

        # 3 agent calls (alpha + beta + leader); γ was replaced by a Python set diff.
        assert len(result.agent_results) == 3
        assert result.total_output_tokens == 500 * 3
        assert result.total_cache_creation_tokens == 24000 * 3

    def test_dry_run_does_not_write(self, vault, monkeypatch):
        _mock_all_agents(monkeypatch)
        videos = [_video(i) for i in range(1, 4)]
        bodies = [f"body{i}" for i in range(1, 4)]

        result = run_stage_synthesis(
            videos,
            bodies,
            run_time=datetime(2026, 4, 15),
            playlist_title="Test Playlist",
            dry_run=True,
            cache=_NO_CACHE,
        )

        assert result.moc_path is None
        assert result.chapter_paths == []
        assert result.leader_output is not None


# =====================================================
# Error propagation
# =====================================================


class TestErrorHandling:
    def test_alpha_parse_error(self, vault, monkeypatch):
        _mock_all_agents(monkeypatch, responses=["not json", BETA_OUT, LEADER_OUT])
        videos = [_video(i) for i in range(1, 4)]
        result = run_stage_synthesis(
            videos,
            ["b1", "b2", "b3"],
            run_time=datetime(2026, 4, 15),
            playlist_title="x",
            cache=_NO_CACHE,
        )
        assert result.error is not None
        assert "alpha_parse_failed" in result.error

    def test_beta_parse_error_after_alpha_ok(self, vault, monkeypatch):
        _mock_all_agents(monkeypatch, responses=[ALPHA_OUT, "not json", LEADER_OUT])
        videos = [_video(i) for i in range(1, 4)]
        result = run_stage_synthesis(
            videos,
            ["b1", "b2", "b3"],
            run_time=datetime(2026, 4, 15),
            playlist_title="x",
            cache=_NO_CACHE,
        )
        assert result.error is not None
        assert "beta_parse_failed" in result.error
        # alpha result is still recorded
        assert len(result.topics) == 2


# =====================================================
# Reflexion loop (β re-run on missing coverage)
# =====================================================


# β produces chapters that miss t002. After the retry prompt is added,
# β's second attempt covers both.
BETA_OUT_MISSING_T002 = json.dumps(
    {
        "chapters": [
            {
                "index": 1,
                "label": "コンテキスト管理の基礎",
                "category": "core",
                "topic_ids": ["t001"],
                "source_videos": ["vid001"],
                "rationale": "r",
            }
        ]
    },
    ensure_ascii=False,
)


class TestReflexionLoop:
    def test_rerun_beta_when_topics_missing(self, vault, monkeypatch):
        """α finds t001+t002, β's first attempt only uses t001, β retry covers both."""
        captured_prompts: list[str] = []

        def spy_invoke(**kw):
            captured_prompts.append(kw.get("prompt", ""))
            return _fake(
                [ALPHA_OUT, BETA_OUT_MISSING_T002, BETA_OUT, LEADER_OUT][len(captured_prompts) - 1]
            )

        monkeypatch.setattr(agents_mod, "invoke_claude", spy_invoke)

        videos = [_video(i) for i in range(1, 4)]
        bodies = [f"body{i}" for i in range(1, 4)]
        result = run_stage_synthesis(
            videos,
            bodies,
            run_time=datetime(2026, 4, 15),
            playlist_title="Test Playlist",
            cache=_NO_CACHE,
        )

        assert result.error is None
        # α + β-first + β-retry + Leader = 4 agent calls (usually 3)
        assert len(result.agent_results) == 4
        # Final coverage must be complete
        assert result.coverage.missing_topic_ids == []
        # The retry prompt must include the missing-IDs reflexion block
        retry_prompt = captured_prompts[2]
        assert "エラー: 前回の章立てに漏れがあります" in retry_prompt
        assert "t002" in retry_prompt

    def test_no_rerun_when_first_attempt_covers_all(self, vault, monkeypatch):
        """Happy path should still issue only 3 LLM calls."""
        _mock_all_agents(monkeypatch)  # ALPHA + BETA (full coverage) + LEADER
        videos = [_video(i) for i in range(1, 4)]
        result = run_stage_synthesis(
            videos,
            [f"body{i}" for i in range(1, 4)],
            run_time=datetime(2026, 4, 15),
            playlist_title="Test Playlist",
            cache=_NO_CACHE,
        )
        assert result.error is None
        assert len(result.agent_results) == 3

    def test_retry_parse_failure_is_swallowed(self, vault, monkeypatch):
        """If the β retry returns garbage, keep first-attempt chapters and proceed to Leader."""
        _mock_all_agents(
            monkeypatch,
            responses=[ALPHA_OUT, BETA_OUT_MISSING_T002, "not json", LEADER_OUT],
        )
        videos = [_video(i) for i in range(1, 4)]
        result = run_stage_synthesis(
            videos,
            [f"body{i}" for i in range(1, 4)],
            run_time=datetime(2026, 4, 15),
            playlist_title="Test Playlist",
            cache=_NO_CACHE,
        )
        # Pipeline completes; coverage still shows the miss so Leader can handle it.
        assert result.error is None
        assert result.coverage.missing_topic_ids == ["t002"]
        # 4 calls were made (α, β-first, β-retry-garbage, Leader)
        assert len(result.agent_results) >= 3
