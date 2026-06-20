"""Tests for the β reflexion retry loop (MAX_BETA_REFLEXION_RETRIES=3).

Locks in:
1. Loop exits as soon as coverage.missing_topic_ids is empty (no wasted call).
2. Loop runs up to MAX_BETA_REFLEXION_RETRIES times, then hands the
   residual to Leader.
3. Parse failure on a retry breaks the loop but preserves the last-good
   chapters (no crash).
4. Each retry call passes the current missing_topic_ids to β.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline_youtube import config as cfg_mod
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.providers.base import LLMResponse as ClaudeResponse
from pipeline_youtube.services.cache import Cache
from pipeline_youtube.stages.synthesis import (
    MAX_BETA_REFLEXION_RETRIES,
    run_stage_synthesis,
)
from pipeline_youtube.synthesis import agents as agents_mod

_NO_CACHE = Cache(None, enabled=False)


@pytest.fixture
def vault(tmp_path: Path):
    cfg_mod.set_vault_root(tmp_path)
    cfg_mod.set_dry_run(True)
    yield tmp_path
    cfg_mod.reset_vault_root()


def _video(i: int) -> VideoMeta:
    vid = f"vid{i:03d}"
    return VideoMeta(
        video_id=vid,
        title=f"Video {i}",
        url=f"https://www.youtube.com/watch?v={vid}",
        duration=600,
        channel="Test",
        upload_date="20260421",
        playlist_title="Test Playlist",
    )


def _resp(text: str) -> ClaudeResponse:
    return ClaudeResponse(
        text=text,
        model="sonnet",
        input_tokens=10,
        output_tokens=20,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        total_cost_usd=0.01,
        duration_ms=500,
    )


_ALPHA_OUT = json.dumps(
    {
        "topics": [
            {
                "topic_id": f"t{i:03d}",
                "label": f"topic{i}",
                "source_videos": ["vid001"],
                "duplication_count": 3,
                "category": "core",
                "summary": f"summary {i}",
                "excerpts": [],
            }
            for i in range(1, 5)
        ]
    }
)


def _beta_out_covering(ids: list[str]) -> str:
    return json.dumps(
        {
            "chapters": [
                {
                    "index": 1,
                    "label": "ch1",
                    "category": "core",
                    "topic_ids": ids,
                    "source_videos": ["vid001"],
                    "rationale": "test",
                }
            ]
        }
    )


_LEADER_OUT = json.dumps(
    {
        "moc": {"title": "t", "body_markdown": "moc body"},
        "chapters": [
            {
                "chapter_index": 1,
                "label": "ch1",
                "category": "core",
                "source_video_ids": ["vid001"],
                "body_markdown": "chapter body",
            }
        ],
    }
)


class TestReflexionLoop:
    def test_no_retry_when_coverage_is_complete(self, vault: Path):
        """β covers all 4 topics on the first shot → no retry."""
        calls: list[dict] = []

        def fake(**kw):
            calls.append(kw)
            sp = kw.get("append_system_prompt") or kw.get("system_prompt") or ""
            if "TopicExtractor" in sp:
                return _resp(_ALPHA_OUT)
            if "ChapterArchitect" in sp:
                return _resp(_beta_out_covering(["t001", "t002", "t003", "t004"]))
            return _resp(_LEADER_OUT)

        with patch.object(agents_mod, "invoke_claude", fake):
            result = run_stage_synthesis(
                [_video(1), _video(2), _video(3)],
                ["b1", "b2", "b3"],
                run_time=datetime(2026, 4, 21),
                playlist_title="Test",
                dry_run=True,
                cache=_NO_CACHE,
            )

        assert result.error is None
        # α + β (no retry) + leader = 3
        assert len(result.agent_results) == 3
        assert result.coverage.missing_topic_ids == []

    def test_one_retry_when_first_attempt_misses(self, vault: Path):
        """β misses t004 first, covers all on retry → single retry only."""
        beta_count = {"n": 0}

        def fake(**kw):
            sp = kw.get("append_system_prompt") or kw.get("system_prompt") or ""
            if "トピックエクストラクター" in sp:
                return _resp(_ALPHA_OUT)
            if "チャプターアーキテクト" in sp:
                beta_count["n"] += 1
                if beta_count["n"] == 1:
                    return _resp(_beta_out_covering(["t001", "t002", "t003"]))
                return _resp(_beta_out_covering(["t001", "t002", "t003", "t004"]))
            return _resp(_LEADER_OUT)

        with patch.object(agents_mod, "invoke_claude", fake):
            result = run_stage_synthesis(
                [_video(1), _video(2), _video(3)],
                ["b1", "b2", "b3"],
                run_time=datetime(2026, 4, 21),
                playlist_title="Test",
                dry_run=True,
                cache=_NO_CACHE,
            )

        assert beta_count["n"] == 2  # 1 initial + 1 retry
        assert result.coverage.missing_topic_ids == []

    def test_exhausts_max_retries_then_lets_leader_handle(self, vault: Path):
        """β never covers t004 → loop runs initial + MAX retries, Leader still executes."""
        beta_count = {"n": 0}

        def fake(**kw):
            sp = kw.get("append_system_prompt") or kw.get("system_prompt") or ""
            if "トピックエクストラクター" in sp:
                return _resp(_ALPHA_OUT)
            if "チャプターアーキテクト" in sp:
                beta_count["n"] += 1
                # Always miss t004
                return _resp(_beta_out_covering(["t001", "t002", "t003"]))
            return _resp(_LEADER_OUT)

        with patch.object(agents_mod, "invoke_claude", fake):
            result = run_stage_synthesis(
                [_video(1), _video(2), _video(3)],
                ["b1", "b2", "b3"],
                run_time=datetime(2026, 4, 21),
                playlist_title="Test",
                dry_run=True,
                cache=_NO_CACHE,
            )

        # initial + MAX retries
        assert beta_count["n"] == 1 + MAX_BETA_REFLEXION_RETRIES
        # Residual miss survives into Leader input
        assert result.coverage.missing_topic_ids == ["t004"]
        # Leader still produced output despite the residual miss
        assert result.leader_output is not None

    def test_retry_passes_missing_ids_to_beta(self, vault: Path):
        """Each retry must feed the current missing IDs back to β."""
        captured_beta: list[list[str] | None] = []
        beta_count = {"n": 0}

        def fake(**kw):
            sp = kw.get("append_system_prompt") or kw.get("system_prompt") or ""
            if "トピックエクストラクター" in sp:
                return _resp(_ALPHA_OUT)
            if "チャプターアーキテクト" in sp:
                beta_count["n"] += 1
                # Capture whether the retry prompt contains the missing IDs
                prompt = kw.get("prompt", "")
                if "t004" in prompt and "漏れ" in prompt:
                    captured_beta.append(["t004"])
                else:
                    captured_beta.append(None)
                if beta_count["n"] < 2:
                    return _resp(_beta_out_covering(["t001", "t002", "t003"]))
                return _resp(_beta_out_covering(["t001", "t002", "t003", "t004"]))
            return _resp(_LEADER_OUT)

        with patch.object(agents_mod, "invoke_claude", fake):
            run_stage_synthesis(
                [_video(1), _video(2), _video(3)],
                ["b1", "b2", "b3"],
                run_time=datetime(2026, 4, 21),
                playlist_title="Test",
                dry_run=True,
                cache=_NO_CACHE,
            )

        # First β call: no missing IDs fed back
        assert captured_beta[0] is None
        # Second β call (retry): missing IDs present
        assert captured_beta[1] == ["t004"]

    def test_parse_failure_on_retry_breaks_loop(self, vault: Path):
        """If a retry returns unparseable JSON, keep last-good chapters."""
        beta_count = {"n": 0}

        def fake(**kw):
            sp = kw.get("append_system_prompt") or kw.get("system_prompt") or ""
            if "トピックエクストラクター" in sp:
                return _resp(_ALPHA_OUT)
            if "チャプターアーキテクト" in sp:
                beta_count["n"] += 1
                if beta_count["n"] == 1:
                    return _resp(_beta_out_covering(["t001", "t002", "t003"]))
                # Retry produces garbage
                return _resp("not valid json")
            return _resp(_LEADER_OUT)

        with patch.object(agents_mod, "invoke_claude", fake):
            result = run_stage_synthesis(
                [_video(1), _video(2), _video(3)],
                ["b1", "b2", "b3"],
                run_time=datetime(2026, 4, 21),
                playlist_title="Test",
                dry_run=True,
                cache=_NO_CACHE,
            )

        # Loop stopped at the first retry failure — only 2 β calls
        assert beta_count["n"] == 2
        # Last-good chapters preserved (3 topics, t004 missing)
        assert result.coverage.missing_topic_ids == ["t004"]
        # Leader still ran
        assert result.leader_output is not None

    def test_max_retries_constant_is_reasonable(self):
        """Value is documented and in a sane band."""
        assert MAX_BETA_REFLEXION_RETRIES == 3
