"""End-to-end CLI test with yt-dlp / ffmpeg / LLM provider mocked.

Runs a 3-video playlist through stages 01-05 to catch regressions in
the orchestration layer (stage sequencing, per-stage model routing,
checkpoint / phase-gate logic, cost aggregation).

All external calls are stubbed:
  - `yt-dlp`'s `fetch_metadata` returns 3 synthetic `VideoMeta`.
  - `run_stage_scripts` returns a canned Japanese transcript result.
  - `prefetch_video_download` / `run_stage_capture` return a success
    stub; no real video download occurs.
  - `invoke_claude` returns stage-appropriate canned bodies.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from pipeline_youtube import main as main_mod
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.providers.base import LLMResponse as ClaudeResponse
from pipeline_youtube.stages.capture import CaptureResult, SummaryRange
from pipeline_youtube.synthesis import agents as agents_mod
from pipeline_youtube.transcript.base import (
    TranscriptSnippet,
    TranscriptSource,
    build_result,
)

SUMMARY_OUTPUT = (
    "ONE_LINER: 本日の核心論点\n\n"
    "## 全体サマリ\n\n動画全体の主要な論点を記載。\n\n"
    "## 要点タイムライン\n\n"
    "### [00:00 ~ 00:30] intro\n本文。\n\n"
    "### [00:30 ~ 01:00] key point\n本文。\n"
)

LEARNING_OUTPUT = (
    "## 学習のポイント\n\n### [00:00 ~ 00:30] intro\n時系列メモ。\n\n## 要点\n本文。\n"
)

ALPHA_OUT = json.dumps(
    {
        "topics": [
            {
                "topic_id": "t1",
                "label": "コンテキスト管理",
                "source_videos": ["vid001", "vid002", "vid003"],
                "duplication_count": 3,
                "category": "core",
                "summary": "s",
            }
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
                "topic_ids": ["t1"],
                "source_videos": ["vid001", "vid002", "vid003"],
                "rationale": "r",
            }
        ]
    },
    ensure_ascii=False,
)
LEADER_OUT = json.dumps(
    {
        "moc": {
            "title": "Test Playlist ハンズオン",
            "body_markdown": "# MOC\n- [[01_コンテキスト管理の基礎]]",
        },
        "chapters": [
            {
                "chapter_index": 1,
                "label": "コンテキスト管理の基礎",
                "category": "core",
                "source_video_ids": ["vid001", "vid002", "vid003"],
                "body_markdown": "## 概念定義\n\n本文。\n",
            }
        ],
    },
    ensure_ascii=False,
)


def _fake_response(text: str, model: str = "sonnet", cost: float = 0.01) -> ClaudeResponse:
    return ClaudeResponse(
        text=text,
        model=model,
        input_tokens=100,
        output_tokens=100,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        total_cost_usd=cost,
        duration_ms=1000,
    )


def _videos() -> list[VideoMeta]:
    return [
        VideoMeta(
            video_id=f"vid{i:03d}",
            title=f"Video {i}",
            url=f"https://www.youtube.com/watch?v=vid{i:03d}",
            duration=120,
            channel="Test",
            upload_date="20260418",
            playlist_title="Test Playlist",
        )
        for i in range(1, 4)
    ]


def _transcript_result(video_id: str):
    return build_result(
        video_id=video_id,
        source=TranscriptSource.OFFICIAL,
        language="ja",
        snippets=[
            TranscriptSnippet("字幕A", 0.0, 30.0),
            TranscriptSnippet("字幕B", 30.0, 30.0),
        ],
    )


def _capture_success() -> CaptureResult:
    return CaptureResult(
        ranges=[SummaryRange(0, 30, "intro"), SummaryRange(30, 60, "key")],
        outcomes=[],
        video_downloaded=True,
        capture_format="webp",
    )


def _stub_invoke_claude_factory():
    """Each call routes to the right canned body based on prompt content."""

    def _route(prompt: str, **kw):
        # Stage 05 agents are dispatched in sequence α→β→Leader; detect by
        # what the prompt/append system prompt contains.
        sp = kw.get("append_system_prompt") or kw.get("system_prompt") or ""
        if "トピック" in sp or "alpha" in sp.lower() or "topic_id" in prompt:
            return ALPHA_OUT
        if "chapters" in prompt and "index" not in sp:
            return BETA_OUT
        return None

    # Simpler: deterministic queue
    queue = [
        _fake_response(SUMMARY_OUTPUT, model="haiku", cost=0.01),  # vid1 summary
        _fake_response(LEARNING_OUTPUT, model="sonnet", cost=0.05),  # vid1 learning
        _fake_response(SUMMARY_OUTPUT, model="haiku", cost=0.01),  # vid2 summary
        _fake_response(LEARNING_OUTPUT, model="sonnet", cost=0.05),  # vid2 learning
        _fake_response(SUMMARY_OUTPUT, model="haiku", cost=0.01),  # vid3 summary
        _fake_response(LEARNING_OUTPUT, model="sonnet", cost=0.05),  # vid3 learning
        _fake_response(ALPHA_OUT, model="haiku", cost=0.02),
        _fake_response(BETA_OUT, model="sonnet", cost=0.03),
        # γ removed — coverage is now a Python set diff, no LLM call.
        _fake_response(LEADER_OUT, model="opus", cost=0.15),
    ]

    def fake_invoke(**kw):
        if not queue:
            pytest.fail("invoke_claude called more times than canned responses")
        return queue.pop(0)

    return fake_invoke


@pytest.fixture
def vault(tmp_path: Path):
    from pipeline_youtube import config

    (tmp_path / ".obsidian").mkdir()  # satisfy strict mode
    yield tmp_path
    config.reset_vault_root()


class TestE2EPlaylist:
    def test_full_cli_3_videos(self, vault: Path, monkeypatch):
        # Mock Stage 01 transcripts (bypass real youtube-transcript-api)
        def fake_scripts(video, path, *, dry_run, include_code_blocks=False):
            return _transcript_result(video.video_id)

        monkeypatch.setattr(main_mod, "run_stage_scripts", fake_scripts)

        # Mock fetch_metadata (no network)
        monkeypatch.setattr(main_mod, "fetch_metadata", lambda url: _videos())

        # Mock Stage 03 capture (no ffmpeg / yt-dlp)
        monkeypatch.setattr(main_mod, "run_stage_capture", lambda *a, **kw: _capture_success())
        monkeypatch.setattr(main_mod, "prefetch_video_download", lambda video: None)

        # SDK version: no claude binary validation needed
        monkeypatch.setattr(main_mod, "configure_providers", lambda *a, **kw: None)

        # Stub Router (genre classification) — avoid real LLM call
        from pipeline_youtube.genres import Genre

        monkeypatch.setattr(
            main_mod, "classify_playlist_genre", lambda *a, **kw: (Genre.OTHER, "stubbed")
        )

        # Stub every invoke_claude in both stages + synthesis
        from pipeline_youtube.stages import learning as learning_mod
        from pipeline_youtube.stages import summary as summary_mod

        fake_invoke = _stub_invoke_claude_factory()
        monkeypatch.setattr(summary_mod, "invoke_claude", fake_invoke)
        monkeypatch.setattr(learning_mod, "invoke_claude", fake_invoke)
        monkeypatch.setattr(agents_mod, "invoke_claude", fake_invoke)

        # Write a minimal config.json pointing at the vault
        cfg = vault / "config.json"
        cfg.write_text(json.dumps({"vault_root": str(vault)}), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            main_mod.cli,
            [
                "https://www.youtube.com/playlist?list=PL_fake",
                "--config",
                str(cfg),
                # Serial so the FIFO canned-response queue stays aligned with
                # the per-video call order (parallelism is covered separately).
                "--concurrency",
                "1",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        # Stages executed
        assert "[01] scripts" in result.output
        assert "[02] summary" in result.output
        assert "[03] capture" in result.output
        assert "[04] learning" in result.output
        # Stage 05 ran
        assert "Stage 05 Synthesis" in result.output
        # MOC + 1 chapter written
        moc = vault / "Permanent Note/08_YouTube学習/05_Synthesis"
        assert moc.exists()
        chapter_files = list(moc.rglob("01_*.md"))
        assert len(chapter_files) == 1
        # Cost breakdown appeared
        assert "Cost breakdown" in result.output
        assert "stage_02" in result.output
        assert "leader" in result.output

    def test_stop_after_capture_skips_04_and_05(self, vault: Path, monkeypatch):
        def fake_scripts(video, path, *, dry_run, include_code_blocks=False):
            return _transcript_result(video.video_id)

        monkeypatch.setattr(main_mod, "run_stage_scripts", fake_scripts)
        monkeypatch.setattr(main_mod, "fetch_metadata", lambda url: _videos())
        monkeypatch.setattr(main_mod, "run_stage_capture", lambda *a, **kw: _capture_success())
        monkeypatch.setattr(main_mod, "prefetch_video_download", lambda video: None)
        monkeypatch.setattr(main_mod, "configure_providers", lambda *a, **kw: None)
        from pipeline_youtube.genres import Genre
        from pipeline_youtube.stages import learning as learning_mod
        from pipeline_youtube.stages import summary as summary_mod

        monkeypatch.setattr(
            main_mod, "classify_playlist_genre", lambda *a, **kw: (Genre.OTHER, "stubbed")
        )

        invoke_count = {"n": 0}

        def fake_invoke(**kw):
            invoke_count["n"] += 1
            return _fake_response(SUMMARY_OUTPUT, model="haiku", cost=0.01)

        monkeypatch.setattr(summary_mod, "invoke_claude", fake_invoke)
        monkeypatch.setattr(learning_mod, "invoke_claude", fake_invoke)
        monkeypatch.setattr(agents_mod, "invoke_claude", fake_invoke)

        cfg = vault / "config.json"
        cfg.write_text(json.dumps({"vault_root": str(vault)}), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            main_mod.cli,
            [
                "https://www.youtube.com/playlist?list=PL_fake",
                "--config",
                str(cfg),
                "--stop-after-capture",
                "--concurrency",
                "1",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        # Only 3 Stage 02 calls — no Stage 04 or synthesis agents
        assert invoke_count["n"] == 3
        assert "stop-after-capture" in result.output
        assert "[04] learning" not in result.output
        assert "Stage 05 Synthesis" not in result.output
