"""Tests for stage 02 (summary) with LLM provider mocked."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from pipeline_youtube import config
from pipeline_youtube.pipeline import create_placeholder_notes
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.providers.base import LLMResponse as ClaudeResponse
from pipeline_youtube.stages import summary as summary_stage
from pipeline_youtube.transcript.base import (
    TranscriptSnippet,
    TranscriptSource,
    build_result,
)


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
        duration=932,
        channel="AI Channel",
        upload_date="20260414",
        playlist_title="Harness Engineering",
    )


def _transcript(snippets: list[TranscriptSnippet] | None = None):
    return build_result(
        video_id="_h3decBW12Q",
        source=TranscriptSource.OFFICIAL,
        language="ja",
        snippets=snippets or [],
    )


def _fake_claude_response(text: str) -> ClaudeResponse:
    return ClaudeResponse(
        text=text,
        model="sonnet",
        input_tokens=120,
        output_tokens=300,
        cache_creation_tokens=0,
        cache_read_tokens=19415,
        total_cost_usd=0.005,
        duration_ms=4500,
        session_id="fake-session",
        stop_reason="end_turn",
    )


SAMPLE_SUMMARY_OUTPUT = """## 全体サマリ
Anthropic が公開した長時間タスク向けハーネス設計の実験解説。GAN を参考にしたジェネレーター/エバリュエーター分離が中心概念。

## 要点タイムライン

### [00:00 ~ 01:30] 導入: ハーネスエンジニアリングとは
AI の能力を最大限引き出す環境整備の考え方。

### [01:56 ~ 03:32] 問題: コンテキスト不安
長時間タスクで文脈が埋まると AI が焦ってタスクを強引にまとめる。

### [04:04 ~ 04:55] 解決策: ジェネレーター/エバリュエーター分離
GAN を参考に生成と評価の役割を分離する。
"""


# =====================================================
# Happy path
# =====================================================


class TestRunStageSummary:
    def test_end_to_end_appends_body(self, vault, monkeypatch):
        video = _video()
        run_time = datetime(2026, 4, 14, 21, 41)
        paths = create_placeholder_notes(video, run_time, dry_run=False)
        summary_path = paths["summary"]

        transcript = _transcript(
            [
                TranscriptSnippet("ハーネスエンジニアリングの導入", 0.0, 30.0),
                TranscriptSnippet("コンテキスト不安の話", 116.0, 30.0),
                TranscriptSnippet("GAN方式の解決策", 244.0, 30.0),
            ]
        )

        monkeypatch.setattr(
            summary_stage,
            "invoke_claude",
            lambda **kw: _fake_claude_response(SAMPLE_SUMMARY_OUTPUT),
        )

        response = summary_stage.run_stage_summary(video, summary_path, transcript)

        assert response.text == SAMPLE_SUMMARY_OUTPUT
        assert response.input_tokens == 120
        assert response.cache_read_tokens == 19415

        post = summary_path.read_text(encoding="utf-8")
        assert post.startswith("---\n")
        assert "## 全体サマリ" in post
        assert "## 要点タイムライン" in post
        assert "[00:00 ~ 01:30]" in post
        assert "[01:56 ~ 03:32]" in post

    def test_empty_transcript_writes_placeholder(self, vault):
        video = _video()
        run_time = datetime(2026, 4, 14, 21, 41)
        paths = create_placeholder_notes(video, run_time, dry_run=False)
        summary_path = paths["summary"]
        transcript = _transcript([])

        response = summary_stage.run_stage_summary(video, summary_path, transcript)

        assert response.input_tokens is None  # synthetic, no API call
        post = summary_path.read_text(encoding="utf-8")
        assert "## 全体サマリ" in post
        assert "字幕を取得できませんでした" in post
        assert "(該当なし)" in post

    def test_dry_run_does_not_write_file(self, vault, monkeypatch):
        video = _video()
        run_time = datetime(2026, 4, 14, 21, 41)
        paths = create_placeholder_notes(video, run_time, dry_run=False)
        summary_path = paths["summary"]
        pre = summary_path.read_text(encoding="utf-8")

        transcript = _transcript([TranscriptSnippet("hi", 0.0, 5.0)])
        monkeypatch.setattr(
            summary_stage,
            "invoke_claude",
            lambda **kw: _fake_claude_response(SAMPLE_SUMMARY_OUTPUT),
        )

        summary_stage.run_stage_summary(video, summary_path, transcript, dry_run=True)

        assert summary_path.read_text(encoding="utf-8") == pre

    def test_missing_placeholder_raises(self, vault, monkeypatch):
        video = _video()
        ghost = config.get_vault_root() / "ghost.md"
        transcript = _transcript([TranscriptSnippet("hi", 0.0, 5.0)])
        monkeypatch.setattr(
            summary_stage,
            "invoke_claude",
            lambda **kw: _fake_claude_response(SAMPLE_SUMMARY_OUTPUT),
        )

        with pytest.raises(FileNotFoundError):
            summary_stage.run_stage_summary(video, ghost, transcript)


# =====================================================
# Prompt construction
# =====================================================


class TestPromptBuilding:
    def test_prompt_wraps_in_untrusted_content(self, vault, monkeypatch):
        video = _video()
        run_time = datetime(2026, 4, 14, 21, 41)
        paths = create_placeholder_notes(video, run_time, dry_run=False)
        transcript = _transcript([TranscriptSnippet("本文テキスト", 0.0, 30.0)])

        captured: dict = {}

        def fake_invoke(**kw):
            captured.update(kw)
            return _fake_claude_response(
                "## 全体サマリ\n\ntest\n\n## 要点タイムライン\n\n### [00:00 ~ 00:30] intro\n本文\n"
            )

        monkeypatch.setattr(summary_stage, "invoke_claude", fake_invoke)

        summary_stage.run_stage_summary(video, paths["summary"], transcript)

        prompt = captured["prompt"]
        assert "<untrusted_content>" in prompt
        assert "</untrusted_content>" in prompt
        assert "本文テキスト" in prompt
        assert video.title in prompt

    def test_prompt_uses_chunk_timestamps(self, vault, monkeypatch):
        video = _video()
        run_time = datetime(2026, 4, 14, 21, 41)
        paths = create_placeholder_notes(video, run_time, dry_run=False)
        # Three snippets spread across ~90s so chunking produces 3 chunks at 30s window
        transcript = _transcript(
            [
                TranscriptSnippet("intro", 0.0, 5.0),
                TranscriptSnippet("middle", 35.0, 5.0),
                TranscriptSnippet("later", 70.0, 5.0),
            ]
        )
        captured: dict = {}
        monkeypatch.setattr(
            summary_stage,
            "invoke_claude",
            lambda **kw: (
                captured.update(kw),
                _fake_claude_response(
                    "## 全体サマリ\n\nok\n\n## 要点タイムライン\n\n### [00:00 ~ 00:30] intro\n本文\n"
                ),
            )[1],
        )

        summary_stage.run_stage_summary(video, paths["summary"], transcript)

        prompt = captured["prompt"]
        assert "[00:00]" in prompt
        assert "[00:35]" in prompt
        assert "[01:10]" in prompt

    def test_system_prompt_is_append_mode(self, vault, monkeypatch):
        video = _video()
        run_time = datetime(2026, 4, 14, 21, 41)
        paths = create_placeholder_notes(video, run_time, dry_run=False)
        transcript = _transcript([TranscriptSnippet("hi", 0.0, 5.0)])
        captured: dict = {}
        monkeypatch.setattr(
            summary_stage,
            "invoke_claude",
            lambda **kw: (
                captured.update(kw),
                _fake_claude_response(
                    "## 全体サマリ\n\nok\n\n## 要点タイムライン\n\n### [00:00 ~ 00:30] intro\n本文\n"
                ),
            )[1],
        )

        summary_stage.run_stage_summary(video, paths["summary"], transcript)

        # Append mode preserves Claude Code default context.
        # (--system-prompt replace mode gave no cache savings in live runs.)
        assert "append_system_prompt" in captured
        assert captured.get("system_prompt") is None
        assert "YouTube" in captured["append_system_prompt"]
        assert "全体サマリ" in captured["append_system_prompt"]

    def test_sanitizes_control_chars_in_transcript(self, vault, monkeypatch):
        video = _video()
        run_time = datetime(2026, 4, 14, 21, 41)
        paths = create_placeholder_notes(video, run_time, dry_run=False)
        # Transcript contains zero-width and control chars
        evil_text = "normal\u200btext\x01with\x08nasties"
        transcript = _transcript([TranscriptSnippet(evil_text, 0.0, 30.0)])
        captured: dict = {}
        monkeypatch.setattr(
            summary_stage,
            "invoke_claude",
            lambda **kw: (
                captured.update(kw),
                _fake_claude_response(
                    "## 全体サマリ\n\nok\n\n## 要点タイムライン\n\n### [00:00 ~ 00:30] intro\n本文\n"
                ),
            )[1],
        )

        summary_stage.run_stage_summary(video, paths["summary"], transcript)

        prompt = captured["prompt"]
        assert "\x01" not in prompt
        assert "\x08" not in prompt
        assert "\u200b" not in prompt
        assert "normaltextwithnasties" in prompt
