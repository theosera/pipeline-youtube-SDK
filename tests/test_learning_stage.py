"""Tests for stage 04 (learning material) with LLM provider mocked."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from pipeline_youtube import config
from pipeline_youtube.pipeline import compute_note_paths, create_placeholder_notes
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.providers.base import LLMResponse as ClaudeResponse
from pipeline_youtube.services.cache import Cache
from pipeline_youtube.stages import learning as learning_stage
from pipeline_youtube.stages.learning import _strip_frontmatter

_NO_CACHE = Cache(None, enabled=False)


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


SAMPLE_SUMMARY_BODY = """## 全体サマリ
Anthropic が公開した長時間タスク向けハーネス設計の実験解説。

## 要点タイムライン

### [00:00 ~ 01:03] ハーネスエンジニアリングとは
AI の能力を最大限引き出す環境整備の考え方。

### [01:03 ~ 02:50] 問題: コンテキストフア
コンテキストウィンドウが埋まると AI が焦ってまとめる。

### [03:26 ~ 05:06] 解決策: GAN 方式
生成役と評価役を分離する。
"""


SAMPLE_CAPTURE_BODY = """[00:00 ~ 01:03]
![[2026-04-14-2141 test.webp]]

[01:03 ~ 02:50]
![[2026-04-14-2141 test-1.webp]]

[03:26 ~ 05:06]
![[2026-04-14-2141 test-2.webp]]
"""


SAMPLE_LEARNING_BODY = """## 概念: ハーネスエンジニアリングとは
[00:00 ~ 01:03]
![[2026-04-14-2141 test.webp]]
- AI の能力を最大限引き出すための環境整備
- Claude Code では CLAUDE.md やガードレールを整備する

## 問題: コンテキストフア
[01:03 ~ 02:50]
![[2026-04-14-2141 test-1.webp]]
- コンテキストウィンドウが埋まると AI が焦る
- 品質が後半で急激に落ちる

## 解決策: GAN 方式のジェネレーター/エバリュエーター分離
[03:26 ~ 05:06]
![[2026-04-14-2141 test-2.webp]]
- 生成役と評価役を別エージェントに分離
- 自己評価の甘さを避ける
"""


def _fake_claude_response(text: str) -> ClaudeResponse:
    return ClaudeResponse(
        text=text,
        model="sonnet",
        input_tokens=5,
        output_tokens=400,
        cache_creation_tokens=0,
        cache_read_tokens=22000,
        total_cost_usd=0.004,
        duration_ms=5800,
        session_id="fake-session",
        stop_reason="end_turn",
    )


def _setup_vault(vault: Path):
    """Create 01/02/03 placeholders and seed 02/03 with sample content."""
    video = _video()
    run_time = datetime(2026, 4, 14, 21, 41)
    paths = create_placeholder_notes(video, run_time, vault_root=config.get_vault_root())

    paths["summary"].write_text(
        paths["summary"].read_text() + "\n" + SAMPLE_SUMMARY_BODY,
        encoding="utf-8",
    )
    paths["capture"].write_text(
        paths["capture"].read_text() + "\n" + SAMPLE_CAPTURE_BODY,
        encoding="utf-8",
    )
    learning_path = compute_note_paths(
        video, run_time, units=("learning",), vault_root=config.get_vault_root()
    )["learning"]
    return video, run_time, paths, learning_path


# =====================================================
# Helpers
# =====================================================


class TestStripFrontmatter:
    def test_removes_yaml_block(self):
        text = '---\ntitle: "x"\ntags: [a]\n---\n\nbody here\n'
        # Trailing newline is preserved (lstrip only, not strip)
        assert _strip_frontmatter(text) == "body here\n"

    def test_no_frontmatter_returns_as_is(self):
        text = "plain body\nwith lines\n"
        assert _strip_frontmatter(text) == "plain body\nwith lines"

    def test_malformed_frontmatter_returns_as_is(self):
        text = "---\nno closing delimiter\n"
        assert _strip_frontmatter(text) == text.strip()


# =====================================================
# End-to-end with LLM provider mocked
# =====================================================


class TestRunStageLearning:
    def test_happy_path_writes_04_md(self, vault, monkeypatch):
        video, run_time, paths, learning_path = _setup_vault(vault)

        monkeypatch.setattr(
            learning_stage,
            "invoke_claude",
            lambda **kw: _fake_claude_response(SAMPLE_LEARNING_BODY),
        )

        response = learning_stage.run_stage_learning(
            video,
            summary_md_path=paths["summary"],
            capture_md_path=paths["capture"],
            learning_md_path=learning_path,
            run_time=run_time,
            cache=_NO_CACHE,
        )

        assert response.text == SAMPLE_LEARNING_BODY
        assert learning_path.exists()
        content = learning_path.read_text(encoding="utf-8")
        assert content.startswith("---\n")
        assert 'title: "Anthropicが公開したハーネス設計、全部解説します"' in content
        assert 'playlist: "Harness Engineering"' in content
        assert 'video_id: "_h3decBW12Q"' in content
        assert "## 概念: ハーネスエンジニアリングとは" in content
        assert "![[2026-04-14-2141 test.webp]]" in content

    def test_writes_full_file_atomically(self, vault, monkeypatch):
        """No empty-file window: the 04 md exists only with full content."""
        video, run_time, paths, learning_path = _setup_vault(vault)
        assert not learning_path.exists()  # pre-check

        monkeypatch.setattr(
            learning_stage,
            "invoke_claude",
            lambda **kw: _fake_claude_response(SAMPLE_LEARNING_BODY),
        )

        learning_stage.run_stage_learning(
            video,
            summary_md_path=paths["summary"],
            capture_md_path=paths["capture"],
            learning_md_path=learning_path,
            run_time=run_time,
            cache=_NO_CACHE,
        )

        # The file exists with full frontmatter + body in a single write
        assert learning_path.exists()
        content = learning_path.read_text(encoding="utf-8")
        assert content.startswith("---\n")
        assert content.endswith("\n")
        # Body is present (no empty-file intermediate state)
        assert len(content) > 500

    def test_dry_run_does_not_write_file(self, vault, monkeypatch):
        video, run_time, paths, learning_path = _setup_vault(vault)

        monkeypatch.setattr(
            learning_stage,
            "invoke_claude",
            lambda **kw: _fake_claude_response(SAMPLE_LEARNING_BODY),
        )

        response = learning_stage.run_stage_learning(
            video,
            summary_md_path=paths["summary"],
            capture_md_path=paths["capture"],
            learning_md_path=learning_path,
            run_time=run_time,
            dry_run=True,
            cache=_NO_CACHE,
        )

        assert response.text == SAMPLE_LEARNING_BODY
        assert not learning_path.exists()

    def test_missing_summary_raises(self, vault, monkeypatch):
        video, run_time, paths, learning_path = _setup_vault(vault)
        paths["summary"].unlink()

        with pytest.raises(FileNotFoundError, match="summary md"):
            learning_stage.run_stage_learning(
                video,
                summary_md_path=paths["summary"],
                capture_md_path=paths["capture"],
                learning_md_path=learning_path,
                run_time=run_time,
                cache=_NO_CACHE,
            )

    def test_missing_capture_raises(self, vault, monkeypatch):
        video, run_time, paths, learning_path = _setup_vault(vault)
        paths["capture"].unlink()

        with pytest.raises(FileNotFoundError, match="capture md"):
            learning_stage.run_stage_learning(
                video,
                summary_md_path=paths["summary"],
                capture_md_path=paths["capture"],
                learning_md_path=learning_path,
                run_time=run_time,
                cache=_NO_CACHE,
            )


class TestPromptBuilding:
    def test_prompt_includes_both_inputs_in_untrusted_content(self, vault, monkeypatch):
        video, run_time, paths, learning_path = _setup_vault(vault)
        captured: dict = {}

        def fake_invoke(**kw):
            captured.update(kw)
            return _fake_claude_response(SAMPLE_LEARNING_BODY)

        monkeypatch.setattr(learning_stage, "invoke_claude", fake_invoke)

        learning_stage.run_stage_learning(
            video,
            summary_md_path=paths["summary"],
            capture_md_path=paths["capture"],
            learning_md_path=learning_path,
            run_time=run_time,
            cache=_NO_CACHE,
        )

        prompt = captured["prompt"]
        # Both sections wrapped in untrusted_content
        assert prompt.count("<untrusted_content>") == 2
        assert prompt.count("</untrusted_content>") == 2
        # Content of both inputs is present
        assert "ハーネスエンジニアリングとは" in prompt  # from summary
        assert "2026-04-14-2141 test.webp" in prompt  # from capture
        # Title is present
        assert video.title in prompt

    def test_prompt_uses_append_system_prompt(self, vault, monkeypatch):
        video, run_time, paths, learning_path = _setup_vault(vault)
        captured: dict = {}
        monkeypatch.setattr(
            learning_stage,
            "invoke_claude",
            lambda **kw: (captured.update(kw), _fake_claude_response(SAMPLE_LEARNING_BODY))[1],
        )

        learning_stage.run_stage_learning(
            video,
            summary_md_path=paths["summary"],
            capture_md_path=paths["capture"],
            learning_md_path=learning_path,
            run_time=run_time,
            cache=_NO_CACHE,
        )

        assert "append_system_prompt" in captured
        assert captured.get("system_prompt") is None
        assert "学習ノート" in captured["append_system_prompt"]

    def test_frontmatter_is_stripped_from_inputs(self, vault, monkeypatch):
        """Only the body of 02/03 should feed into the prompt, not their frontmatter."""
        video, run_time, paths, learning_path = _setup_vault(vault)
        captured: dict = {}
        monkeypatch.setattr(
            learning_stage,
            "invoke_claude",
            lambda **kw: (captured.update(kw), _fake_claude_response(SAMPLE_LEARNING_BODY))[1],
        )

        learning_stage.run_stage_learning(
            video,
            summary_md_path=paths["summary"],
            capture_md_path=paths["capture"],
            learning_md_path=learning_path,
            run_time=run_time,
            cache=_NO_CACHE,
        )

        prompt = captured["prompt"]
        # Frontmatter fields should NOT appear in the prompt
        assert "tags: [memo, youtube]" not in prompt
        assert 'URL: "https://www.youtube.com/' not in prompt
