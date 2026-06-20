"""Tests for the code-bearing addendum in Stage 04 learning prompt.

Verify that:
1. When ``code_bearing=False`` (default) the system prompt is exactly
   the base ``LEARNING_SYSTEM_PROMPT``.
2. When ``code_bearing=True`` the addendum is appended and instructs
   the model to split output into `# 概念` / `# 実践`.
3. The addendum mentions the ``## 関連コード`` section produced by
   Stage 01 so the Leader can reference fetched GitHub snippets.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.providers.base import LLMResponse as ClaudeResponse
from pipeline_youtube.services.cache import Cache
from pipeline_youtube.stages import learning as learning_mod
from pipeline_youtube.stages.learning import (
    LEARNING_CODE_BEARING_ADDENDUM,
    LEARNING_SYSTEM_PROMPT,
    run_stage_learning,
)

_NO_CACHE = Cache(None, enabled=False)


@pytest.fixture
def vault(tmp_path: Path, monkeypatch):
    """Minimal vault setup + a pair of 02/03 md files."""
    from pipeline_youtube import config as cfg

    cfg.set_dry_run(False)
    yield tmp_path


def _video() -> VideoMeta:
    return VideoMeta(
        video_id="vid001",
        title="Claude Code チュートリアル",
        url="https://www.youtube.com/watch?v=vid001",
        duration=600,
        channel="test",
        upload_date="20260420",
        playlist_title="Test Playlist",
    )


def _write_stub(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nfoo: bar\n---\n\n{body}\n", encoding="utf-8")


def _fake_response(
    text: str = "## 概念: test\n[00:00 ~ 01:00]\n![[x.webp]]\n- ok",
) -> ClaudeResponse:
    return ClaudeResponse(
        text=text,
        model="sonnet",
        input_tokens=10,
        output_tokens=20,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        total_cost_usd=0.01,
        duration_ms=1000,
    )


class TestCodeBearingAddendum:
    def test_default_uses_base_prompt_only(self, vault: Path):
        summary_path = vault / "summary.md"
        capture_path = vault / "capture.md"
        learning_path = vault / "learning.md"
        _write_stub(summary_path, "sample summary")
        _write_stub(capture_path, "[00:00 ~ 01:00]\n![[x.webp]]")

        captured: dict = {}

        def fake_invoke(**kw):
            captured.update(kw)
            return _fake_response()

        with patch.object(learning_mod, "invoke_claude", fake_invoke):
            run_stage_learning(
                _video(),
                summary_path,
                capture_path,
                learning_path,
                run_time=datetime(2026, 4, 21, 12, 0),
                model="sonnet",
                cache=_NO_CACHE,
            )

        assert captured["append_system_prompt"] == LEARNING_SYSTEM_PROMPT
        assert "# 概念 (Concepts)" not in captured["append_system_prompt"]

    def test_code_bearing_appends_addendum(self, vault: Path):
        summary_path = vault / "summary.md"
        capture_path = vault / "capture.md"
        learning_path = vault / "learning.md"
        _write_stub(summary_path, "sample summary")
        _write_stub(capture_path, "[00:00 ~ 01:00]\n![[x.webp]]")

        captured: dict = {}

        def fake_invoke(**kw):
            captured.update(kw)
            return _fake_response()

        with patch.object(learning_mod, "invoke_claude", fake_invoke):
            run_stage_learning(
                _video(),
                summary_path,
                capture_path,
                learning_path,
                run_time=datetime(2026, 4, 21, 12, 0),
                model="sonnet",
                code_bearing=True,
                cache=_NO_CACHE,
            )

        sp = captured["append_system_prompt"]
        # Base prompt still present
        assert LEARNING_SYSTEM_PROMPT in sp
        # Addendum appended verbatim
        assert LEARNING_CODE_BEARING_ADDENDUM in sp
        # Key instructions from the addendum are visible
        assert "# 概念 (Concepts)" in sp
        assert "# 実践 (Practice)" in sp
        # Mentions the GitHub snippet section from Stage 01
        assert "## 関連コード" in sp

    def test_addendum_lists_concept_and_practice_types(self):
        """Sanity check that the category partition is spelled out."""
        # concept categories
        for cat in ("概念", "問題", "結果", "まとめ", "背景"):
            assert cat in LEARNING_CODE_BEARING_ADDENDUM
        # practice categories
        for cat in ("解決策", "実装手順", "コマンド", "コード例", "セットアップ", "デバッグ"):
            assert cat in LEARNING_CODE_BEARING_ADDENDUM
