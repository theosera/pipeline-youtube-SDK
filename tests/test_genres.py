"""Tests for the Genre router (Stage 00.5).

Verify:
1. Each known Genre value parses correctly.
2. The router prompt includes the playlist title and a sample of titles.
3. JSON output (with or without code fences) parses to the right Genre.
4. Errors (network, parse, unknown enum value) collapse to Genre.OTHER
   without raising.
5. CODE_BEARING_GENRES is the gate that downstream features check.
"""

from __future__ import annotations

import pytest

from pipeline_youtube.genres import (
    CODE_BEARING_GENRES,
    Genre,
    classify_playlist_genre,
)
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.providers import registry as claude_cli_mod
from pipeline_youtube.providers.base import LLMError as ClaudeCliError
from pipeline_youtube.providers.base import LLMResponse as ClaudeResponse


def _video(i: int, title: str) -> VideoMeta:
    vid = f"vid{i:03d}"
    return VideoMeta(
        video_id=vid,
        title=title,
        url=f"https://www.youtube.com/watch?v={vid}",
        duration=600,
        channel="Test",
        upload_date="20260420",
        playlist_title="Test Playlist",
    )


def _resp(text: str) -> ClaudeResponse:
    return ClaudeResponse(
        text=text,
        model="haiku",
        input_tokens=10,
        output_tokens=20,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        total_cost_usd=0.001,
        duration_ms=500,
    )


# =====================================================
# Genre enum + CODE_BEARING_GENRES
# =====================================================


class TestGenreEnum:
    def test_all_known_values(self):
        assert {g.value for g in Genre} == {
            "coding",
            "business",
            "humanities",
            "science",
            "lifestyle",
            "entertainment",
            "other",
        }

    def test_code_bearing_only_includes_coding(self):
        assert frozenset({Genre.CODING}) == CODE_BEARING_GENRES

    def test_genre_is_str_compatible(self):
        """Genre values should compare to plain strings (str enum semantics)."""
        assert Genre.CODING == "coding"
        assert Genre.HUMANITIES == "humanities"


# =====================================================
# classify_playlist_genre — happy paths
# =====================================================


class TestClassifyHappyPath:
    def test_coding_classification(self, monkeypatch):
        captured: dict = {}

        def fake_invoke(**kw):
            captured.update(kw)
            return _resp('{"genre": "coding", "rationale": "Claude Code チュートリアル"}')

        monkeypatch.setattr(claude_cli_mod, "invoke_llm", fake_invoke)
        # Re-import the binding the genres module uses
        from pipeline_youtube import genres as genres_mod

        monkeypatch.setattr(genres_mod, "invoke_claude", fake_invoke)

        videos = [
            _video(1, "Claude Code 入門"),
            _video(2, "Cursor で AI コーディング"),
        ]
        genre, rationale = classify_playlist_genre("AI 開発入門", videos)

        assert genre == Genre.CODING
        assert "Claude Code" in rationale

        # Prompt contains both playlist title and video titles
        prompt = captured["prompt"]
        assert "AI 開発入門" in prompt
        assert "Claude Code 入門" in prompt
        assert "Cursor" in prompt

        # System prompt enforces JSON-only output
        assert "JSON" in captured["system_prompt"]

    def test_humanities_classification(self, monkeypatch):
        from pipeline_youtube import genres as genres_mod

        monkeypatch.setattr(
            genres_mod,
            "invoke_claude",
            lambda **kw: _resp('{"genre": "humanities", "rationale": "哲学講義"}'),
        )

        videos = [_video(1, "ニーチェの永劫回帰"), _video(2, "カントの定言命法")]
        genre, _ = classify_playlist_genre("哲学入門", videos)
        assert genre == Genre.HUMANITIES

    def test_strips_code_fences(self, monkeypatch):
        """If the LLM ignores instructions and wraps JSON in ```, still parse."""
        from pipeline_youtube import genres as genres_mod

        monkeypatch.setattr(
            genres_mod,
            "invoke_claude",
            lambda **kw: _resp('```json\n{"genre": "science", "rationale": "x"}\n```'),
        )

        videos = [_video(1, "量子力学")]
        genre, _ = classify_playlist_genre("物理学", videos)
        assert genre == Genre.SCIENCE

    def test_uses_haiku_by_default(self, monkeypatch):
        captured: dict = {}
        from pipeline_youtube import genres as genres_mod

        def fake(**kw):
            captured.update(kw)
            return _resp('{"genre": "other", "rationale": ""}')

        monkeypatch.setattr(genres_mod, "invoke_claude", fake)
        classify_playlist_genre("p", [_video(1, "t")])
        assert captured["model"] == "haiku"

    def test_explicit_model_override(self, monkeypatch):
        captured: dict = {}
        from pipeline_youtube import genres as genres_mod

        def fake(**kw):
            captured.update(kw)
            return _resp('{"genre": "other", "rationale": ""}')

        monkeypatch.setattr(genres_mod, "invoke_claude", fake)
        classify_playlist_genre("p", [_video(1, "t")], model="sonnet")
        assert captured["model"] == "sonnet"


# =====================================================
# classify_playlist_genre — error paths
# =====================================================


class TestClassifyErrorPaths:
    def test_invoke_failure_returns_other(self, monkeypatch):
        from pipeline_youtube import genres as genres_mod

        def fake(**kw):
            raise ClaudeCliError("network down")

        monkeypatch.setattr(genres_mod, "invoke_claude", fake)
        genre, rationale = classify_playlist_genre("p", [_video(1, "t")])
        assert genre == Genre.OTHER
        assert "router_call_failed" in rationale

    def test_invalid_json_returns_other(self, monkeypatch):
        from pipeline_youtube import genres as genres_mod

        monkeypatch.setattr(genres_mod, "invoke_claude", lambda **kw: _resp("definitely not json"))
        genre, rationale = classify_playlist_genre("p", [_video(1, "t")])
        assert genre == Genre.OTHER
        assert "router_parse_failed" in rationale

    def test_unknown_genre_value_returns_other(self, monkeypatch):
        from pipeline_youtube import genres as genres_mod

        monkeypatch.setattr(
            genres_mod,
            "invoke_claude",
            lambda **kw: _resp('{"genre": "cooking", "rationale": "lol"}'),
        )
        genre, _ = classify_playlist_genre("p", [_video(1, "t")])
        assert genre == Genre.OTHER

    def test_missing_genre_key_returns_other(self, monkeypatch):
        from pipeline_youtube import genres as genres_mod

        monkeypatch.setattr(
            genres_mod,
            "invoke_claude",
            lambda **kw: _resp('{"category": "coding"}'),
        )
        genre, rationale = classify_playlist_genre("p", [_video(1, "t")])
        assert genre == Genre.OTHER
        assert "router_parse_failed" in rationale

    def test_empty_playlist_returns_other_without_call(self, monkeypatch):
        called: list[bool] = []
        from pipeline_youtube import genres as genres_mod

        def fake(**kw):
            called.append(True)
            return _resp('{"genre": "coding", "rationale": ""}')

        monkeypatch.setattr(genres_mod, "invoke_claude", fake)
        genre, rationale = classify_playlist_genre("p", [])
        assert genre == Genre.OTHER
        assert rationale == "no videos"
        assert not called  # no LLM call when there's nothing to classify


# =====================================================
# Sample size + sanitization
# =====================================================


class TestPromptShape:
    def test_truncates_to_30_titles(self, monkeypatch):
        captured: dict = {}
        from pipeline_youtube import genres as genres_mod

        def fake(**kw):
            captured.update(kw)
            return _resp('{"genre": "other", "rationale": ""}')

        monkeypatch.setattr(genres_mod, "invoke_claude", fake)

        videos = [_video(i, f"Video {i}") for i in range(50)]
        classify_playlist_genre("Big Playlist", videos)

        # Only first 30 titles in prompt
        prompt = captured["prompt"]
        assert "Video 0" in prompt
        assert "Video 29" in prompt
        assert "Video 30" not in prompt
        assert "計 50 本中、先頭 30 本を表示" in prompt

    def test_sanitizes_control_chars_in_titles(self, monkeypatch):
        captured: dict = {}
        from pipeline_youtube import genres as genres_mod

        def fake(**kw):
            captured.update(kw)
            return _resp('{"genre": "other", "rationale": ""}')

        monkeypatch.setattr(genres_mod, "invoke_claude", fake)

        videos = [_video(1, "title\x01with\x07control")]
        classify_playlist_genre("p", videos)
        prompt = captured["prompt"]
        assert "\x01" not in prompt
        assert "\x07" not in prompt

    @pytest.mark.parametrize(
        "title",
        ["", "Untitled​playlist"],
    )
    def test_handles_edge_case_titles(self, monkeypatch, title: str):
        from pipeline_youtube import genres as genres_mod

        monkeypatch.setattr(
            genres_mod,
            "invoke_claude",
            lambda **kw: _resp('{"genre": "other", "rationale": ""}'),
        )
        # Should not raise
        genre, _ = classify_playlist_genre(title, [_video(1, "t")])
        assert genre == Genre.OTHER
