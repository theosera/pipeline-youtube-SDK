"""End-to-end tests for stage 01 (scripts).

The transcript fetcher chain is replaced with stubs so these tests
run offline. We verify the markdown body format matches the dummy data
in `Permanent Note/08_YouTube学習/01_Scripts_Processing_Unit/`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from pipeline_youtube import config
from pipeline_youtube.pipeline import create_placeholder_notes
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.services.cache import Cache
from pipeline_youtube.stages import scripts as scripts_stage
from pipeline_youtube.transcript.base import (
    TranscriptSnippet,
    TranscriptSource,
    build_result,
)

# These stage-01 tests stub the transcript chain; caching is verified separately
# (test_injected_cache_*), so they thread a disabled (no-op) cache.
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
        duration=932,
        channel="AI Channel",
        upload_date="20260414",
        playlist_title="Harness Engineering",
    )


def _fake_fetch_success(source: TranscriptSource = TranscriptSource.OFFICIAL):
    def _fetch(video_id, languages, **kw):
        # Return a deterministic transcript matching dummy-data cadence
        return build_result(
            video_id=video_id,
            source=source,
            language="ja",
            snippets=[
                TranscriptSnippet("本さん、最近ハーネス", 0.0, 2.0),
                TranscriptSnippet("エンジニアリングが話題", 2.0, 3.0),
                TranscriptSnippet("実験して結果を公開", 32.0, 3.0),
                TranscriptSnippet("してくれたんです", 35.0, 2.0),
                TranscriptSnippet("MDEですね、ルールズ", 67.0, 3.0),
            ],
        )

    return _fetch


class TestRunStageScripts:
    def test_end_to_end_writes_formatted_body(self, vault, monkeypatch):
        video = _video()
        run_time = datetime(2026, 4, 14, 21, 41)
        paths = create_placeholder_notes(
            video, run_time, dry_run=False, vault_root=config.get_vault_root()
        )
        scripts_path = paths["scripts"]

        # Ensure placeholder has frontmatter but no body
        pre = scripts_path.read_text(encoding="utf-8")
        assert pre.startswith("---\n")
        assert "[00:00]" not in pre

        # Patch the fallback chain to return a fake result
        monkeypatch.setattr(
            scripts_stage,
            "fetch_with_fallback",
            lambda video_id, languages, fetchers, **kw: _fake_fetch_success()(video_id, languages),
        )

        result = scripts_stage.run_stage_scripts(
            video, scripts_path, window_seconds=30.0, cache=_NO_CACHE
        )

        assert result.source == TranscriptSource.OFFICIAL

        post = scripts_path.read_text(encoding="utf-8")
        # Frontmatter still present
        assert post.startswith("---\n")
        # Body has chunk lines
        assert "[00:00](https://www.youtube.com/watch?v=_h3decBW12Q&t=0)" in post
        assert "[00:32](https://www.youtube.com/watch?v=_h3decBW12Q&t=32)" in post
        assert "[01:07](https://www.youtube.com/watch?v=_h3decBW12Q&t=67)" in post
        assert "本さん、最近ハーネス エンジニアリングが話題" in post

    def test_correction_feeds_corrected_text_downstream(self, vault, monkeypatch):
        """When correct_model is set, the corrected text must land in the
        returned TranscriptResult.snippets (consumed by Stage 02), not only in
        the rendered 01 markdown."""
        from pipeline_youtube.transcript.chunking import Chunk
        from pipeline_youtube.transcript.correction import CorrectionResult

        video = _video()
        run_time = datetime(2026, 4, 14, 21, 41)
        paths = create_placeholder_notes(
            video, run_time, dry_run=False, vault_root=config.get_vault_root()
        )
        scripts_path = paths["scripts"]

        monkeypatch.setattr(
            scripts_stage,
            "fetch_with_fallback",
            lambda video_id, languages, fetchers, **kw: _fake_fetch_success()(video_id, languages),
        )
        seen: dict[str, object] = {}

        def _fake_correct(chunks, *, model, known_terms=None, cache=None):
            seen["known_terms"] = known_terms
            return CorrectionResult(
                chunks=[Chunk(start=c.start, text=c.text + " [FIX]") for c in chunks],
                cost_usd=0.42,
                confirmed_terms=["Anthropic"],
            )

        monkeypatch.setattr(scripts_stage, "correct_chunks", _fake_correct)

        result = scripts_stage.run_stage_scripts(
            video,
            scripts_path,
            window_seconds=30.0,
            correct_model="opus",
            known_terms=[("ぐぐる", "Google")],
            cache=_NO_CACHE,
        )

        assert result.snippets
        assert all("[FIX]" in s.text for s in result.snippets)
        assert result.snippets[0].start == 0.0
        assert result.correction_cost_usd == 0.42
        assert result.confirmed_terms == ("Anthropic",)
        assert "[FIX]" in scripts_path.read_text(encoding="utf-8")
        # The sheet's known terms must reach correct_chunks (web-search skip path).
        assert seen["known_terms"] == [("ぐぐる", "Google")]

    def test_dry_run_does_not_touch_file(self, vault, monkeypatch):
        video = _video()
        run_time = datetime(2026, 4, 14, 21, 41)
        paths = create_placeholder_notes(
            video, run_time, dry_run=False, vault_root=config.get_vault_root()
        )
        scripts_path = paths["scripts"]
        pre_content = scripts_path.read_text(encoding="utf-8")

        monkeypatch.setattr(
            scripts_stage,
            "fetch_with_fallback",
            lambda video_id, languages, fetchers, **kw: _fake_fetch_success()(video_id, languages),
        )

        result = scripts_stage.run_stage_scripts(
            video, scripts_path, window_seconds=30.0, dry_run=True, cache=_NO_CACHE
        )

        assert result.source == TranscriptSource.OFFICIAL
        assert scripts_path.read_text(encoding="utf-8") == pre_content

    def test_empty_transcript_writes_nothing(self, vault, monkeypatch):
        video = _video()
        run_time = datetime(2026, 4, 14, 21, 41)
        paths = create_placeholder_notes(
            video, run_time, dry_run=False, vault_root=config.get_vault_root()
        )
        scripts_path = paths["scripts"]
        pre_content = scripts_path.read_text(encoding="utf-8")

        def _empty_fetch(video_id, languages, fetchers, **kw):
            return build_result(
                video_id=video_id,
                source=TranscriptSource.ERROR,
                language=None,
                snippets=[],
            )

        monkeypatch.setattr(scripts_stage, "fetch_with_fallback", _empty_fetch)

        result = scripts_stage.run_stage_scripts(video, scripts_path, cache=_NO_CACHE)
        assert result.source == TranscriptSource.ERROR
        assert scripts_path.read_text(encoding="utf-8") == pre_content

    def test_missing_placeholder_raises(self, vault, monkeypatch):
        video = _video()
        ghost_path = config.get_vault_root() / "does_not_exist.md"

        monkeypatch.setattr(
            scripts_stage,
            "fetch_with_fallback",
            lambda video_id, languages, fetchers, **kw: _fake_fetch_success()(video_id, languages),
        )

        with pytest.raises(FileNotFoundError):
            scripts_stage.run_stage_scripts(video, ghost_path, cache=_NO_CACHE)

    def test_injected_cache_reaches_transcript_and_correction(self, vault, monkeypatch):
        """DI: the cache passed to run_stage_scripts is forwarded verbatim to the
        transcript fallback chain and the Stage 01b correction call."""
        from pipeline_youtube.services.cache import Cache
        from pipeline_youtube.transcript.correction import CorrectionResult

        video = _video()
        run_time = datetime(2026, 4, 14, 21, 41)
        paths = create_placeholder_notes(
            video, run_time, dry_run=False, vault_root=config.get_vault_root()
        )
        scripts_path = paths["scripts"]

        injected = Cache(config.get_vault_root() / "cache", enabled=True)
        seen: dict[str, object] = {}

        def _spy_fetch(video_id, languages, fetchers, *, cache=None):
            seen["fetch_cache"] = cache
            return _fake_fetch_success()(video_id, languages)

        def _spy_correct(chunks, *, model, known_terms=None, cache=None):
            seen["correct_cache"] = cache
            return CorrectionResult(chunks=list(chunks), cost_usd=0.0)

        monkeypatch.setattr(scripts_stage, "fetch_with_fallback", _spy_fetch)
        monkeypatch.setattr(scripts_stage, "correct_chunks", _spy_correct)

        scripts_stage.run_stage_scripts(
            video, scripts_path, window_seconds=30.0, correct_model="opus", cache=injected
        )

        assert seen["fetch_cache"] is injected
        assert seen["correct_cache"] is injected
