"""Tests for backward-compatible re-exports and domain package public API.

Verifies that:
1. pipeline_youtube.domain exports all expected names via __all__
2. pipeline_youtube.playlist still exposes VideoMeta (backward compat)
3. pipeline_youtube.run_result still exposes VideoRunResult (backward compat)
4. pipeline_youtube.transcript.base still exposes all transcript types (backward compat)
5. The re-exported symbols are the same objects as those in domain/
"""

from __future__ import annotations

import pytest

import pipeline_youtube.domain as domain_pkg
import pipeline_youtube.playlist as playlist_mod
import pipeline_youtube.run_result as run_result_mod
import pipeline_youtube.transcript.base as transcript_base_mod
from pipeline_youtube.domain.results import VideoRunResult as DomainVideoRunResult
from pipeline_youtube.domain.transcript import (
    TranscriptNotAvailable as DomainTranscriptNotAvailable,
)
from pipeline_youtube.domain.transcript import (
    TranscriptResult as DomainTranscriptResult,
)
from pipeline_youtube.domain.transcript import (
    TranscriptSnippet as DomainTranscriptSnippet,
)
from pipeline_youtube.domain.transcript import (
    TranscriptSource as DomainTranscriptSource,
)
from pipeline_youtube.domain.video import VideoMeta as DomainVideoMeta

# ---------------------------------------------------------------------------
# domain/__init__.py public API
# ---------------------------------------------------------------------------


class TestDomainPackageAll:
    def test_all_contains_video_meta(self):
        assert "VideoMeta" in domain_pkg.__all__

    def test_all_contains_video_run_result(self):
        assert "VideoRunResult" in domain_pkg.__all__

    def test_all_contains_transcript_source(self):
        assert "TranscriptSource" in domain_pkg.__all__

    def test_all_contains_transcript_not_available(self):
        assert "TranscriptNotAvailable" in domain_pkg.__all__

    def test_all_contains_transcript_snippet(self):
        assert "TranscriptSnippet" in domain_pkg.__all__

    def test_all_contains_transcript_result(self):
        assert "TranscriptResult" in domain_pkg.__all__

    def test_all_contains_domain_errors(self):
        # SDK keeps the provider exception (LLMError) in the provider layer, so
        # ClaudeBinaryError is not part of the SDK domain surface.
        for name in (
            "VaultRootError",
            "SynthesisParseError",
            "GlossaryParseError",
            "GlossaryConflictError",
        ):
            assert name in domain_pkg.__all__

    def test_all_has_expected_names(self):
        assert len(domain_pkg.__all__) == 10

    def test_video_meta_accessible(self):
        assert domain_pkg.VideoMeta is DomainVideoMeta

    def test_video_run_result_accessible(self):
        assert domain_pkg.VideoRunResult is DomainVideoRunResult

    def test_transcript_source_accessible(self):
        assert domain_pkg.TranscriptSource is DomainTranscriptSource

    def test_transcript_not_available_accessible(self):
        assert domain_pkg.TranscriptNotAvailable is DomainTranscriptNotAvailable

    def test_transcript_snippet_accessible(self):
        assert domain_pkg.TranscriptSnippet is DomainTranscriptSnippet

    def test_transcript_result_accessible(self):
        assert domain_pkg.TranscriptResult is DomainTranscriptResult


# ---------------------------------------------------------------------------
# playlist.py backward-compat re-export
# ---------------------------------------------------------------------------


class TestPlaylistVideoMetaReexport:
    def test_video_meta_available(self):
        assert hasattr(playlist_mod, "VideoMeta")

    def test_video_meta_is_same_class(self):
        # Must be the same class object (not a copy)
        assert playlist_mod.VideoMeta is DomainVideoMeta

    def test_video_meta_constructible_from_playlist(self):
        v = playlist_mod.VideoMeta(
            video_id="abc123",
            title="Test",
            url="https://www.youtube.com/watch?v=abc123",
            duration=60,
            channel="Chan",
            upload_date="20240101",
            playlist_title=None,
        )
        assert v.video_id == "abc123"
        assert v.watch_url == "https://www.youtube.com/watch?v=abc123"

    def test_instance_from_playlist_is_domain_type(self):
        v = playlist_mod.VideoMeta(
            video_id="xyz",
            title="T",
            url="u",
            duration=None,
            channel=None,
            upload_date=None,
            playlist_title=None,
        )
        assert isinstance(v, DomainVideoMeta)


# ---------------------------------------------------------------------------
# run_result.py backward-compat re-export
# ---------------------------------------------------------------------------


class TestRunResultVideoRunResultReexport:
    def test_video_run_result_available(self):
        assert hasattr(run_result_mod, "VideoRunResult")

    def test_video_run_result_is_same_class(self):
        assert run_result_mod.VideoRunResult is DomainVideoRunResult

    def test_video_run_result_constructible_from_run_result(self):
        video = DomainVideoMeta(
            video_id="test",
            title="T",
            url="u",
            duration=None,
            channel=None,
            upload_date=None,
            playlist_title=None,
        )
        r = run_result_mod.VideoRunResult(video=video, learning_md_body="# hi")
        assert r.ok is True

    def test_instance_from_run_result_is_domain_type(self):
        video = DomainVideoMeta(
            video_id="vid",
            title="T",
            url="u",
            duration=None,
            channel=None,
            upload_date=None,
            playlist_title=None,
        )
        r = run_result_mod.VideoRunResult(video=video)
        assert isinstance(r, DomainVideoRunResult)


# ---------------------------------------------------------------------------
# transcript/base.py backward-compat re-exports
# ---------------------------------------------------------------------------


class TestTranscriptBaseReexports:
    def test_transcript_source_available(self):
        assert hasattr(transcript_base_mod, "TranscriptSource")

    def test_transcript_not_available_available(self):
        assert hasattr(transcript_base_mod, "TranscriptNotAvailable")

    def test_transcript_snippet_available(self):
        assert hasattr(transcript_base_mod, "TranscriptSnippet")

    def test_transcript_result_available(self):
        assert hasattr(transcript_base_mod, "TranscriptResult")

    def test_transcript_source_is_same_class(self):
        assert transcript_base_mod.TranscriptSource is DomainTranscriptSource

    def test_transcript_not_available_is_same_class(self):
        assert transcript_base_mod.TranscriptNotAvailable is DomainTranscriptNotAvailable

    def test_transcript_snippet_is_same_class(self):
        assert transcript_base_mod.TranscriptSnippet is DomainTranscriptSnippet

    def test_transcript_result_is_same_class(self):
        assert transcript_base_mod.TranscriptResult is DomainTranscriptResult

    def test_dunder_all_contains_transcript_source(self):
        assert "TranscriptSource" in transcript_base_mod.__all__

    def test_dunder_all_contains_transcript_not_available(self):
        assert "TranscriptNotAvailable" in transcript_base_mod.__all__

    def test_dunder_all_contains_transcript_snippet(self):
        assert "TranscriptSnippet" in transcript_base_mod.__all__

    def test_dunder_all_contains_transcript_result(self):
        assert "TranscriptResult" in transcript_base_mod.__all__

    def test_dunder_all_contains_fetcher(self):
        assert "Fetcher" in transcript_base_mod.__all__

    def test_dunder_all_contains_build_result(self):
        assert "build_result" in transcript_base_mod.__all__

    def test_dunder_all_contains_fetch_with_fallback(self):
        assert "fetch_with_fallback" in transcript_base_mod.__all__

    def test_instances_interchangeable(self):
        # An instance created via re-exported class is accepted by domain isinstance check
        snippet = transcript_base_mod.TranscriptSnippet(text="hi", start=0.0, duration=1.0)
        assert isinstance(snippet, DomainTranscriptSnippet)

    def test_exception_raised_via_base_caught_as_domain(self):
        # TranscriptNotAvailable raised via base module can be caught as domain type
        with pytest.raises(DomainTranscriptNotAvailable):
            raise transcript_base_mod.TranscriptNotAvailable("tier failed")
