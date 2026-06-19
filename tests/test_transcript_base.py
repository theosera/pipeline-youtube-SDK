"""Tests for transcript fallback chain logic in base.py."""

from __future__ import annotations

from pipeline_youtube.transcript.base import (
    TranscriptNotAvailable,
    TranscriptResult,
    TranscriptSnippet,
    TranscriptSource,
    build_result,
    fetch_with_fallback,
)


def _stub_success(source: TranscriptSource):
    def fetcher(video_id: str, languages: list[str]) -> TranscriptResult:
        return build_result(
            video_id=video_id,
            source=source,
            language=languages[0] if languages else "ja",
            snippets=[TranscriptSnippet(text="hi", start=0.0, duration=1.0)],
        )

    return fetcher


def _stub_failure(reason: str):
    def fetcher(video_id: str, languages: list[str]) -> TranscriptResult:
        raise TranscriptNotAvailable(reason)

    return fetcher


class TestFetchWithFallback:
    def test_first_tier_succeeds(self):
        result = fetch_with_fallback(
            "vid1",
            ["ja"],
            [
                ("official", _stub_success(TranscriptSource.OFFICIAL)),
                ("auto", _stub_success(TranscriptSource.AUTO)),
            ],
        )
        assert result.source == TranscriptSource.OFFICIAL
        assert result.fallback_reason is None
        assert result.video_id == "vid1"

    def test_falls_back_to_second_tier(self):
        result = fetch_with_fallback(
            "vid1",
            ["ja"],
            [
                ("official", _stub_failure("no_manual_transcript_in_languages")),
                ("auto", _stub_success(TranscriptSource.AUTO)),
            ],
        )
        assert result.source == TranscriptSource.AUTO
        assert result.fallback_reason == "official:no_manual_transcript_in_languages"

    def test_falls_back_to_third_tier(self):
        result = fetch_with_fallback(
            "vid1",
            ["ja"],
            [
                ("official", _stub_failure("disabled")),
                ("auto", _stub_failure("disabled")),
                ("whisper", _stub_success(TranscriptSource.WHISPER)),
            ],
        )
        assert result.source == TranscriptSource.WHISPER
        assert "official:disabled" in (result.fallback_reason or "")
        assert "auto:disabled" in (result.fallback_reason or "")

    def test_all_tiers_fail_returns_error_result(self):
        result = fetch_with_fallback(
            "vid1",
            ["ja"],
            [
                ("official", _stub_failure("err1")),
                ("auto", _stub_failure("err2")),
                ("whisper", _stub_failure("err3")),
            ],
        )
        assert result.source == TranscriptSource.ERROR
        assert result.error == "all transcript tiers failed"
        assert "official:err1" in (result.fallback_reason or "")
        assert "auto:err2" in (result.fallback_reason or "")
        assert "whisper:err3" in (result.fallback_reason or "")
        assert result.snippets == []

    def test_none_fetcher_is_skipped(self):
        result = fetch_with_fallback(
            "vid1",
            ["ja"],
            [
                ("official", _stub_failure("no")),
                ("whisper", None),  # whisper extra not installed
                ("auto", _stub_success(TranscriptSource.AUTO)),
            ],
        )
        assert result.source == TranscriptSource.AUTO
        assert "whisper:disabled" in (result.fallback_reason or "")

    def test_fallback_reason_none_when_first_tier_wins(self):
        result = fetch_with_fallback(
            "vid1",
            ["ja"],
            [
                ("official", _stub_success(TranscriptSource.OFFICIAL)),
            ],
        )
        assert result.fallback_reason is None


class TestInjectedCache:
    """DI: an explicitly injected cache is consulted instead of the global."""

    def test_injected_cache_is_read_and_written(self, tmp_path):
        from pipeline_youtube.services.cache import Cache

        cache = Cache(tmp_path, enabled=True)

        # First call populates the injected cache from a successful fetch.
        first = fetch_with_fallback(
            "vid1",
            ["ja"],
            [("official", _stub_success(TranscriptSource.OFFICIAL))],
            cache=cache,
        )
        assert first.source == TranscriptSource.OFFICIAL

        # Second call: the only fetcher raises, so a non-error result can come
        # *only* from the injected cache — proving the hit path uses it (the
        # process-global cache is disabled by default and would miss).
        second = fetch_with_fallback(
            "vid1",
            ["ja"],
            [("official", _stub_failure("must_not_run"))],
            cache=cache,
        )
        assert second.source == TranscriptSource.OFFICIAL
        assert second.snippets == first.snippets
