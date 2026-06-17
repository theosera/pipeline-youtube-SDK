"""Tests for Tier 0 InnerTube caption fetch (transcript/innertube.py).

All network is injected, so these run offline and assert the selection /
parsing / source-mapping contracts the fallback chain depends on.
"""

from __future__ import annotations

from typing import Any

import pytest

from pipeline_youtube.transcript.base import TranscriptNotAvailable, TranscriptSource
from pipeline_youtube.transcript.innertube import (
    _extract_caption_tracks,
    _parse_timedtext,
    _retry_after_seconds,
    _select_track,
    fetch_innertube,
)


class TestRetryAfter:
    def test_numeric_header(self) -> None:
        assert _retry_after_seconds({"Retry-After": "30"}) == 30.0

    def test_missing_header(self) -> None:
        assert _retry_after_seconds({}) is None

    def test_non_numeric_header(self) -> None:
        # HTTP-date form is not parsed; we fall back to exponential backoff.
        assert _retry_after_seconds({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}) is None


class TestFetchTrackTextRetry:
    """Retry/backoff behavior of the real track-text fetcher (httpx mocked)."""

    class _Resp:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code
            self.headers: dict[str, str] = {}
            self.text = ""

    def test_retryable_status_exhausts_with_reason(self, monkeypatch) -> None:
        # A persistent 429 must report retries_exhausted:<status> after using
        # every attempt — not a bare http_429 that looks like a single blip.
        import httpx

        from pipeline_youtube.transcript import innertube as it

        calls = {"n": 0}

        def _fake_get(*_a: Any, **_k: Any) -> TestFetchTrackTextRetry._Resp:
            calls["n"] += 1
            return self._Resp(429)

        monkeypatch.setattr(httpx, "get", _fake_get)
        monkeypatch.setattr(it.time, "sleep", lambda *_: None)
        monkeypatch.setattr(it.random, "uniform", lambda *_: 0.0)
        with pytest.raises(TranscriptNotAvailable, match="innertube_track_retries_exhausted:429"):
            it._default_fetch_track_text("http://x", 1.0, max_retries=2)
        assert calls["n"] == 3  # initial attempt + 2 retries

    def test_non_retryable_status_raises_http_immediately(self, monkeypatch) -> None:
        import httpx

        from pipeline_youtube.transcript import innertube as it

        calls = {"n": 0}

        def _fake_get(*_a: Any, **_k: Any) -> TestFetchTrackTextRetry._Resp:
            calls["n"] += 1
            return self._Resp(404)

        monkeypatch.setattr(httpx, "get", _fake_get)
        with pytest.raises(TranscriptNotAvailable, match="innertube_track_http_404"):
            it._default_fetch_track_text("http://x", 1.0, max_retries=2)
        assert calls["n"] == 1  # no retries on a non-retryable status


class TestParseTimedtext:
    def test_text_form_seconds(self) -> None:
        xml = (
            '<transcript><text start="0.0" dur="1.5">Hello</text>'
            '<text start="1.5" dur="2.0">world</text></transcript>'
        )
        out = _parse_timedtext(xml)
        assert [(s.text, s.start, s.duration) for s in out] == [
            ("Hello", 0.0, 1.5),
            ("world", 1.5, 2.0),
        ]

    def test_p_form_milliseconds(self) -> None:
        xml = '<timedtext><p t="0" d="1500">Hello</p><p t="1500" d="2000">world</p></timedtext>'
        out = _parse_timedtext(xml)
        assert [(s.text, s.start, s.duration) for s in out] == [
            ("Hello", 0.0, 1.5),
            ("world", 1.5, 2.0),
        ]

    def test_text_form_tolerates_attr_order_and_missing_dur(self) -> None:
        # Attribute order is not guaranteed, and the final cue can omit dur;
        # both must still parse (missing dur → 0.0) rather than be dropped.
        xml = '<text dur="1.5" start="0.0">A</text><text start="1.5">B</text>'
        out = _parse_timedtext(xml)
        assert [(s.text, s.start, s.duration) for s in out] == [
            ("A", 0.0, 1.5),
            ("B", 1.5, 0.0),
        ]

    def test_p_form_tolerates_attr_order_and_missing_d(self) -> None:
        xml = '<p d="1500" t="0">A</p><p t="1500">B</p>'
        out = _parse_timedtext(xml)
        assert [(s.text, s.start, s.duration) for s in out] == [
            ("A", 0.0, 1.5),
            ("B", 1.5, 0.0),
        ]

    def test_unparseable_text_tags_fall_through_to_p(self) -> None:
        # <text> tags present but yielding no cues (e.g. missing start) must not
        # short-circuit to an empty result; fall through to the <p>/srv3 shape.
        xml = '<text>oops</text><p t="0" d="1500">hi</p>'
        out = _parse_timedtext(xml)
        assert [(s.text, s.start, s.duration) for s in out] == [("hi", 0.0, 1.5)]

    def test_unescapes_entities_and_strips_inner_tags(self) -> None:
        xml = '<text start="0" dur="1">A &amp; <b>B</b>\nC</text>'
        out = _parse_timedtext(xml)
        assert out[0].text == "A & B C"

    def test_skips_empty_lines(self) -> None:
        xml = '<text start="0" dur="1"></text><text start="1" dur="1">x</text>'
        out = _parse_timedtext(xml)
        assert [s.text for s in out] == ["x"]

    def test_prefers_text_form_when_both_absent_returns_empty(self) -> None:
        assert _parse_timedtext("<nothing/>") == []


class TestSelectTrack:
    def _tracks(self) -> list[dict[str, Any]]:
        return [
            {"languageCode": "en", "kind": "asr", "baseUrl": "u-en-asr"},
            {"languageCode": "ja", "baseUrl": "u-ja"},
            {"languageCode": "ja", "kind": "asr", "baseUrl": "u-ja-asr"},
        ]

    def test_exact_language_match(self) -> None:
        assert _select_track(self._tracks(), ["ja"])["baseUrl"] == "u-ja"

    def test_manual_preferred_over_asr(self) -> None:
        # Two ja tracks exist (manual + asr); manual must win.
        assert _select_track(self._tracks(), ["ja"]).get("kind") != "asr"

    def test_language_order_respected(self) -> None:
        assert _select_track(self._tracks(), ["ja", "en"])["languageCode"] == "ja"
        assert _select_track(self._tracks(), ["en", "ja"])["languageCode"] == "en"

    def test_prefix_match(self) -> None:
        tracks = [{"languageCode": "en-US", "baseUrl": "u"}]
        assert _select_track(tracks, ["en"])["baseUrl"] == "u"

    def test_reverse_prefix_match(self) -> None:
        tracks = [{"languageCode": "en", "baseUrl": "u"}]
        assert _select_track(tracks, ["en-GB"])["baseUrl"] == "u"

    def test_returns_none_when_requested_language_unavailable(self) -> None:
        # Language is strict: a fr-only video must NOT satisfy a ja request, so
        # the chain falls through to the later (also strict) tiers.
        tracks = [{"languageCode": "fr", "baseUrl": "u-fr"}]
        assert _select_track(tracks, ["ja"]) is None

    def test_falls_back_to_first_only_when_no_language_requested(self) -> None:
        tracks = [{"languageCode": "fr", "baseUrl": "u-fr"}]
        assert _select_track(tracks, [])["baseUrl"] == "u-fr"

    def test_empty_tracks_returns_none(self) -> None:
        assert _select_track([], ["ja"]) is None


class TestExtractCaptionTracks:
    def test_well_formed(self) -> None:
        payload = {
            "captions": {
                "playerCaptionsTracklistRenderer": {
                    "captionTracks": [{"languageCode": "ja", "baseUrl": "u"}]
                }
            }
        }
        assert _extract_caption_tracks(payload) == [{"languageCode": "ja", "baseUrl": "u"}]

    def test_missing_captions(self) -> None:
        assert _extract_caption_tracks({}) == []

    def test_malformed_renderer(self) -> None:
        assert _extract_caption_tracks({"captions": {"playerCaptionsTracklistRenderer": []}}) == []


def _player(tracks: list[dict[str, Any]], *, status: str | None = "OK") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "captions": {"playerCaptionsTracklistRenderer": {"captionTracks": tracks}}
    }
    if status is not None:
        payload["playabilityStatus"] = {"status": status}
    return payload


class TestFetchInnertube:
    def test_manual_track_maps_to_official(self) -> None:
        tracks = [{"languageCode": "ja", "baseUrl": "u-ja"}]
        result = fetch_innertube(
            "vid",
            ["ja"],
            fetch_player_json=lambda *_: _player(tracks),
            fetch_track_text=lambda *_: '<text start="0" dur="1">こんにちは</text>',
        )
        assert result.source == TranscriptSource.OFFICIAL
        assert result.language == "ja"
        assert result.snippets[0].text == "こんにちは"

    def test_asr_track_maps_to_auto(self) -> None:
        tracks = [{"languageCode": "ja", "kind": "asr", "baseUrl": "u"}]
        result = fetch_innertube(
            "vid",
            ["ja"],
            fetch_player_json=lambda *_: _player(tracks),
            fetch_track_text=lambda *_: '<text start="0" dur="1">x</text>',
        )
        assert result.source == TranscriptSource.AUTO

    def test_no_caption_tracks_raises(self) -> None:
        with pytest.raises(TranscriptNotAvailable, match="no_caption_tracks"):
            fetch_innertube(
                "vid",
                ["ja"],
                fetch_player_json=lambda *_: _player([]),
                fetch_track_text=lambda *_: "",
            )

    def test_non_ok_playability_surfaces_as_reason_when_no_tracks(self) -> None:
        # When there are no caption tracks, a non-OK playabilityStatus is the
        # informative reason (vs a bare no_caption_tracks).
        with pytest.raises(TranscriptNotAvailable, match="playability:LOGIN_REQUIRED"):
            fetch_innertube(
                "vid",
                ["ja"],
                fetch_player_json=lambda *_: _player([], status="LOGIN_REQUIRED"),
                fetch_track_text=lambda *_: "x",
            )

    def test_gated_video_with_captions_is_still_fetched(self) -> None:
        # Captions present but playback gated (age/region/login): tier-0 must
        # still serve the captions rather than abort on playabilityStatus —
        # gated videos often expose fetchable caption tracks.
        tracks = [{"languageCode": "ja", "baseUrl": "u"}]
        result = fetch_innertube(
            "vid",
            ["ja"],
            fetch_player_json=lambda *_: _player(tracks, status="LOGIN_REQUIRED"),
            fetch_track_text=lambda *_: '<text start="0" dur="1">x</text>',
        )
        assert result.snippets[0].text == "x"

    def test_empty_xml_raises(self) -> None:
        tracks = [{"languageCode": "ja", "baseUrl": "u"}]
        with pytest.raises(TranscriptNotAvailable, match="empty_transcript"):
            fetch_innertube(
                "vid",
                ["ja"],
                fetch_player_json=lambda *_: _player(tracks),
                fetch_track_text=lambda *_: "<nothing/>",
            )

    def test_empty_video_id_raises(self) -> None:
        with pytest.raises(TranscriptNotAvailable, match="empty video_id"):
            fetch_innertube("", ["ja"])

    def test_parser_error_degrades_to_transcript_not_available(self) -> None:
        # A malformed numeric attribute makes float() raise; it must surface as
        # TranscriptNotAvailable so the fallback chain continues, not crash.
        tracks = [{"languageCode": "ja", "baseUrl": "u"}]
        with pytest.raises(TranscriptNotAvailable, match="innertube_parse_failed"):
            fetch_innertube(
                "vid",
                ["ja"],
                fetch_player_json=lambda *_: _player(tracks),
                fetch_track_text=lambda *_: '<text start="1.2.3" dur="1">x</text>',
            )

    def test_track_missing_baseurl_raises(self) -> None:
        tracks = [{"languageCode": "ja"}]
        with pytest.raises(TranscriptNotAvailable, match="track_missing_baseurl"):
            fetch_innertube(
                "vid",
                ["ja"],
                fetch_player_json=lambda *_: _player(tracks),
                fetch_track_text=lambda *_: "x",
            )
