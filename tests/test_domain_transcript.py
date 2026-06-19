"""Tests for pipeline_youtube.domain.transcript types.

Covers:
- TranscriptSource (StrEnum)
- TranscriptNotAvailable (Exception)
- TranscriptSnippet (frozen dataclass)
- TranscriptResult (frozen dataclass)
"""

from __future__ import annotations

import pytest

from pipeline_youtube.domain.transcript import (
    TranscriptNotAvailable,
    TranscriptResult,
    TranscriptSnippet,
    TranscriptSource,
)

# ---------------------------------------------------------------------------
# TranscriptSource
# ---------------------------------------------------------------------------


class TestTranscriptSource:
    def test_official_value(self):
        assert TranscriptSource.OFFICIAL == "official"

    def test_auto_value(self):
        assert TranscriptSource.AUTO == "auto-generated"

    def test_whisper_value(self):
        assert TranscriptSource.WHISPER == "whisper"

    def test_error_value(self):
        assert TranscriptSource.ERROR == "error"

    def test_is_str_enum(self):
        # StrEnum members compare equal to their string values
        assert TranscriptSource.OFFICIAL == "official"
        assert isinstance(TranscriptSource.OFFICIAL, str)

    def test_all_four_members(self):
        members = set(TranscriptSource)
        assert members == {
            TranscriptSource.OFFICIAL,
            TranscriptSource.AUTO,
            TranscriptSource.WHISPER,
            TranscriptSource.ERROR,
        }

    def test_usable_in_string_context(self):
        source = TranscriptSource.WHISPER
        assert f"source={source}" == "source=whisper"

    def test_construct_from_string(self):
        assert TranscriptSource("official") is TranscriptSource.OFFICIAL
        assert TranscriptSource("auto-generated") is TranscriptSource.AUTO
        assert TranscriptSource("whisper") is TranscriptSource.WHISPER
        assert TranscriptSource("error") is TranscriptSource.ERROR

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            TranscriptSource("unknown")


# ---------------------------------------------------------------------------
# TranscriptNotAvailable
# ---------------------------------------------------------------------------


class TestTranscriptNotAvailable:
    def test_is_exception(self):
        assert issubclass(TranscriptNotAvailable, Exception)

    def test_can_raise_and_catch(self):
        with pytest.raises(TranscriptNotAvailable):
            raise TranscriptNotAvailable("no transcript found")

    def test_message_preserved(self):
        exc = TranscriptNotAvailable("captions disabled")
        assert str(exc) == "captions disabled"

    def test_can_catch_as_exception(self):
        caught = None
        try:
            raise TranscriptNotAvailable("tier failed")
        except Exception as e:
            caught = e
        assert isinstance(caught, TranscriptNotAvailable)

    def test_empty_message(self):
        exc = TranscriptNotAvailable()
        assert isinstance(exc, Exception)

    def test_not_caught_by_other_exception_type(self):
        with pytest.raises(TranscriptNotAvailable):
            try:
                raise TranscriptNotAvailable("x")
            except ValueError:
                pass  # Should not be caught here


# ---------------------------------------------------------------------------
# TranscriptSnippet
# ---------------------------------------------------------------------------


class TestTranscriptSnippet:
    def test_fields_stored(self):
        s = TranscriptSnippet(text="Hello world", start=1.5, duration=2.0)
        assert s.text == "Hello world"
        assert s.start == 1.5
        assert s.duration == 2.0

    def test_end_property(self):
        s = TranscriptSnippet(text="hi", start=5.0, duration=3.0)
        assert s.end == pytest.approx(8.0)

    def test_end_property_zero_start(self):
        s = TranscriptSnippet(text="first", start=0.0, duration=1.5)
        assert s.end == pytest.approx(1.5)

    def test_end_property_fractional(self):
        s = TranscriptSnippet(text="fragment", start=10.3, duration=2.7)
        assert s.end == pytest.approx(13.0)

    def test_frozen_rejects_mutation(self):
        s = TranscriptSnippet(text="test", start=0.0, duration=1.0)
        with pytest.raises(AttributeError):
            s.text = "changed"  # type: ignore[misc]

    def test_equality(self):
        s1 = TranscriptSnippet(text="hello", start=0.0, duration=1.0)
        s2 = TranscriptSnippet(text="hello", start=0.0, duration=1.0)
        assert s1 == s2

    def test_inequality_different_text(self):
        s1 = TranscriptSnippet(text="hello", start=0.0, duration=1.0)
        s2 = TranscriptSnippet(text="world", start=0.0, duration=1.0)
        assert s1 != s2

    def test_inequality_different_start(self):
        s1 = TranscriptSnippet(text="hello", start=1.0, duration=1.0)
        s2 = TranscriptSnippet(text="hello", start=2.0, duration=1.0)
        assert s1 != s2

    def test_hashable(self):
        s = TranscriptSnippet(text="test", start=0.0, duration=1.0)
        s_set = {s}
        assert s in s_set

    def test_empty_text(self):
        s = TranscriptSnippet(text="", start=0.0, duration=0.5)
        assert s.text == ""

    def test_unicode_text(self):
        s = TranscriptSnippet(text="日本語テスト", start=3.14, duration=2.72)
        assert s.text == "日本語テスト"
        assert s.end == pytest.approx(3.14 + 2.72)

    def test_zero_duration(self):
        s = TranscriptSnippet(text="instant", start=5.0, duration=0.0)
        assert s.end == pytest.approx(5.0)

    def test_large_timestamps(self):
        s = TranscriptSnippet(text="late in video", start=7200.0, duration=5.0)
        assert s.end == pytest.approx(7205.0)


# ---------------------------------------------------------------------------
# TranscriptResult
# ---------------------------------------------------------------------------


def _make_snippet(n: int = 0) -> TranscriptSnippet:
    return TranscriptSnippet(text=f"snippet {n}", start=float(n), duration=1.0)


def _make_result(**kwargs) -> TranscriptResult:
    defaults = dict(
        video_id="dQw4w9WgXcQ",
        source=TranscriptSource.OFFICIAL,
        language="en",
    )
    defaults.update(kwargs)
    return TranscriptResult(**defaults)


class TestTranscriptResult:
    def test_required_fields(self):
        r = _make_result()
        assert r.video_id == "dQw4w9WgXcQ"
        assert r.source == TranscriptSource.OFFICIAL
        assert r.language == "en"

    def test_defaults(self):
        r = _make_result()
        assert r.snippets == []
        assert r.retrieved_at == ""
        assert r.fallback_reason is None
        assert r.error is None
        assert r.correction_cost_usd is None
        assert r.confirmed_terms == ()

    def test_snippets_stored(self):
        snippets = [_make_snippet(0), _make_snippet(1)]
        r = _make_result(snippets=snippets)
        assert r.snippets == snippets
        assert len(r.snippets) == 2

    def test_frozen(self):
        r = _make_result()
        with pytest.raises(AttributeError):
            r.video_id = "changed"  # type: ignore[misc]

    def test_language_none(self):
        r = _make_result(language=None)
        assert r.language is None

    def test_error_source(self):
        r = _make_result(source=TranscriptSource.ERROR, error="all tiers failed")
        assert r.source == TranscriptSource.ERROR
        assert r.error == "all tiers failed"

    def test_whisper_source(self):
        r = _make_result(source=TranscriptSource.WHISPER)
        assert r.source == TranscriptSource.WHISPER

    def test_auto_source(self):
        r = _make_result(source=TranscriptSource.AUTO)
        assert r.source == TranscriptSource.AUTO

    def test_fallback_reason_stored(self):
        r = _make_result(fallback_reason="innertube:bot-detection; official:no captions")
        assert r.fallback_reason == "innertube:bot-detection; official:no captions"

    def test_retrieved_at_stored(self):
        r = _make_result(retrieved_at="2024-01-15T10:30:00+00:00")
        assert r.retrieved_at == "2024-01-15T10:30:00+00:00"

    def test_correction_cost_zero(self):
        r = _make_result(correction_cost_usd=0.0)
        assert r.correction_cost_usd == 0.0

    def test_correction_cost_none_vs_zero(self):
        # None means correction was not attempted; 0.0 means it ran at no cost
        r_none = _make_result(correction_cost_usd=None)
        r_zero = _make_result(correction_cost_usd=0.0)
        assert r_none.correction_cost_usd is None
        assert r_zero.correction_cost_usd == 0.0

    def test_confirmed_terms_stored(self):
        r = _make_result(confirmed_terms=("GPT-4", "Transformer", "RLHF"))
        assert r.confirmed_terms == ("GPT-4", "Transformer", "RLHF")

    def test_confirmed_terms_empty(self):
        r = _make_result(confirmed_terms=())
        assert r.confirmed_terms == ()

    def test_equality(self):
        r1 = _make_result()
        r2 = _make_result()
        assert r1 == r2

    def test_inequality_different_video_id(self):
        r1 = _make_result(video_id="aaaa")
        r2 = _make_result(video_id="bbbb")
        assert r1 != r2

    def test_hashable(self):
        # frozen dataclass with a list field: the list is not hashable so hash() raises
        r = _make_result()
        with pytest.raises(TypeError):
            hash(r)

    def test_snippets_default_is_independent_per_instance(self):
        # Verify default_factory produces a new list each time (no shared state)
        r1 = _make_result()
        r2 = _make_result()
        assert r1.snippets is not r2.snippets
