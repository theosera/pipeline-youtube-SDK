"""Tests for filler word stripping in chunk_by_window."""

from __future__ import annotations

from pipeline_youtube.transcript.base import TranscriptSnippet
from pipeline_youtube.transcript.chunking import DEFAULT_FILLER_WORDS, chunk_by_window


def _s(text: str, start: float) -> TranscriptSnippet:
    return TranscriptSnippet(text=text, start=start, duration=2.0)


class TestFillerStripping:
    def test_no_fillers_when_none_provided(self):
        snips = [_s("えーと AI 駆動経営", 0.0)]
        chunks = chunk_by_window(snips, window_seconds=30.0)
        assert "えーと" in chunks[0].text

    def test_strips_provided_fillers(self):
        snips = [_s("えーと AI 駆動経営 あのー 重要です", 0.0)]
        chunks = chunk_by_window(snips, window_seconds=30.0, filler_words=("えーと", "あのー"))
        assert "えーと" not in chunks[0].text
        assert "あのー" not in chunks[0].text
        assert "AI 駆動経営" in chunks[0].text
        assert "重要です" in chunks[0].text

    def test_default_list_includes_common_fillers(self):
        for word in ("えー", "あの", "まあ", "なんか"):
            assert word in DEFAULT_FILLER_WORDS

    def test_collapses_triple_repeats(self):
        snips = [_s("the the the dog the cat the cat", 0.0)]
        chunks = chunk_by_window(snips, window_seconds=30.0, filler_words=())
        assert "the the the" not in chunks[0].text
        assert "the dog" in chunks[0].text

    def test_empty_filler_word_ignored(self):
        snips = [_s("alpha beta gamma delta", 0.0)]
        chunks = chunk_by_window(snips, window_seconds=30.0, filler_words=("",))
        assert chunks[0].text == "alpha beta gamma delta"

    def test_collapses_short_jp_doubled(self):
        """2-token repeat of short Japanese tokens (ASR stutter) gets collapsed."""
        snips = [_s("これ これ は テスト", 0.0)]
        chunks = chunk_by_window(snips, window_seconds=30.0, filler_words=())
        assert chunks[0].text == "これ は テスト"

    def test_preserves_english_doubles(self):
        """Legitimate English repetition like 'very very' must NOT be collapsed."""
        snips = [_s("very very good", 0.0)]
        chunks = chunk_by_window(snips, window_seconds=30.0, filler_words=())
        assert chunks[0].text == "very very good"
