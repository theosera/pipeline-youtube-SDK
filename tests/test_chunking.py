"""Tests for N-second window chunking."""

from __future__ import annotations

import pytest

from pipeline_youtube.transcript.base import TranscriptSnippet
from pipeline_youtube.transcript.chunking import Chunk, chunk_by_window


def _snippet(text: str, start: float, duration: float = 2.0) -> TranscriptSnippet:
    return TranscriptSnippet(text=text, start=start, duration=duration)


class TestChunkByWindow:
    def test_empty_input(self):
        assert chunk_by_window([]) == []

    def test_single_snippet(self):
        snips = [_snippet("hello", 0.0)]
        chunks = chunk_by_window(snips, window_seconds=30.0)
        assert len(chunks) == 1
        assert chunks[0].start == 0.0
        assert chunks[0].text == "hello"

    def test_all_snippets_within_window(self):
        snips = [
            _snippet("one", 0.0),
            _snippet("two", 10.0),
            _snippet("three", 20.0),
        ]
        chunks = chunk_by_window(snips, window_seconds=30.0)
        assert len(chunks) == 1
        assert chunks[0].text == "one two three"

    def test_splits_when_window_exceeded(self):
        snips = [
            _snippet("a", 0.0),
            _snippet("b", 10.0),
            _snippet("c", 30.0),  # exactly at boundary → new chunk
            _snippet("d", 40.0),
        ]
        chunks = chunk_by_window(snips, window_seconds=30.0)
        assert len(chunks) == 2
        assert chunks[0].start == 0.0
        assert chunks[0].text == "a b"
        assert chunks[1].start == 30.0
        assert chunks[1].text == "c d"

    def test_multiple_splits(self):
        snips = [_snippet(f"s{i}", float(i * 5)) for i in range(20)]
        # 100 seconds total, ~30s windows → 4 chunks (0, 30, 60, 90)
        chunks = chunk_by_window(snips, window_seconds=30.0)
        assert len(chunks) == 4
        assert chunks[0].start == 0.0
        assert chunks[1].start == 30.0
        assert chunks[2].start == 60.0
        assert chunks[3].start == 90.0

    def test_internal_whitespace_collapsed(self):
        snips = [
            _snippet("hello   world", 0.0),
            _snippet("foo\tbar", 5.0),
        ]
        chunks = chunk_by_window(snips)
        assert chunks[0].text == "hello world foo bar"

    def test_skips_empty_snippet_text(self):
        snips = [
            _snippet("real", 0.0),
            _snippet("   ", 2.0),  # whitespace-only
            _snippet("text", 4.0),
        ]
        chunks = chunk_by_window(snips)
        assert chunks[0].text == "real text"

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            chunk_by_window([_snippet("x", 0.0)], window_seconds=0)

    def test_negative_window_raises(self):
        with pytest.raises(ValueError):
            chunk_by_window([_snippet("x", 0.0)], window_seconds=-5.0)

    def test_very_small_window_creates_many_chunks(self):
        snips = [_snippet(f"s{i}", float(i)) for i in range(5)]
        chunks = chunk_by_window(snips, window_seconds=1.0)
        # Each snippet is 1 second apart; window=1 → each becomes its own chunk
        assert len(chunks) == 5


class TestChunkFormatting:
    def test_mmss_basic(self):
        c = Chunk(start=0.0, text="")
        assert c.mmss == "00:00"

    def test_mmss_minutes_only(self):
        c = Chunk(start=125.0, text="")
        assert c.mmss == "02:05"

    def test_mmss_hour_overflow_stays_in_mm(self):
        # The plan doesn't require HH formatting; chunks at 1h+ still use MM:SS
        # with MM >= 60 (e.g., "60:05" for 3605s)
        c = Chunk(start=3605.0, text="")
        assert c.mmss == "60:05"

    def test_start_int(self):
        c = Chunk(start=125.7, text="")
        assert c.start_int == 125

    def test_start_int_zero(self):
        c = Chunk(start=0.3, text="")
        assert c.start_int == 0


class TestDummyDataReproduction:
    """Reproduce the dummy-data layout loosely to catch format drift."""

    def test_format_matches_dummy_pattern(self):
        snips = [
            _snippet("本さん、最近ハーネス…", 0.0),
            _snippet("実験して結果を公開…", 32.0),
            _snippet("MDEですね、ルールズ…", 67.0),
        ]
        chunks = chunk_by_window(snips, window_seconds=30.0)
        # Should produce 3 chunks matching the dummy's cadence
        assert len(chunks) == 3
        assert chunks[0].mmss == "00:00"
        assert chunks[1].mmss == "00:32"
        assert chunks[2].mmss == "01:07"
