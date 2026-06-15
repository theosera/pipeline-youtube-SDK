"""Tests for Stage 01b transcript correction (transcript/correction.py)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from pipeline_youtube.providers.base import LLMError, LLMResponse
from pipeline_youtube.transcript.chunking import Chunk
from pipeline_youtube.transcript.correction import (
    _parse_corrections,
    chunks_to_snippets,
    correct_chunks,
)


def _response(text: str) -> LLMResponse:
    return LLMResponse(text=text, model="opus", provider="anthropic")


def _stub_invoke(text: str):
    calls: list[dict[str, Any]] = []

    def invoke(**kwargs: Any) -> LLMResponse:
        calls.append(kwargs)
        return _response(text)

    invoke.calls = calls  # type: ignore[attr-defined]
    return invoke


class TestParseCorrections:
    def test_plain_json(self) -> None:
        out = _parse_corrections('[{"idx": 0, "text": "直した"}, {"idx": 1, "text": "B"}]')
        assert out == {0: "直した", 1: "B"}

    def test_strips_code_fence(self) -> None:
        fenced = '```json\n[{"idx": 2, "text": "X"}]\n```'
        assert _parse_corrections(fenced) == {2: "X"}

    def test_non_array_raises(self) -> None:
        with pytest.raises(ValueError, match="JSON array"):
            _parse_corrections('{"idx": 0, "text": "a"}')

    def test_missing_keys_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_corrections('[{"idx": 0}]')

    def test_bad_json_raises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            _parse_corrections("not json")


class TestCorrectChunks:
    def _chunks(self) -> list[Chunk]:
        return [Chunk(start=0.0, text="ぐぐる"), Chunk(start=30.0, text="てんさーふろー")]

    def test_applies_corrections_and_preserves_timestamps(self) -> None:
        invoke = _stub_invoke('[{"idx": 0, "text": "Google"}, {"idx": 1, "text": "TensorFlow"}]')
        out = correct_chunks(self._chunks(), model="opus", invoke=invoke)
        assert [c.text for c in out] == ["Google", "TensorFlow"]
        assert [c.start for c in out] == [0.0, 30.0]

    def test_enables_web_search_and_thinking(self) -> None:
        invoke = _stub_invoke('[{"idx": 0, "text": "Google"}, {"idx": 1, "text": "x"}]')
        correct_chunks(self._chunks(), model="opus", invoke=invoke)
        call = invoke.calls[0]  # type: ignore[attr-defined]
        assert call["web_search"] is True
        assert call["thinking"] is True
        assert call["role"] == "stage_01_correct"
        assert call["model"] == "opus"

    def test_missing_index_keeps_original(self) -> None:
        invoke = _stub_invoke('[{"idx": 0, "text": "Google"}]')
        out = correct_chunks(self._chunks(), model="opus", invoke=invoke)
        assert out[0].text == "Google"
        assert out[1].text == "てんさーふろー"

    def test_empty_correction_keeps_original(self) -> None:
        invoke = _stub_invoke('[{"idx": 0, "text": ""}, {"idx": 1, "text": "x"}]')
        out = correct_chunks(self._chunks(), model="opus", invoke=invoke)
        assert out[0].text == "ぐぐる"

    def test_bad_json_falls_back_to_original(self) -> None:
        invoke = _stub_invoke("the model rambled instead of returning JSON")
        out = correct_chunks(self._chunks(), model="opus", invoke=invoke)
        assert [c.text for c in out] == ["ぐぐる", "てんさーふろー"]

    def test_llm_error_falls_back_to_original(self) -> None:
        def invoke(**kwargs: Any) -> LLMResponse:
            raise LLMError("boom")

        out = correct_chunks(self._chunks(), model="opus", invoke=invoke)
        assert [c.text for c in out] == ["ぐぐる", "てんさーふろー"]

    def test_empty_input(self) -> None:
        assert correct_chunks([], model="opus", invoke=_stub_invoke("[]")) == []

    def test_chunks_to_snippets_preserves_timeline(self) -> None:
        chunks = [
            Chunk(start=0.0, text="A"),
            Chunk(start=30.0, text="B"),
            Chunk(start=70.0, text="C"),
        ]
        snippets = chunks_to_snippets(chunks, last_end=95.0)
        assert [s.text for s in snippets] == ["A", "B", "C"]
        assert [s.start for s in snippets] == [0.0, 30.0, 70.0]
        assert [s.duration for s in snippets] == [30.0, 40.0, 25.0]

    def test_batching_splits_calls(self) -> None:
        chunks = [Chunk(start=float(i), text=str(i)) for i in range(5)]

        def invoke(**kwargs: Any) -> LLMResponse:
            prompt = kwargs["prompt"]
            idxs = [int(line[1 : line.index("]")]) for line in prompt.splitlines()]
            return _response(json.dumps([{"idx": i, "text": "ok"} for i in idxs]))

        out = correct_chunks(chunks, model="opus", invoke=invoke, batch_size=2)
        assert all(c.text == "ok" for c in out)
        assert [c.start for c in out] == [0.0, 1.0, 2.0, 3.0, 4.0]
