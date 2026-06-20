"""LLM evaluator sub-agents (provider mocked)."""

from __future__ import annotations

from pipeline_youtube.evaluation import evaluators as ev
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.providers.base import LLMResponse
from pipeline_youtube.services.cache import Cache
from pipeline_youtube.stages.synthesis import SynthesisStageResult
from pipeline_youtube.synthesis.scoring import (
    ChapterPlan,
    CoverageReport,
    LeaderOutput,
    SynthesisChapterBody,
    SynthesisMoc,
)

_NO_CACHE = Cache(None, enabled=False)


def _video() -> VideoMeta:
    return VideoMeta(
        video_id="vid000",
        title="V",
        url="https://www.youtube.com/watch?v=vid000",
        duration=100,
        channel="c",
        upload_date="20260415",
        playlist_title="P",
    )


def _synth(leader: bool) -> SynthesisStageResult:
    lo = (
        LeaderOutput(
            moc=SynthesisMoc(title="T", body_markdown="b"),
            chapters=[SynthesisChapterBody(1, "ch", "core", ["vid000"], "body")],
        )
        if leader
        else None
    )
    return SynthesisStageResult(
        leader_output=lo,
        coverage=CoverageReport(missing_topic_ids=["t009"]),
        chapters=[ChapterPlan(1, "ch", "core", ["t001"], ["vid000"])],
    )


_COVERAGE_JSON = (
    '{"summary": "ok", "findings": [{"finding_id": "f001", "severity": "high", '
    '"target_scope": "04", "target_video_id": "vid000", "description": "d", '
    '"suggested_fix": "x"}]}'
)


def test_coverage_evaluator_parses_and_injects_missing_signal(monkeypatch) -> None:
    captured: dict = {}

    def fake_invoke(**kw):
        captured.update(kw)
        return LLMResponse(text=_COVERAGE_JSON, model="sonnet")

    monkeypatch.setattr(ev, "invoke_claude", fake_invoke)

    report, _result = ev.call_coverage_evaluator(
        [_video()], ["body"], _synth(leader=True), cache=_NO_CACHE
    )

    assert report.perspective == "coverage"
    assert report.findings[0].finding_id == "f001"
    assert report.findings[0].perspective == "coverage"  # forced role
    assert captured["role"] == "eval_coverage"
    assert "t009" in captured["prompt"]  # deterministic missing-topic signal injected


def test_pedagogy_evaluator_parses(monkeypatch) -> None:
    monkeypatch.setattr(
        ev,
        "invoke_claude",
        lambda **kw: LLMResponse(text='{"summary": "p", "findings": []}', model="sonnet"),
    )
    report, _ = ev.call_pedagogy_evaluator(
        [_video()], ["body"], _synth(leader=True), cache=_NO_CACHE
    )
    assert report.perspective == "pedagogy"
    assert report.findings == []


def test_evaluator_returns_empty_when_no_leader_output(monkeypatch) -> None:
    monkeypatch.setattr(ev, "invoke_claude", lambda **kw: LLMResponse(text="", model="sonnet"))
    report, _ = ev.call_coverage_evaluator(
        [_video()], ["body"], _synth(leader=False), cache=_NO_CACHE
    )
    assert report.perspective == "coverage"
    assert report.findings == []
