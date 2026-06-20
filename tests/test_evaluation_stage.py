"""Advisory evaluation stage (no regeneration).

``run_stage_evaluation`` runs the evaluators ONCE, writes a report, and
returns the synthesis result unchanged. The earlier regen-loop scaffold
tests were removed along with the regen machinery (advisory-only design).
LLM evaluators are mocked; the deterministic fidelity scan runs for real.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from pipeline_youtube import config
from pipeline_youtube.evaluation.schemas import EvaluatorReport, Finding
from pipeline_youtube.glossary.schema import Glossary, GlossaryEntry
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.providers.base import LLMResponse
from pipeline_youtube.services.cache import Cache
from pipeline_youtube.stages import evaluation as eval_stage
from pipeline_youtube.stages.evaluation import run_stage_evaluation
from pipeline_youtube.stages.synthesis import SynthesisStageResult
from pipeline_youtube.synthesis.agents import AgentCallResult
from pipeline_youtube.synthesis.scoring import (
    ChapterPlan,
    CoverageReport,
    LeaderOutput,
    SynthesisChapterBody,
    SynthesisMoc,
)

_NO_CACHE = Cache(None, enabled=False)


@pytest.fixture
def vault(tmp_path: Path):
    config.set_vault_root(tmp_path)
    config.set_dry_run(False)
    yield config.get_vault_root()
    config.reset_vault_root()


def _video(i: int = 0) -> VideoMeta:
    return VideoMeta(
        video_id=f"vid{i:03d}",
        title=f"Video {i}",
        url=f"https://www.youtube.com/watch?v=vid{i:03d}",
        duration=100,
        channel="Test",
        upload_date="20260415",
        playlist_title="Test Playlist",
    )


def _agent_result() -> AgentCallResult:
    resp = LLMResponse(text="{}", model="sonnet")
    return AgentCallResult(
        response=resp,
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        total_cost_usd=0.0,
        duration_ms=1,
    )


def _synth_with_leader() -> SynthesisStageResult:
    leader = LeaderOutput(
        moc=SynthesisMoc(title="T", body_markdown="b"),
        chapters=[SynthesisChapterBody(1, "ch", "core", ["vid000"], "body")],
    )
    return SynthesisStageResult(
        leader_output=leader,
        coverage=CoverageReport(covered_topic_ids=["t001"], missing_topic_ids=[]),
        chapters=[ChapterPlan(1, "ch", "core", ["t001"], ["vid000"])],
    )


def _cov_finding() -> Finding:
    return Finding(
        finding_id="f001",
        perspective="coverage",
        severity="low",
        target_scope="05",
        description="cov",
        suggested_fix="x",
    )


def _patch_evaluators(monkeypatch, *, coverage=None, pedagogy=None, cov_raises=False) -> None:
    def fake_cov(*_a, **_k):
        if cov_raises:
            raise RuntimeError("boom")
        report = coverage if coverage is not None else EvaluatorReport(perspective="coverage")
        return report, _agent_result()

    def fake_ped(*_a, **_k):
        report = pedagogy if pedagogy is not None else EvaluatorReport(perspective="pedagogy")
        return report, _agent_result()

    monkeypatch.setattr(eval_stage, "call_coverage_evaluator", fake_cov)
    monkeypatch.setattr(eval_stage, "call_pedagogy_evaluator", fake_ped)


def test_eval_zero_is_noop(vault) -> None:
    synth = _synth_with_leader()
    result = run_stage_evaluation(
        [_video()],
        ["body"],
        synth,
        run_time=datetime(2026, 6, 15),
        playlist_title="P",
        max_loops=0,
        cache=_NO_CACHE,
    )
    assert result.loop_result.skipped is True
    assert result.synthesis_result is synth
    assert result.report_paths == []


def test_advisory_writes_report_and_returns_synthesis_unchanged(vault, monkeypatch) -> None:
    synth = _synth_with_leader()
    _patch_evaluators(
        monkeypatch,
        coverage=EvaluatorReport(perspective="coverage", findings=[_cov_finding()], summary="c"),
    )
    glossary = Glossary(
        entries=(GlossaryEntry(canonical="Vibe Coding", aliases=["ビブコーディング"]),)
    )
    result = run_stage_evaluation(
        [_video()],
        ["本文にビブコーディングが出る"],
        synth,
        run_time=datetime(2026, 6, 15),
        playlist_title="P",
        max_loops=1,
        glossary=glossary,
        folder_name_override="2026-06-15 Test",
        cache=_NO_CACHE,
    )

    # synthesis is returned UNCHANGED (advisory only)
    assert result.synthesis_result is synth
    assert result.loop_result.loops_run == 1
    it = result.loop_result.iterations[0]
    assert it.synthesis_rerun is False
    assert it.regenerated_video_ids == []

    # fidelity scan ran for real and flagged the variant
    assert any(f.perspective == "fidelity" for f in it.report.all_findings)
    assert any(f.perspective == "coverage" for f in it.report.all_findings)

    # artifacts written
    assert result.report_paths and result.report_paths[0].name == "loop_0.json"
    assert result.summary_path is not None and result.summary_path.name == "summary.md"
    assert result.report_paths[0].exists()
    assert "ビブコーディング" in result.report_paths[0].read_text(encoding="utf-8")


def test_dry_run_skips_artifact_writes(vault, monkeypatch) -> None:
    _patch_evaluators(monkeypatch)
    result = run_stage_evaluation(
        [_video()],
        ["body"],
        _synth_with_leader(),
        run_time=datetime(2026, 6, 15),
        playlist_title="P",
        max_loops=1,
        dry_run=True,
        cache=_NO_CACHE,
    )
    assert result.report_paths == []
    assert result.summary_path is None


def test_evaluator_failure_degrades_to_empty_report(vault, monkeypatch) -> None:
    _patch_evaluators(monkeypatch, cov_raises=True)
    result = run_stage_evaluation(
        [_video()],
        ["body"],
        _synth_with_leader(),
        run_time=datetime(2026, 6, 15),
        playlist_title="P",
        max_loops=1,
        folder_name_override="2026-06-15 Test",
        cache=_NO_CACHE,
    )
    # coverage crashed -> empty coverage report, stage still succeeds
    report = result.loop_result.iterations[0].report
    assert report.coverage.findings == []
    # only the pedagogy call produced an agent result (coverage raised)
    assert len(result.agent_results) == 1


def test_no_glossary_yields_empty_fidelity(vault, monkeypatch) -> None:
    _patch_evaluators(monkeypatch)
    result = run_stage_evaluation(
        [_video()],
        ["本文にビブコーディング"],
        _synth_with_leader(),
        run_time=datetime(2026, 6, 15),
        playlist_title="P",
        max_loops=1,
        glossary=None,
        folder_name_override="2026-06-15 Test",
        cache=_NO_CACHE,
    )
    assert result.loop_result.iterations[0].report.fidelity.findings == []
