"""Deterministic routing/aggregation behavior (no LLM)."""

from __future__ import annotations

import pytest

from pipeline_youtube.evaluation.routing import (
    aggregate_reports,
    has_blocking_findings,
    route_findings,
    target_video_ids_for_04,
)
from pipeline_youtube.evaluation.schemas import EvaluatorReport, Finding


def _finding(
    fid: str,
    *,
    perspective: str = "coverage",
    severity: str = "high",
    scope: str = "05",
    video_id: str | None = None,
) -> Finding:
    return Finding(
        finding_id=fid,
        perspective=perspective,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        target_scope=scope,  # type: ignore[arg-type]
        description="d",
        suggested_fix="f",
        target_video_id=video_id,
    )


def test_aggregate_places_reports_in_slots() -> None:
    cov = EvaluatorReport(perspective="coverage", findings=[_finding("f1")])
    ped = EvaluatorReport(perspective="pedagogy")
    fid = EvaluatorReport(
        perspective="fidelity",
        findings=[_finding("f2", perspective="fidelity", scope="04", video_id="vid000")],
    )
    report = aggregate_reports(0, cov, ped, fid)
    assert report.iteration == 0
    assert report.coverage is cov
    assert report.fidelity is fid
    assert {f.finding_id for f in report.all_findings} == {"f1", "f2"}


def test_aggregate_without_fidelity_uses_empty_default() -> None:
    cov = EvaluatorReport(perspective="coverage")
    ped = EvaluatorReport(perspective="pedagogy")
    report = aggregate_reports(1, cov, ped)
    assert report.fidelity.perspective == "fidelity"
    assert report.fidelity.findings == []


def test_aggregate_rejects_misslotted_report() -> None:
    cov = EvaluatorReport(perspective="coverage")
    ped = EvaluatorReport(perspective="pedagogy")
    with pytest.raises(ValueError, match="coverage slot"):
        aggregate_reports(0, ped, ped)
    with pytest.raises(ValueError, match="fidelity slot"):
        aggregate_reports(0, cov, ped, cov)


def test_route_partitions_by_scope() -> None:
    cov = EvaluatorReport(
        perspective="coverage", findings=[_finding("f1", scope="04", video_id="vidA")]
    )
    ped = EvaluatorReport(perspective="pedagogy", findings=[_finding("f2", perspective="pedagogy")])
    report = aggregate_reports(0, cov, ped)
    for_04, for_05 = route_findings(report)
    assert [f.finding_id for f in for_04] == ["f1"]
    assert [f.finding_id for f in for_05] == ["f2"]


def test_has_blocking_findings_only_counts_high() -> None:
    cov = EvaluatorReport(perspective="coverage", findings=[_finding("f1", severity="low")])
    ped = EvaluatorReport(perspective="pedagogy")
    assert has_blocking_findings(aggregate_reports(0, cov, ped)) is False
    cov2 = EvaluatorReport(perspective="coverage", findings=[_finding("f1", severity="high")])
    assert has_blocking_findings(aggregate_reports(0, cov2, ped)) is True


def test_target_video_ids_dedupe_and_sort() -> None:
    findings = [
        _finding("f1", scope="04", video_id="vidB"),
        _finding("f2", scope="04", video_id="vidA"),
        _finding("f3", scope="04", video_id="vidB"),
        _finding("f4", scope="04", video_id=None),  # ignored
    ]
    assert target_video_ids_for_04(findings) == ["vidA", "vidB"]
