"""Evaluation schema behavior + parser scaffold markers.

The frozen-dataclass routing/severity properties are IMPLEMENTED and tested
here. The ``parse_*`` functions are scaffold stubs (logic TODO) and are
marked skipped until implemented.
"""

from __future__ import annotations

import pytest

from pipeline_youtube.evaluation.schemas import (
    EvaluationReport,
    EvaluatorReport,
    Finding,
    parse_coverage_evaluator_output,
)


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


def test_evaluator_report_blocking_filters_high() -> None:
    report = EvaluatorReport(
        perspective="coverage",
        findings=[_finding("f1", severity="high"), _finding("f2", severity="low")],
    )
    assert [f.finding_id for f in report.blocking_findings] == ["f1"]


def test_evaluation_report_partitions_by_scope() -> None:
    cov = EvaluatorReport(
        perspective="coverage",
        findings=[_finding("f1", scope="04", video_id="vid0000000A")],
    )
    ped = EvaluatorReport(
        perspective="pedagogy",
        findings=[_finding("f2", perspective="pedagogy", scope="05")],
    )
    report = EvaluationReport(iteration=0, coverage=cov, pedagogy=ped)

    assert [f.finding_id for f in report.findings_for_04] == ["f1"]
    assert [f.finding_id for f in report.findings_for_05] == ["f2"]
    assert {f.finding_id for f in report.all_findings} == {"f1", "f2"}
    assert {f.finding_id for f in report.blocking_findings} == {"f1", "f2"}


def test_fidelity_slot_defaults_to_empty_and_merges_into_findings() -> None:
    cov = EvaluatorReport(perspective="coverage", findings=[_finding("f1", scope="05")])
    ped = EvaluatorReport(perspective="pedagogy")
    # Constructed WITHOUT fidelity (backward compat) -> empty fidelity slot.
    report = EvaluationReport(iteration=0, coverage=cov, pedagogy=ped)
    assert report.fidelity.perspective == "fidelity"
    assert report.fidelity.findings == []

    fid = EvaluatorReport(
        perspective="fidelity",
        findings=[_finding("f9", perspective="fidelity", scope="04", video_id="vid000")],
    )
    report2 = EvaluationReport(iteration=0, coverage=cov, pedagogy=ped, fidelity=fid)
    assert {f.finding_id for f in report2.all_findings} == {"f1", "f9"}
    assert [f.finding_id for f in report2.findings_for_04] == ["f9"]
    assert [f.finding_id for f in report2.blocking_findings] == ["f1", "f9"]


@pytest.mark.skip(reason="scaffold: parser logic TODO")
def test_parse_coverage_round_trip() -> None:
    raw = '{"summary": "s", "findings": [{"finding_id": "f001", '
    '"perspective": "coverage", "severity": "high", "target_scope": "05", '
    '"description": "d", "suggested_fix": "x"}]}'
    report = parse_coverage_evaluator_output(raw)
    assert report.perspective == "coverage"
    assert report.findings[0].finding_id == "f001"


@pytest.mark.skip(reason="scaffold: parser logic TODO")
def test_parse_malformed_defaults_to_empty() -> None:
    report = parse_coverage_evaluator_output("not json")
    assert report.findings == []


@pytest.mark.skip(reason="scaffold: parser logic TODO")
def test_finding_04_without_video_id_demoted_to_05() -> None:
    raw = '{"findings": [{"finding_id": "f1", "perspective": "coverage", '
    '"severity": "high", "target_scope": "04", "description": "d", '
    '"suggested_fix": "x"}]}'
    report = parse_coverage_evaluator_output(raw)
    assert report.findings[0].target_scope == "05"
