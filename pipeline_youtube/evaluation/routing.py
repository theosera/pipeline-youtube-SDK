"""Deterministic finding aggregation + 04-vs-05 routing (no LLM).

Analogous to ``synthesis.agents.compute_coverage``: a trivial, fully
deterministic Python step kept out of the LLM path. The evaluators decide
each finding's ``target_scope``; this module merges the perspective
reports and partitions findings for the orchestrator to apply.
"""

from __future__ import annotations

from .schemas import EvaluationReport, EvaluatorReport, Finding


def aggregate_reports(
    iteration: int,
    coverage: EvaluatorReport,
    pedagogy: EvaluatorReport,
    fidelity: EvaluatorReport | None = None,
) -> EvaluationReport:
    """Merge the perspective reports into one ``EvaluationReport``.

    Asserts each report sits in its matching slot (a mis-slotted report is
    a programming error, not data noise). ``fidelity`` is optional so a
    coverage+pedagogy-only run keeps working; when omitted the
    ``EvaluationReport`` default (an empty fidelity report) applies.
    """
    if coverage.perspective != "coverage":
        raise ValueError(f"coverage slot got perspective {coverage.perspective!r}")
    if pedagogy.perspective != "pedagogy":
        raise ValueError(f"pedagogy slot got perspective {pedagogy.perspective!r}")
    if fidelity is not None and fidelity.perspective != "fidelity":
        raise ValueError(f"fidelity slot got perspective {fidelity.perspective!r}")
    if fidelity is None:
        return EvaluationReport(iteration=iteration, coverage=coverage, pedagogy=pedagogy)
    return EvaluationReport(
        iteration=iteration, coverage=coverage, pedagogy=pedagogy, fidelity=fidelity
    )


def route_findings(report: EvaluationReport) -> tuple[list[Finding], list[Finding]]:
    """Partition findings into ``(for_04, for_05)`` by ``target_scope``.

    A ``"04"`` finding lacking ``target_video_id`` must already have been
    demoted to ``"05"`` by the parser (``schemas._parse_findings``); this
    function trusts that invariant and partitions on ``target_scope`` only.
    """
    return (report.findings_for_04, report.findings_for_05)


def has_blocking_findings(report: EvaluationReport) -> bool:
    """True iff any finding has ``severity == "high"`` (stop-condition)."""
    return bool(report.blocking_findings)


def target_video_ids_for_04(findings_04: list[Finding]) -> list[str]:
    """Deduped, sorted video_ids targeted by 04-scoped findings.

    Ignores any finding whose ``target_video_id`` is ``None`` (such a
    finding should already have been demoted to ``"05"`` upstream).
    """
    return sorted({f.target_video_id for f in findings_04 if f.target_video_id is not None})


__all__ = [
    "aggregate_reports",
    "has_blocking_findings",
    "route_findings",
    "target_video_ids_for_04",
]
