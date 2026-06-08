"""Deterministic finding aggregation + 04-vs-05 routing (no LLM).

SCAFFOLD — signatures complete; bodies are stubs (TODO).

Analogous to ``synthesis.agents.compute_coverage``: a trivial, fully
deterministic Python step kept out of the LLM path. The evaluators decide
each finding's ``target_scope``; this module merges the two perspective
reports and partitions findings for the orchestrator to apply.
"""

from __future__ import annotations

from .schemas import EvaluationReport, EvaluatorReport, Finding


def aggregate_reports(
    iteration: int,
    coverage: EvaluatorReport,
    pedagogy: EvaluatorReport,
) -> EvaluationReport:
    """Merge the two perspective reports into one ``EvaluationReport``.

    TODO(scaffold): assert perspectives match their slots, then construct.
    """
    raise NotImplementedError("scaffold: aggregate TODO")


def route_findings(report: EvaluationReport) -> tuple[list[Finding], list[Finding]]:
    """Partition findings into ``(for_04, for_05)`` by ``target_scope``.

    A ``"04"`` finding lacking ``target_video_id`` must already have been
    demoted to ``"05"`` by the parser (``schemas._parse_findings``); this
    function trusts that invariant and partitions on ``target_scope`` only.

    TODO(scaffold): return ``(report.findings_for_04, report.findings_for_05)``.
    """
    raise NotImplementedError("scaffold: route TODO")


def has_blocking_findings(report: EvaluationReport) -> bool:
    """True iff any finding has ``severity == "high"`` (stop-condition).

    TODO(scaffold): return ``bool(report.blocking_findings)``.
    """
    raise NotImplementedError("scaffold: blocking predicate TODO")


def target_video_ids_for_04(findings_04: list[Finding]) -> list[str]:
    """Deduped, sorted video_ids targeted by 04-scoped findings.

    TODO(scaffold): collect non-None ``target_video_id`` into a sorted set.
    """
    raise NotImplementedError("scaffold: target id collection TODO")


__all__ = [
    "aggregate_reports",
    "has_blocking_findings",
    "route_findings",
    "target_video_ids_for_04",
]
