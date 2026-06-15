"""Data structures and JSON parsing for the Evaluation phase.

SCAFFOLD — schemas are complete; ``parse_*`` bodies are stubs (TODO).

Idiom note (documented deviation): like ``synthesis/scoring.py`` — the
structural sibling of this module — the evaluation domain objects use
``@dataclass(frozen=True)`` + ``extract_json()`` + hand-written
``parse_*`` functions rather than Pydantic ``BaseModel``. CLAUDE.md is
Pydantic-first; this phase deliberately follows the local synthesis
idiom because it is the same agent/parse/orchestrate shape.
``extract_json`` is reused from ``synthesis.scoring`` (DRY).

Each evaluator sub-agent emits JSON with a well-defined shape:

- **CoverageEvaluator** outputs ``{"summary": str, "findings": Finding[]}``
  judging topic coverage, important-concept omissions, inter-chapter overlap.
- **PedagogyEvaluator** outputs the same shape judging chapter ordering,
  difficulty progression, clarity, learner usefulness.

A ``Finding`` carries ``target_scope`` ("04" | "05") so the orchestrator
knows where to write the fix back (per-video Stage 04 regen vs Stage 05
re-synthesis). See ``stages/evaluation.py`` and ``evaluation/routing.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from ..synthesis.scoring import (  # reuse robust JSON extractor (DRY)
    SynthesisParseError,
    extract_json,
)

if TYPE_CHECKING:
    from ..stages.synthesis import SynthesisStageResult

Severity = Literal["info", "low", "high"]  # "high" == blocking (drives regen)
Perspective = Literal["coverage", "pedagogy", "fidelity"]
TargetScope = Literal["04", "05"]  # write-back target for a finding's fix


class EvaluationParseError(RuntimeError):
    """Raised when an evaluator's JSON output cannot be parsed.

    Parsers should NEVER raise this mid-loop (evaluation is advisory).
    It exists for direct/test callers that opt into strict parsing.
    """


@dataclass(frozen=True)
class Finding:
    """A single evaluation defect routed to a write-back target.

    ``target_scope`` decides where the orchestrator applies the fix:

    - ``"04"``: defect localized to one video's source material; regen
      that video's Stage 04 note, then re-run Stage 05.
    - ``"05"``: cross-cutting structural/synthesis defect; regen only
      Stage 05.

    ``target_video_id`` MUST be set when ``target_scope == "04"``; a
    ``"04"`` finding without it is demoted to ``"05"`` by the parser so a
    malformed finding can never trigger an under-specified 04 regen.
    """

    finding_id: str  # e.g. "f001"
    perspective: Perspective
    severity: Severity
    target_scope: TargetScope
    description: str
    suggested_fix: str
    target_video_id: str | None = None  # required iff target_scope == "04"
    topic_ids: list[str] = field(default_factory=list)  # optional coverage linkage
    chapter_index: int | None = None  # optional Stage 05 linkage


@dataclass(frozen=True)
class EvaluatorReport:
    """Output of ONE evaluator sub-agent for ONE iteration."""

    perspective: Perspective
    findings: list[Finding] = field(default_factory=list)
    summary: str = ""

    @property
    def blocking_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "high"]


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate of the evaluators for a single iteration.

    ``fidelity`` (proper-noun / mis-transcription perspective) is the
    third evaluator slot. It defaults to an empty report so callers that
    only run coverage + pedagogy (and the existing scaffold tests) keep
    constructing this without change.
    """

    iteration: int  # 0-based
    coverage: EvaluatorReport
    pedagogy: EvaluatorReport
    fidelity: EvaluatorReport = field(
        default_factory=lambda: EvaluatorReport(perspective="fidelity")
    )

    @property
    def all_findings(self) -> list[Finding]:
        return [*self.coverage.findings, *self.pedagogy.findings, *self.fidelity.findings]

    @property
    def blocking_findings(self) -> list[Finding]:
        return [f for f in self.all_findings if f.severity == "high"]

    @property
    def findings_for_04(self) -> list[Finding]:
        return [f for f in self.all_findings if f.target_scope == "04"]

    @property
    def findings_for_05(self) -> list[Finding]:
        return [f for f in self.all_findings if f.target_scope == "05"]


@dataclass(frozen=True)
class EvaluationIteration:
    """One full evaluate→route→apply cycle record (for artifacts/audit)."""

    iteration: int
    report: EvaluationReport
    regenerated_video_ids: list[str] = field(default_factory=list)
    synthesis_rerun: bool = False
    stopped: bool = False
    stop_reason: str | None = None


@dataclass(frozen=True)
class EvaluationLoopResult:
    """Returned by ``run_stage_evaluation`` across all iterations."""

    iterations: list[EvaluationIteration] = field(default_factory=list)
    loops_run: int = 0
    final_synthesis: SynthesisStageResult | None = None
    skipped: bool = False
    skip_reason: str | None = None
    error: str | None = None


# =====================================================
# Parsers (mirror synthesis.scoring.parse_reviewer_output's defensive style)
# =====================================================


def parse_coverage_evaluator_output(raw: str) -> EvaluatorReport:
    """Parse the CoverageEvaluator JSON into an ``EvaluatorReport``.

    Defensive (advisory): a non-dict / malformed payload yields an empty
    report rather than raising, so one bad evaluator can never abort the
    loop. Mirrors ``synthesis.scoring.parse_reviewer_output``.
    """
    return _parse_evaluator_output(raw, "coverage")


def parse_pedagogy_evaluator_output(raw: str) -> EvaluatorReport:
    """Parse the PedagogyEvaluator JSON into an ``EvaluatorReport``.

    Same defensive contract as ``parse_coverage_evaluator_output``.
    """
    return _parse_evaluator_output(raw, "pedagogy")


def _parse_evaluator_output(raw: str, perspective: Perspective) -> EvaluatorReport:
    """Shared defensive parse for any single-perspective evaluator output.

    Never raises: a missing/garbled JSON payload (``extract_json`` raising
    ``SynthesisParseError``) yields an empty report for ``perspective`` so
    one bad evaluator can never abort the bounded loop.
    """
    try:
        data = extract_json(raw)
    except SynthesisParseError:
        return EvaluatorReport(perspective=perspective)
    summary = str(data.get("summary") or "")
    findings = _parse_findings(data, perspective)
    return EvaluatorReport(perspective=perspective, findings=findings, summary=summary)


def _parse_findings(
    data: dict[str, object],
    perspective: Perspective,
) -> list[Finding]:
    """Coerce a parsed JSON object into a list of ``Finding``.

    Coerces every field defensively; clamps unknown ``severity`` to
    ``"info"`` and unknown ``target_scope`` to ``"05"`` (both non-blocking
    / safe); forces ``perspective`` to the evaluator's fixed role
    (ignoring any self-reported value); and DEMOTES any
    ``target_scope == "04"`` finding that lacks ``target_video_id`` to
    ``"05"`` so a malformed finding can never trigger an under-specified
    04 regen. A non-list ``findings`` or a non-dict element is skipped.
    """
    raw_findings = data.get("findings")
    if not isinstance(raw_findings, list):
        return []
    findings: list[Finding] = []
    for i, item in enumerate(raw_findings):
        if not isinstance(item, dict):
            continue
        findings.append(_coerce_finding(item, perspective, index=i))
    return findings


def _coerce_finding(
    item: dict[str, object],
    perspective: Perspective,
    *,
    index: int,
) -> Finding:
    """Build one ``Finding`` from a raw dict with safe coercion + demotion."""
    severity_raw = str(item.get("severity") or "").lower()
    severity: Severity = severity_raw if severity_raw in {"info", "low", "high"} else "info"  # type: ignore[assignment]

    scope_raw = str(item.get("target_scope") or "").strip()
    target_scope: TargetScope = scope_raw if scope_raw in {"04", "05"} else "05"  # type: ignore[assignment]

    video_raw = item.get("target_video_id")
    target_video_id = str(video_raw) if isinstance(video_raw, str) and video_raw.strip() else None

    # Safety demotion: a 04-scoped finding with no concrete video target
    # cannot drive a 04 regen — downgrade it to a 05-only finding.
    if target_scope == "04" and target_video_id is None:
        target_scope = "05"

    chapter_raw = item.get("chapter_index")
    chapter_index = (
        chapter_raw if isinstance(chapter_raw, int) and not isinstance(chapter_raw, bool) else None
    )

    topic_ids_raw = item.get("topic_ids")
    topic_ids = [str(t) for t in topic_ids_raw if t] if isinstance(topic_ids_raw, list) else []

    return Finding(
        finding_id=str(item.get("finding_id") or f"f{index + 1:03d}"),
        perspective=perspective,
        severity=severity,
        target_scope=target_scope,
        description=str(item.get("description") or ""),
        suggested_fix=str(item.get("suggested_fix") or ""),
        target_video_id=target_video_id,
        topic_ids=topic_ids,
        chapter_index=chapter_index,
    )


__all__ = [
    "EvaluationIteration",
    "EvaluationLoopResult",
    "EvaluationParseError",
    "EvaluationReport",
    "EvaluatorReport",
    "Finding",
    "Perspective",
    "Severity",
    "TargetScope",
    "extract_json",
    "parse_coverage_evaluator_output",
    "parse_pedagogy_evaluator_output",
]
