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

from ..synthesis.scoring import extract_json  # reuse robust JSON extractor (DRY)

if TYPE_CHECKING:
    from ..stages.synthesis import SynthesisStageResult

Severity = Literal["info", "low", "high"]  # "high" == blocking (drives regen)
Perspective = Literal["coverage", "pedagogy"]
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
    """Aggregate of both evaluators for a single iteration."""

    iteration: int  # 0-based
    coverage: EvaluatorReport
    pedagogy: EvaluatorReport

    @property
    def all_findings(self) -> list[Finding]:
        return [*self.coverage.findings, *self.pedagogy.findings]

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

    TODO(scaffold): implement via ``extract_json(raw)`` + ``_parse_findings``.
    """
    raise NotImplementedError("scaffold: coverage parser TODO")


def parse_pedagogy_evaluator_output(raw: str) -> EvaluatorReport:
    """Parse the PedagogyEvaluator JSON into an ``EvaluatorReport``.

    Same defensive contract as ``parse_coverage_evaluator_output``.

    TODO(scaffold): implement via ``extract_json(raw)`` + ``_parse_findings``.
    """
    raise NotImplementedError("scaffold: pedagogy parser TODO")


def _parse_findings(
    data: dict[str, object],
    perspective: Perspective,
) -> list[Finding]:
    """Coerce a parsed JSON object into a list of ``Finding``.

    Coerce every field with ``str(...)`` / ``int(...)``; clamp unknown
    ``severity``/``target_scope`` to safe defaults; and DEMOTE any
    ``target_scope == "04"`` finding that lacks ``target_video_id`` to
    ``"05"`` (safety: never trigger an under-specified 04 regen).

    TODO(scaffold): implement coercion + demotion. ``extract_json`` already
    guarantees ``data`` is a dict.
    """
    raise NotImplementedError("scaffold: finding coercion TODO")


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
