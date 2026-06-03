"""Evaluation phase: multi-agent, multi-perspective evaluation of the
01–05 deliverables with a bounded feedback loop (max 2 iterations).

Conceptually a phase SEPARATE from the linear 01–05 pipeline. It runs
AFTER Stage 05 and does not modify the existing Reviewer (ε) path.

Public domain types are re-exported here for convenience. The orchestrator
entrypoint ``run_stage_evaluation`` lives in ``pipeline_youtube.stages.evaluation``
(imported from there directly to avoid an import cycle with the stages package).
"""

from __future__ import annotations

from .schemas import (
    EvaluationIteration,
    EvaluationLoopResult,
    EvaluationParseError,
    EvaluationReport,
    EvaluatorReport,
    Finding,
    Perspective,
    Severity,
    TargetScope,
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
]
