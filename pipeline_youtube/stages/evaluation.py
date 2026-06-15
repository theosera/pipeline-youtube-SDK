"""Stage 06 — Evaluation phase orchestrator (separate from linear 01–05).

ADVISORY MODE (current): runs the evaluators ONCE after Stage 05 and
writes a report; it does NOT regenerate 04/05 and returns the synthesis
result unchanged. Rationale: Stage 02's deterministic glossary
normalization already fixes proper nouns at the source, so the
cost/regression risk of an LLM-driven regeneration loop is not justified
— "detect and record" captures the value at zero risk. The per-video 04
regeneration + 05 re-synthesis helpers below remain as documented future
stubs (an opt-in regen mode), unused by the advisory path.

One advisory pass::

    run coverage (LLM) + pedagogy (LLM) + fidelity (deterministic)
      → aggregate (routing) → write loop_0.json + summary.md
      → return synthesis_result unchanged

It does NOT modify the existing Reviewer (ε). ``learning_md_bodies`` is
index-aligned with ``videos`` (the 1:1 contract ``format_learning_materials``
requires). Every evaluator is advisory: any failure degrades to an empty
report rather than aborting the stage.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..evaluation.evaluators import call_coverage_evaluator, call_pedagogy_evaluator
from ..evaluation.fidelity import scan_fidelity
from ..evaluation.report import resolve_eval_dir, write_evaluation_loop, write_evaluation_summary
from ..evaluation.routing import aggregate_reports
from ..evaluation.schemas import (
    EvaluationIteration,
    EvaluationLoopResult,
    EvaluatorReport,
    Perspective,
)
from ..glossary.schema import Glossary
from ..obsidian import format_playlist_folder_name
from ..playlist import VideoMeta
from ..synthesis.agents import AgentCallResult
from .synthesis import SynthesisStageResult


@dataclass(frozen=True)
class EvaluationStageResult:
    """Returned by ``run_stage_evaluation``.

    ``synthesis_result`` is the FINAL (post-evaluation) Stage 05 output the
    caller should adopt in place of the pre-evaluation one.
    """

    loop_result: EvaluationLoopResult
    synthesis_result: SynthesisStageResult
    report_paths: list[Path] = field(default_factory=list)
    summary_path: Path | None = None
    agent_results: list[AgentCallResult] = field(default_factory=list)


def run_stage_evaluation(
    videos: list[VideoMeta],
    learning_md_bodies: list[str],
    synthesis_result: SynthesisStageResult,
    *,
    run_time: datetime,
    playlist_title: str,
    max_loops: int = 0,
    model: str = "sonnet",
    eval_models: dict[str, str] | None = None,
    glossary: Glossary | None = None,
    summary_bodies: dict[str, str] | None = None,
    folder_name_override: str | None = None,
    dry_run: bool = False,
) -> EvaluationStageResult:
    """Run ONE advisory evaluation pass after Stage 05 (no regeneration).

    ``max_loops <= 0`` short-circuits to a passthrough (``skipped=True``)
    leaving ``synthesis_result`` untouched. Any positive value enables the
    single advisory pass (there is no loop — advisory mode never mutates,
    so re-evaluating would be redundant).

    Runs coverage (LLM) + pedagogy (LLM) + fidelity (deterministic, only
    when ``glossary`` is given), aggregates them, and — unless ``dry_run``
    — writes ``loop_0.json`` + ``summary.md`` under the Stage 05
    ``_meta/evaluation`` dir. ``synthesis_result`` is returned unchanged.

    Every evaluator is advisory: a failure degrades to an empty report for
    that perspective rather than aborting the stage.
    """
    if max_loops <= 0:
        return EvaluationStageResult(
            loop_result=EvaluationLoopResult(
                loops_run=0,
                final_synthesis=synthesis_result,
                skipped=True,
                skip_reason="eval_loop=0",
            ),
            synthesis_result=synthesis_result,
        )

    em = eval_models or {}
    agent_results: list[AgentCallResult] = []

    coverage = _safe_evaluate(
        "coverage",
        lambda: call_coverage_evaluator(
            videos,
            learning_md_bodies,
            synthesis_result,
            model=em.get("eval_coverage", model),
            playlist_title=playlist_title,
            summary_bodies=summary_bodies,
        ),
        agent_results,
    )
    pedagogy = _safe_evaluate(
        "pedagogy",
        lambda: call_pedagogy_evaluator(
            videos,
            learning_md_bodies,
            synthesis_result,
            model=em.get("eval_pedagogy", model),
            playlist_title=playlist_title,
        ),
        agent_results,
    )
    fidelity = _run_fidelity(videos, learning_md_bodies, glossary)

    report = aggregate_reports(0, coverage, pedagogy, fidelity)
    iteration = EvaluationIteration(
        iteration=0,
        report=report,
        regenerated_video_ids=[],
        synthesis_rerun=False,
        stopped=True,
        stop_reason="advisory: report only, no regeneration",
    )
    loop_result = EvaluationLoopResult(
        iterations=[iteration],
        loops_run=1,
        final_synthesis=synthesis_result,
    )

    report_paths: list[Path] = []
    summary_path: Path | None = None
    if not dry_run:
        folder = folder_name_override or format_playlist_folder_name(run_time, playlist_title)
        eval_dir = resolve_eval_dir(folder)
        report_paths.append(write_evaluation_loop(iteration, eval_dir))
        summary_path = write_evaluation_summary(loop_result, eval_dir)

    return EvaluationStageResult(
        loop_result=loop_result,
        synthesis_result=synthesis_result,
        report_paths=report_paths,
        summary_path=summary_path,
        agent_results=agent_results,
    )


def _safe_evaluate(
    perspective: Perspective,
    call: Callable[[], tuple[EvaluatorReport, AgentCallResult]],
    agent_results: list[AgentCallResult],
) -> EvaluatorReport:
    """Run one LLM evaluator; on ANY failure return an empty report.

    Advisory contract: a crashed/timed-out evaluator must never abort the
    evaluation stage (the broad ``except`` mirrors the Reviewer fallback in
    ``stages/synthesis.py``).
    """
    try:
        report, result = call()
    except Exception:
        return EvaluatorReport(perspective=perspective)
    agent_results.append(result)
    return report


def _run_fidelity(
    videos: list[VideoMeta],
    learning_md_bodies: list[str],
    glossary: Glossary | None,
) -> EvaluatorReport:
    """Deterministic fidelity scan; empty report when no glossary / on error."""
    if glossary is None:
        return EvaluatorReport(perspective="fidelity")
    try:
        return scan_fidelity(videos, learning_md_bodies, glossary)
    except Exception:
        return EvaluatorReport(perspective="fidelity")


__all__ = [
    "EvaluationStageResult",
    "run_stage_evaluation",
]
