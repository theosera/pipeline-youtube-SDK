"""Stage 06 — Evaluation phase orchestrator (separate from linear 01–05).

SCAFFOLD — signatures + control-flow contract are complete; bodies are
stubs (TODO). No real LLM logic yet.

The bounded feedback loop (max 2 iterations) runs AFTER Stage 05:

    for each iteration (≤ max_loops, hard cap 2):
        run 2 evaluators in parallel  → aggregate → route findings
        apply: regen targeted 04 subset (then 05) and/or regen 05 only
        re-evaluate
    stop when no blocking (high) findings remain OR max_loops reached

It does NOT modify the existing Reviewer (ε). ``learning_md_bodies`` is
treated as an index-aligned, copy-on-write working list: when a 04 subset
is regenerated, only the targeted indices are replaced, preserving the
1:1 alignment with ``videos`` that ``run_stage_synthesis`` /
``format_learning_materials`` require.

04 subset regen + checkpoint: the 04 checkpoint is a filesystem-presence
check read only at the START of a ``main.py`` run — there is no in-loop
checkpoint gate here. So regen needs NO checkpoint mutation: calling
``run_stage_learning`` overwrites the 04 md atomically (precedent:
``--force-video``), leaving a valid checkpoint marker.

Cost envelope: worst case ≈ ``max_loops × (K stage_04 calls + 1 full
synthesis)`` where K = deduped targeted videos. Bounded by the hard
``max_loops ≤ 2`` cap; mitigate K by acting only on ``high`` findings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..evaluation.schemas import EvaluationLoopResult
from ..playlist import VideoMeta
from ..synthesis.agents import AgentCallResult
from .synthesis import SynthesisStageResult

_MAX_LOOPS_HARD_CAP = 2


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
    summary_md_paths: dict[str, Path] | None = None,
    capture_md_paths: dict[str, Path] | None = None,
    learning_md_paths: dict[str, Path] | None = None,
    max_loops: int = 2,
    model: str = "sonnet",
    eval_models: dict[str, str] | None = None,
    agent_models: dict[str, str] | None = None,
    folder_name_override: str | None = None,
    code_bearing: bool = False,
    synthesis_timeout: int | None = None,
    profile: str | None = None,
    dry_run: bool = False,
) -> EvaluationStageResult:
    """Run the bounded evaluation feedback loop after Stage 05.

    See module docstring for the loop contract and stop conditions
    (``no_blocking_findings`` / ``max_loops_reached`` / ``regen_failed``).

    ``max_loops`` is clamped to ``[0, _MAX_LOOPS_HARD_CAP]``; 0 short-circuits
    to a passthrough (``skipped=True``) leaving ``synthesis_result`` untouched.

    Degraded resume mode: under ``--synthesis-only`` the 02/03/04 source
    paths may be absent. When ``learning_md_paths`` is ``None`` a 04-scoped
    finding cannot be applied, so it is safely downgraded to a 05-only regen
    (logged), and ``summary_md_paths=None`` disables the coverage evaluator's
    optional fidelity input.

    TODO(scaffold): implement the loop (evaluate→aggregate→route→apply),
    artifact writing, and stop bookkeeping.
    """
    raise NotImplementedError("scaffold: evaluation loop TODO")


def _regen_learning_subset(
    videos: list[VideoMeta],
    bodies: list[str],
    target_video_ids: list[str],
    learning_md_paths: dict[str, Path] | None,
    *,
    run_time: datetime,
    code_bearing: bool,
    model: str,
    dry_run: bool,
) -> list[str]:
    """Re-run Stage 04 for ONLY ``target_video_ids``; return a new
    index-aligned ``bodies`` list (untargeted entries copied unchanged).

    Resolves each target's 02/03/04 paths the way ``main.py`` does
    (``_find_summary_md`` for 02; ``pipeline.compute_note_paths`` for
    03/04), calls ``stages.learning.run_stage_learning`` (atomic overwrite,
    bypasses the presence-based checkpoint), then stores
    ``learning._strip_frontmatter(learning_md_path.read_text())`` back at
    the SAME index (resolve id→index via ``videos``, never by finding order).

    Correctness invariant: ``len(result) == len(videos)`` always.

    TODO(scaffold): implement subset regen with copy-on-write alignment.
    """
    raise NotImplementedError("scaffold: 04 subset regen TODO")


def _regen_synthesis(
    videos: list[VideoMeta],
    bodies: list[str],
    *,
    run_time: datetime,
    playlist_title: str,
    folder_name_override: str | None,
    model: str,
    agent_models: dict[str, str] | None,
    synthesis_timeout: int | None,
    profile: str | None,
    dry_run: bool,
) -> SynthesisStageResult:
    """Thin wrapper over ``run_stage_synthesis`` that re-runs Stage 05 with
    the current working ``bodies`` into the SAME playlist folder
    (``folder_name_override``), overwriting MOC/chapters in place.

    TODO(scaffold): forward to ``run_stage_synthesis``.
    """
    raise NotImplementedError("scaffold: 05 regen TODO")


__all__ = [
    "EvaluationStageResult",
    "run_stage_evaluation",
]
