"""Hands-on stage orchestrator: one long-form talk video → tutorial notes.

Sequence (per ``docs/handson-mode.md``):

    01  transcript      — existing ``run_stage_scripts`` reused verbatim
    H1  segments        — handson.segmenter (LECTURE / QA / TIPS)
    H2  step plan       — handson.planner (lossless insight assignment)
    H3  step clips      — capture.capture_step_clips (one window per step)
    H4  step bodies     — handson.steps (one LLM call per step)
    H5  MOC + summary   — handson.steps
    W   vault writes    — handson.writer (fold → validate → write)

Output lives under ``Permanent Note/09_YouTube学習_Session_only`` — a
sibling of the 08 tree with the SAME unit-directory structure (user
requirement). Only the units this mode actually produces are created:
01_Scripts_Processing_Unit (the transcript note) and 05_Synthesis (the
final MOC / step notes / QA-Tips summary / ``_meta``). 02/04 never run
here and no 03 unit note is written — clips embed directly into the step
notes, their files landing in the shared ``_assets`` tree keyed by the
same run folder name.

Degradation ladder: every LLM stage falls back deterministically
(all-LECTURE segments → per-segment fallback plan → degraded step body →
deterministic MOC/summary), so once a transcript exists the run always
produces notes. Only a missing transcript aborts.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from ..handson.planner import plan_steps
from ..handson.schemas import HandsonPlan, Insight, Segment, SegmentLabel, StepBody
from ..handson.segmenter import build_insights, classify_segments
from ..handson.steps import generate_moc_summary, generate_step_body
from ..handson.writer import (
    MOC_FILENAME,
    QA_TIPS_NOTE_FILENAME,
    step_note_filename,
    write_handson_meta,
    write_handson_moc,
    write_qa_tips_summary,
    write_step_note,
)
from ..obsidian import (
    build_frontmatter,
    format_playlist_folder_name,
    format_video_note_base,
    resolve_unique_path,
)
from ..path_safety import ensure_safe_path
from ..playlist import VideoMeta
from ..providers.base import LLMResponse
from ..sanitize import sanitize_untrusted_text
from .capture import CaptureResult, SummaryRange, capture_step_clips
from .scripts import run_stage_scripts

if TYPE_CHECKING:
    from ..services.cache import Cache

# Output root for hands-on mode (user-specified sibling of the 08 tree,
# same internal unit structure). The folder was created manually in the
# vault; every write below goes through mkdir(parents=True, exist_ok=True)
# so a pre-existing (or missing) root is equally fine.
SESSION_BASE = "Permanent Note/09_YouTube学習_Session_only"
SESSION_SCRIPTS_BASE = f"{SESSION_BASE}/01_Scripts_Processing_Unit"
SESSION_SYNTHESIS_BASE = f"{SESSION_BASE}/05_Synthesis"
META_SUBDIR = "_meta"


@dataclass(frozen=True)
class HandsonStageResult:
    """Everything one hands-on run produced (mirrors SynthesisStageResult)."""

    segments: list[Segment] = field(default_factory=list)
    insights: list[Insight] = field(default_factory=list)
    plan: HandsonPlan | None = None
    scripts_path: Path | None = None
    moc_path: Path | None = None
    step_paths: list[Path] = field(default_factory=list)
    summary_path: Path | None = None
    meta_path: Path | None = None
    capture: CaptureResult | None = None
    responses: list[LLMResponse] = field(default_factory=list)
    segment_fallback_reason: str | None = None
    transcript_source: str | None = None
    correction_cost_usd: float | None = None
    error: str | None = None

    def label_count(self, label: SegmentLabel) -> int:
        return sum(1 for s in self.segments if s.label is label)

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens or 0 for r in self.responses)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens or 0 for r in self.responses)

    @property
    def total_cache_read_tokens(self) -> int:
        return sum(r.cache_read_tokens or 0 for r in self.responses)

    @property
    def total_cache_creation_tokens(self) -> int:
        return sum(r.cache_creation_tokens or 0 for r in self.responses)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.total_cost_usd or 0.0 for r in self.responses)

    @property
    def total_duration_ms(self) -> int:
        return sum(r.duration_ms or 0 for r in self.responses)


def run_stage_handson(
    video: VideoMeta,
    *,
    run_time: datetime,
    folder_title: str,
    models: dict[str, str],
    filler_words: tuple[str, ...] = (),
    capture_format: str = "auto",
    capture_backend: Any = None,
    code_bearing: bool = False,
    correct_model: str | None = None,
    known_terms: list[tuple[str, str]] | None = None,
    use_innertube: bool = True,
    dry_run: bool = False,
    cache: Cache,
    vault_root: Path,
) -> HandsonStageResult:
    """Run the full hands-on flow for one video; never raises.

    Returns a ``HandsonStageResult`` whose ``error`` is set only when no
    notes could be produced at all (no transcript, or an unexpected
    exception). Everything else degrades per stage and keeps going.
    """
    try:
        return _run(
            video,
            run_time=run_time,
            folder_title=folder_title,
            models=models,
            filler_words=filler_words,
            capture_format=capture_format,
            capture_backend=capture_backend,
            code_bearing=code_bearing,
            correct_model=correct_model,
            known_terms=known_terms,
            use_innertube=use_innertube,
            dry_run=dry_run,
            cache=cache,
            vault_root=vault_root,
        )
    except Exception as e:
        traceback.print_exc()
        return HandsonStageResult(error=f"{type(e).__name__}: {e}")


def _run(
    video: VideoMeta,
    *,
    run_time: datetime,
    folder_title: str,
    models: dict[str, str],
    filler_words: tuple[str, ...],
    capture_format: str,
    capture_backend: Any,
    code_bearing: bool,
    correct_model: str | None,
    known_terms: list[tuple[str, str]] | None,
    use_innertube: bool,
    dry_run: bool,
    cache: Cache,
    vault_root: Path,
) -> HandsonStageResult:
    folder_name = format_playlist_folder_name(run_time, folder_title)
    note_base = format_video_note_base(run_time, video.title)

    # --- 01: transcript note under 09/01_Scripts_Processing_Unit ---------
    scripts_rel = ensure_safe_path(f"{SESSION_SCRIPTS_BASE}/{folder_name}", vault_root=vault_root)
    scripts_dir = vault_root / scripts_rel
    if dry_run:
        scripts_path = scripts_dir / f"{note_base}.md"
    else:
        scripts_dir.mkdir(parents=True, exist_ok=True)
        scripts_path = resolve_unique_path(scripts_dir, note_base, ".md")
        scripts_path.write_text(
            build_frontmatter(
                dt=run_time,
                title=video.title,
                url=video.watch_url,
                tags=["memo", "youtube", "handson"],
                extra={"video_id": video.video_id},
            ),
            encoding="utf-8",
        )

    if correct_model:
        click.echo(f"  [01] scripts (correct={correct_model})...", nl=False)
    else:
        click.echo("  [01] scripts...", nl=False)
    transcript = run_stage_scripts(
        video,
        scripts_path,
        dry_run=dry_run,
        include_code_blocks=code_bearing,
        correct_model=correct_model,
        known_terms=known_terms,
        use_innertube=use_innertube,
        cache=cache,
    )
    reason = sanitize_untrusted_text(
        transcript.fallback_reason, 500, context="transcript.fallback_reason"
    )
    click.echo(
        f" source={transcript.source.value}"
        f" snippets={len(transcript.snippets)}"
        f" lang={transcript.language or '-'}" + (f" reason=({reason})" if reason else "")
    )
    if not transcript.snippets:
        return HandsonStageResult(scripts_path=scripts_path, error="no_transcript_snippets")

    duration = video.duration or int(transcript.snippets[-1].end)
    responses: list[LLMResponse] = []

    # --- H1: segment classification --------------------------------------
    click.echo(f"  [H1] segments (model={models['handson_segment']})...", nl=False)
    segments, seg_responses, seg_fallback = classify_segments(
        video,
        transcript.snippets,
        transcript.chapters,
        duration=duration,
        model=models["handson_segment"],
        filler_words=filler_words,
        cache=cache,
    )
    responses.extend(seg_responses)
    counts = {label: sum(1 for s in segments if s.label is label) for label in SegmentLabel}
    click.echo(
        f" lecture={counts[SegmentLabel.LECTURE]}"
        f" qa={counts[SegmentLabel.QA]}"
        f" tips={counts[SegmentLabel.TIPS]}"
        + (f" FALLBACK({seg_fallback})" if seg_fallback else "")
    )
    insights = build_insights(segments, transcript.snippets)

    # --- H2: step planning ------------------------------------------------
    click.echo(f"  [H2] step plan (model={models['handson_plan']})...", nl=False)
    plan, plan_responses = plan_steps(
        video,
        segments,
        insights,
        transcript.chapters,
        duration=duration,
        model=models["handson_plan"],
        cache=cache,
    )
    responses.extend(plan_responses)
    assigned_total = sum(len(s.insight_ids) for s in plan.steps)
    click.echo(
        f" steps={len(plan.steps)}"
        f" assigned={assigned_total}"
        f" unassigned={len(plan.unassigned_insight_ids)}"
    )

    # --- H3: one window clip per step (failure never blocks) --------------
    click.echo("  [H3] step clips...", nl=False)
    ranges = [SummaryRange(s.start_sec, s.end_sec, s.label) for s in plan.steps]
    capture_result = capture_step_clips(
        video,
        ranges,
        assets_subfolder=folder_name,
        capture_format=capture_format,  # type: ignore[arg-type]
        dry_run=dry_run,
        backend=capture_backend,
        cache=cache,
        vault_root=vault_root,
    )
    if capture_result.error and not capture_result.outcomes:
        click.echo(f" SKIPPED: {capture_result.error}")
    else:
        click.echo(
            f" {capture_result.success_count}/{len(ranges)} clips"
            f" fmt={capture_result.capture_format}"
        )

    # --- H4: per-step tutorial bodies ------------------------------------
    insights_by_id = {i.insight_id: i for i in insights}
    step_bodies: list[StepBody] = []
    for pos, step in enumerate(plan.steps):
        prev_label = plan.steps[pos - 1].label if pos > 0 else None
        next_label = plan.steps[pos + 1].label if pos + 1 < len(plan.steps) else None
        assigned = [insights_by_id[iid] for iid in step.insight_ids if iid in insights_by_id]
        click.echo(
            f"  [H4] step {step.index}/{len(plan.steps)} (model={models['handson_step']})...",
            nl=False,
        )
        body, body_responses = generate_step_body(
            video,
            step,
            transcript.snippets,
            assigned,
            prev_label=prev_label,
            next_label=next_label,
            code_bearing=code_bearing,
            model=models["handson_step"],
            filler_words=filler_words,
            cache=cache,
        )
        responses.extend(body_responses)
        if body_responses:
            last = body_responses[-1]
            click.echo(f" in={last.input_tokens or 0} out={last.output_tokens or 0}")
        else:
            click.echo(" degraded (llm unavailable)")
        step_bodies.append(body)

    # --- H5: MOC + final QA/Tips summary ---------------------------------
    step_links = {b.index: step_note_filename(b.index, b.label)[: -len(".md")] for b in step_bodies}
    click.echo(f"  [H5] MOC+summary (model={models['handson_moc']})...", nl=False)
    moc_output, moc_responses = generate_moc_summary(
        video,
        plan,
        insights,
        step_bodies,
        step_links,
        model=models["handson_moc"],
        cache=cache,
    )
    responses.extend(moc_responses)
    click.echo(" done")

    if dry_run:
        return HandsonStageResult(
            segments=segments,
            insights=insights,
            plan=plan,
            scripts_path=scripts_path,
            capture=capture_result,
            responses=responses,
            segment_fallback_reason=seg_fallback,
            transcript_source=transcript.source.value,
            correction_cost_usd=transcript.correction_cost_usd,
        )

    # --- W: vault writes under 09/05_Synthesis ---------------------------
    handson_rel = ensure_safe_path(f"{SESSION_SYNTHESIS_BASE}/{folder_name}", vault_root=vault_root)
    handson_dir = vault_root / handson_rel
    handson_dir.mkdir(parents=True, exist_ok=True)

    # Outcome i maps to step i by construction (capture_step_clips keeps
    # range-positional indices), so a failed clip leaves just that step
    # without an embed.
    embed_by_index: dict[int, str] = {}
    for pos, outcome in enumerate(capture_result.outcomes):
        if outcome.image_path is not None and pos < len(plan.steps):
            embed = f"{outcome.image_path.parent.name}/{outcome.image_path.name}"
            embed_by_index[plan.steps[pos].index] = embed

    step_paths: list[Path] = []
    for body, step in zip(step_bodies, plan.steps, strict=True):
        assigned = [insights_by_id[iid] for iid in step.insight_ids if iid in insights_by_id]
        step_paths.append(
            write_step_note(
                body,
                step,
                assigned,
                embed_by_index.get(step.index),
                handson_dir,
                run_time=run_time,
                video=video,
            )
        )

    moc_path = handson_dir / MOC_FILENAME
    write_handson_moc(
        moc_output,
        moc_path,
        run_time=run_time,
        video=video,
        step_link_targets=frozenset(p.stem for p in step_paths),
        has_insights=bool(insights),
    )

    summary_path: Path | None = None
    if insights:
        stem_by_index = {
            step.index: path.stem for step, path in zip(plan.steps, step_paths, strict=True)
        }
        assigned_map = {iid: step.index for step in plan.steps for iid in step.insight_ids}
        insight_links = {
            iid: stem_by_index[idx] for iid, idx in assigned_map.items() if idx in stem_by_index
        }
        summary_path = handson_dir / QA_TIPS_NOTE_FILENAME
        write_qa_tips_summary(
            moc_output,
            insights,
            insight_links,
            summary_path,
            run_time=run_time,
            video=video,
        )

    meta_path = write_handson_meta(
        _build_meta(video, duration, seg_fallback, segments, insights, plan, capture_result),
        handson_dir / META_SUBDIR,
    )

    return HandsonStageResult(
        segments=segments,
        insights=insights,
        plan=plan,
        scripts_path=scripts_path,
        moc_path=moc_path,
        step_paths=step_paths,
        summary_path=summary_path,
        meta_path=meta_path,
        capture=capture_result,
        responses=responses,
        segment_fallback_reason=seg_fallback,
        transcript_source=transcript.source.value,
        correction_cost_usd=transcript.correction_cost_usd,
    )


def _build_meta(
    video: VideoMeta,
    duration: int,
    seg_fallback: str | None,
    segments: list[Segment],
    insights: list[Insight],
    plan: HandsonPlan,
    capture_result: CaptureResult,
) -> dict[str, Any]:
    """The ``_meta/handson_meta.json`` payload (run provenance for review)."""
    return {
        "video_id": video.video_id,
        "video_title": video.title,
        "duration_sec": duration,
        "segment_fallback_reason": seg_fallback,
        "segments": [
            {
                "start_sec": s.start_sec,
                "end_sec": s.end_sec,
                "label": s.label.value,
                "summary": s.summary,
            }
            for s in segments
        ],
        "insights": [
            {
                "insight_id": i.insight_id,
                "label": i.label.value,
                "start_sec": i.start_sec,
                "end_sec": i.end_sec,
                "summary": i.summary,
            }
            for i in insights
        ],
        "steps": [
            {
                "index": s.index,
                "label": s.label,
                "start_sec": s.start_sec,
                "end_sec": s.end_sec,
                "goal": s.goal,
                "insight_ids": list(s.insight_ids),
            }
            for s in plan.steps
        ],
        "unassigned_insight_ids": list(plan.unassigned_insight_ids),
        "capture": {
            "format": capture_result.capture_format,
            "error": capture_result.error,
            "outcomes": [
                {
                    "image": o.image_path.name if o.image_path else None,
                    "error": o.error,
                }
                for o in capture_result.outcomes
            ],
        },
    }
