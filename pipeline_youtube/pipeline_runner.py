"""ExecutionPlan に沿った処理実行 (pipeline runner)。

「現場監督」: sub-agent 親の分散起動、worker のシャード切り出し、
checkpoint / resume の選別、Stage 01-04 の逐次・並列実行、固有名詞シート更新、
そして Stage 05 (``synthesis_runner``) への接続と結果レポートを配線する。
動画 1 本の HOW は ``video_processing`` が、各段の HOW は ``stages/`` が持つ。
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import click

from .checkpoint import get_completed_video_ids
from .cli_types import CliRequest, ExecutionPlan, ResolvedInput, Runtime
from .parallel import orchestrate_sub_agents, strip_cli_option
from .playlist import VideoMeta
from .proper_noun_sheet import (
    _promote_corrections_to_glossary,
    _proper_noun_sheet_path,
    _update_proper_noun_sheet,
)
from .reporting import report_costs, report_synthesis, report_video_summary
from .resume import (
    _collect_existing_learning_bodies,
    _filter_to_reviewed,
    _load_existing_04_body,
)
from .run_result import VideoRunResult
from .stages.scripts import DEFAULT_TRANSCRIPT_CONCURRENCY, warm_transcript_cache
from .synthesis.agents import compute_synthesis_timeouts
from .synthesis_runner import run_synthesis
from .video_processing import _process_video, _run_videos_concurrent


def run_pipeline(
    request: CliRequest, runtime: Runtime, resolved: ResolvedInput, plan: ExecutionPlan
) -> None:
    """Drive stages 01-04 → 05 according to the execution plan."""
    videos = resolved.videos

    # Sub-agent orchestration (opt-in via --sub-agents N; default 1 keeps the
    # original single-process flow). Splits the playlist into N contiguous
    # shards, runs stages 01-04 as independent parallel worker processes, then
    # runs Stage 05 once over the merged output. See docs/sub-agents.md.
    if plan.is_sub_agent_parent:
        click.echo(f"run_time: {plan.run_time.isoformat(timespec='seconds')}")
        exit_code = orchestrate_sub_agents(
            total_videos=len(videos),
            shard_count=request.sub_agents,
            run_time=plan.run_time,
            logs_dir=runtime.logs_dir,
            base_argv=strip_cli_option(sys.argv[1:], "--sub-agents"),
            run_synthesis=plan.run_synthesis,
            code_bearing=resolved.code_bearing,
        )
        sys.exit(exit_code)

    # Sub-agent shard slicing (internal --video-range). Restricts the per-video
    # work below to this shard's contiguous slice; genre is already pinned.
    if plan.video_range is not None:
        shard_start, shard_end = plan.video_range
        videos = videos[shard_start:shard_end]
        click.echo(
            f"sub-agent shard: videos [{shard_start}:{shard_end}] → {len(videos)} this shard"
        )
        if not videos:
            click.echo("No videos in this shard; nothing to do.")
            return

    est_timeouts = compute_synthesis_timeouts(len(videos), override=runtime.synthesis_timeout)
    total_duration = sum(v.duration or 0 for v in videos)
    click.echo(
        f"synthesis_estimate: {len(videos)} videos"
        f" → timeout α={est_timeouts['alpha']}s β={est_timeouts['beta']}s"
        f" leader={est_timeouts['leader']}s"
        + (f", total_duration={total_duration // 60}min" if total_duration else "")
    )

    run_time = plan.run_time
    click.echo(f"run_time: {run_time.isoformat(timespec='seconds')}")

    proper_noun_sheet_path, known_terms = _prepare_proper_noun_sheet(
        runtime, videos, run_time, plan
    )

    results: list[VideoRunResult] = []
    selection = _select_synthesis_inputs(
        request,
        runtime,
        resolved,
        videos,
        run_time,
        proper_noun_sheet_path,
        known_terms,
        results,
        plan,
    )
    if selection is None:
        return
    synthesis_videos, synthesis_bodies, folder_override = selection

    synthesis_result = run_synthesis(
        request,
        runtime,
        resolved,
        run_time,
        synthesis_videos,
        synthesis_bodies,
        folder_override,
        proper_noun_sheet_path,
    )
    report_synthesis(synthesis_result)

    # NOTE: the evaluation phase (run_stage_evaluation, gated by --eval-loop)
    # hooks in here after Stage 05 once implemented. It is rejected up front
    # while scaffolded — see the --eval-loop guard in cli_validation.

    if plan.run_video_stages:
        report_costs(results, synthesis_result)


def _prepare_proper_noun_sheet(
    runtime: Runtime,
    videos: list[VideoMeta],
    run_time: datetime,
    plan: ExecutionPlan,
) -> tuple[Path | None, list[tuple[str, str]] | None]:
    """Load the per-playlist proper-noun sheet and seed known terms / glossary.

    On load we (a) feed already-known terms to Stage 01b so they skip the web
    search, and (b) promote the user's corrections into glossary.json. Gated on
    transcript_correction. Returns (sheet_path, known_terms) — both None when
    correction is off or under --dry-run.
    """
    from .glossary import known_pairs, load_sheet

    if not plan.allow_proper_noun_sheet:
        return None, None
    proper_noun_sheet_path = _proper_noun_sheet_path(videos[0], run_time)
    start_sheet = load_sheet(proper_noun_sheet_path)
    known_terms = known_pairs(start_sheet) or None
    if runtime.cfg.glossary_path is not None:
        promoted = _promote_corrections_to_glossary(start_sheet, runtime.cfg.glossary_path)
        if promoted:
            click.echo(
                f"glossary: promoted {promoted} corrected term(s) into {runtime.cfg.glossary_path.name}"
            )
    return proper_noun_sheet_path, known_terms


def _select_synthesis_inputs(
    request: CliRequest,
    runtime: Runtime,
    resolved: ResolvedInput,
    videos: list[VideoMeta],
    run_time: datetime,
    proper_noun_sheet_path: Path | None,
    known_terms: list[tuple[str, str]] | None,
    results: list[VideoRunResult],
    plan: ExecutionPlan,
) -> tuple[list[VideoMeta], list[str], str | None] | None:
    """Pick the (videos, bodies, folder_override) feeding Stage 05.

    --synthesis-only loads existing 04 md; otherwise runs stages 01-04 over the
    checkpoint/resume-filtered set. ``results`` is populated in place for the
    cost breakdown. Returns None to signal an early stop (nothing to synthesize).
    """
    if not plan.run_video_stages:
        click.echo("\n=== --synthesis-only: loading existing 04 md files ===")
        matched_videos, matched_bodies, folder_override = _collect_existing_learning_bodies(
            videos, resolved.playlist_title, run_time
        )
        click.echo(f"matched: {len(matched_videos)}/{len(videos)} videos")
        if len(matched_videos) < request.min_playlist_size:
            click.echo(
                f"[skip] only {len(matched_videos)} matched (< {request.min_playlist_size}), "
                "stage 05 skipped"
            )
            return None
        return matched_videos, matched_bodies, folder_override

    succeeded = _process_all_videos(
        request,
        runtime,
        resolved,
        videos,
        run_time,
        proper_noun_sheet_path,
        known_terms,
        results,
        plan,
    )
    if succeeded is None:
        return None
    return [r.video for r in succeeded], [r.learning_md_body or "" for r in succeeded], None


def _process_all_videos(
    request: CliRequest,
    runtime: Runtime,
    resolved: ResolvedInput,
    videos: list[VideoMeta],
    run_time: datetime,
    proper_noun_sheet_path: Path | None,
    known_terms: list[tuple[str, str]] | None,
    results: list[VideoRunResult],
    plan: ExecutionPlan,
) -> list[VideoRunResult] | None:
    """Run stages 01-04 over the checkpoint/resume-filtered set.

    Returns the succeeded results, or None when the run should stop before
    Stage 05 (--stop-after-capture / --skip-synthesis / too few succeeded).
    """
    cfg = runtime.cfg
    playlist_title = resolved.playlist_title

    # Checkpoint: detect already-completed videos in one pass
    force_set = set(request.force_video)
    completed_ids = (
        get_completed_video_ids(playlist_title, run_time, vault_root=runtime.vault_root)
        if plan.allow_checkpoint
        else set()
    )
    if completed_ids:
        skippable = completed_ids - force_set
        if skippable:
            click.echo(f"checkpoint: {len(skippable)} videos already complete, will skip")

    # Separate videos into skip (checkpoint) and process lists
    to_process: list[tuple[int, VideoMeta]] = []
    for i, video in enumerate(videos, 1):
        if video.video_id in completed_ids and video.video_id not in force_set:
            click.echo(f"\n[{i}/{len(videos)}] {video.video_id} {video.title}")
            click.echo("  [skip] checkpoint: stage 04 already exists")
            body = _load_existing_04_body(video.video_id, playlist_title, run_time)
            results.append(VideoRunResult(video=video, learning_md_body=body))
        else:
            to_process.append((i, video))

    if plan.filter_reviewed_only:
        # Phase 3: filter to videos whose 02_Summary.md has `reviewed: true`.
        to_process = _filter_to_reviewed(to_process, playlist_title, run_time)

    # Warm the transcript cache for all to-be-processed videos up front, at a
    # higher fan-out than --concurrency. Skipped under --resume-reviewed (Stage
    # 01 doesn't run) and a no-op when the cache is disabled or Whisper is the
    # only available tier. Skipped under --local-media (warm-up fetches YouTube
    # captions, which the offline path never uses).
    if to_process and plan.allow_transcript_warmup:
        warm_conc = (
            request.transcript_concurrency
            or cfg.transcript_concurrency
            or DEFAULT_TRANSCRIPT_CONCURRENCY
        )
        warmed = warm_transcript_cache(
            [v for _, v in to_process],
            concurrency=warm_conc,
            use_innertube=cfg.use_innertube,
            cache=runtime.cache,
        )
        if warmed:
            click.echo(f"transcript warm-up: cached {warmed}/{len(to_process)} captions")

    # Process remaining videos
    if to_process and request.concurrency > 1:
        process_videos = [v for _, v in to_process]
        concurrent_results = asyncio.run(
            _run_videos_concurrent(
                process_videos,
                run_time,
                concurrency=request.concurrency,
                dry_run=plan.dry_run,
                capture_format=request.capture_format,
                models=runtime.models,
                filler_words=runtime.filler_words,
                stop_after_capture=plan.stop_after_capture,
                capture_backend=runtime.capture_backend,
                code_bearing=resolved.code_bearing,
                glossary=cfg.glossary,
                media_map=resolved.media_map,
                correct_transcript=cfg.transcript_correction,
                known_terms=known_terms,
                use_innertube=cfg.use_innertube,
                cache=runtime.cache,
            )
        )
        results.extend(concurrent_results)
    else:
        for i, video in to_process:
            click.echo(f"\n[{i}/{len(videos)}] {video.video_id} {video.title}")
            result = _process_video(
                video,
                run_time,
                dry_run=plan.dry_run,
                capture_format=request.capture_format,
                models=runtime.models,
                filler_words=runtime.filler_words,
                stop_after_capture=plan.stop_after_capture,
                capture_backend=runtime.capture_backend,
                code_bearing=resolved.code_bearing,
                glossary=cfg.glossary,
                media_path=resolved.media_map.get(video.video_id),
                correct_transcript=cfg.transcript_correction,
                known_terms=known_terms,
                use_innertube=cfg.use_innertube,
                cache=runtime.cache,
            )
            results.append(result)

    # Write the proper nouns Stage 01b confirmed into the per-playlist sheet
    # (preserving any user corrections) so they can be reviewed before Stage 05
    # and reused on the next run. Done before the stop-after-capture return so
    # the sheet is available during a manual review pause.
    if proper_noun_sheet_path is not None:
        _update_proper_noun_sheet(proper_noun_sheet_path, results)

    if plan.stop_after_capture:
        click.echo(
            "\n[stop-after-capture] Phase 1 complete. Review 02_Summary.md, "
            "set `reviewed: true`, then re-run with --resume-reviewed."
        )
        return None

    succeeded = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    report_video_summary(len(videos), succeeded, failed)

    if not plan.run_synthesis:
        click.echo("[skip] --skip-synthesis: stage 05 bypassed")
        return None

    if len(succeeded) < request.min_playlist_size:
        click.echo(
            f"[skip] only {len(succeeded)} videos succeeded (< {request.min_playlist_size}), "
            "stage 05 skipped"
        )
        return None
    return succeeded
