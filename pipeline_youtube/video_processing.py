"""Per-video stage 01-04 driver and its bounded-concurrency runner.

Extracted from `main.py`. `_process_video` runs one video through stages
01 (scripts) → 02 (summary) → 03 (capture) → 04 (learning), returning a
`VideoRunResult`; `_run_videos_concurrent` fans out N of them under an
asyncio semaphore.
"""

from __future__ import annotations

import asyncio
import contextlib
import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from .glossary import Glossary
from .pipeline import compute_note_paths, create_placeholder_notes
from .playlist import VideoMeta
from .run_result import VideoRunResult, _strip_frontmatter
from .sanitize import sanitize_untrusted_text
from .stages.capture import (
    DEFAULT_RESOLUTION,
    prefetch_video_download,
    run_stage_capture,
)
from .stages.learning import run_stage_learning
from .stages.scripts import run_stage_scripts
from .stages.summary import run_stage_summary
from .stats import record_transcript_stat

if TYPE_CHECKING:
    from .services.cache import Cache


def _process_video(
    video: VideoMeta,
    run_time: datetime,
    *,
    dry_run: bool,
    capture_format: str,
    models: dict[str, str],
    filler_words: tuple[str, ...] = (),
    stop_after_capture: bool = False,
    capture_backend: Any = None,
    code_bearing: bool = False,
    glossary: Glossary | None = None,
    media_path: Path | None = None,
    correct_transcript: bool = False,
    known_terms: list[tuple[str, str]] | None = None,
    use_innertube: bool = True,
    cache: Cache | None = None,
) -> VideoRunResult:
    try:
        paths = compute_note_paths(video, run_time)
        create_placeholder_notes(video, run_time, dry_run=dry_run)

        correct_model = models["stage_01_correct"] if correct_transcript else None
        if correct_model:
            click.echo(f"  [01] scripts (correct={correct_model})...", nl=False)
        else:
            click.echo("  [01] scripts...", nl=False)
        transcript = run_stage_scripts(
            video,
            paths["scripts"],
            dry_run=dry_run,
            include_code_blocks=code_bearing,
            media_path=media_path,
            correct_model=correct_model,
            known_terms=known_terms,
            use_innertube=use_innertube,
        )
        with contextlib.suppress(Exception):
            record_transcript_stat(video, transcript)
        # Surface the per-tier fallback reason inline so a failed transcript
        # (source=error) shows *why* — e.g. "auto:ip_blocked; whisper:
        # whisper_not_installed" — without digging into transcript_stats.jsonl.
        # fallback_reason embeds external tool output (yt-dlp errors carry ANSI
        # escapes), so sanitize/cap it before echoing — the same untrusted
        # treatment the stats sink applies — to avoid terminal-escape injection.
        reason = sanitize_untrusted_text(
            transcript.fallback_reason, 500, context="transcript.fallback_reason"
        )
        # Stage 01b correction is the only paid work Stage 01 does; surface its
        # cost inline like Stage 02/04 (None ⇒ correction disabled, omit it).
        cost_suffix = (
            f" cost=${transcript.correction_cost_usd:.3f}"
            if transcript.correction_cost_usd is not None
            else ""
        )
        click.echo(
            f" source={transcript.source.value}"
            f" snippets={len(transcript.snippets)}"
            f" lang={transcript.language or '-'}"
            + cost_suffix
            + (f" reason=({reason})" if reason else "")
        )

        if not transcript.snippets:
            return VideoRunResult(video=video, error="no_transcript_snippets")

        # Kick off Stage 03 video download in parallel with Stage 02 LLM call.
        # Stage 03 still waits for Stage 02's output to parse timeline ranges,
        # but the download — the bulk of Stage 03's wall time — can overlap.
        #
        # Skip the prefetch when the video is already in the persistent cache:
        # the prefetch always downloads, so an unconditional prefetch would
        # re-fetch the mp4 every rerun and overwrite the cache, defeating it.
        # On a cache hit `run_stage_capture` reuses the cached copy via its own
        # `cache.get_video` lookup (prefetched_path stays None).
        # In --local-media mode the file is already on disk, so skip the
        # network prefetch entirely (Stage 03 uses media_path directly below).
        prefetch = None
        if media_path is None and not dry_run:
            if cache is None:
                from .cache import get_cache

                cache = get_cache()
            if cache.get_video(video.video_id, DEFAULT_RESOLUTION) is None:
                with contextlib.suppress(Exception):
                    prefetch = prefetch_video_download(video, backend=capture_backend)

        click.echo(f"  [02] summary (model={models['stage_02']})...", nl=False)
        summary_resp = run_stage_summary(
            video,
            paths["summary"],
            transcript,
            model=models["stage_02"],
            filler_words=filler_words,
            glossary=glossary,
            dry_run=dry_run,
        )
        click.echo(
            f" in={summary_resp.input_tokens or 0}"
            f" out={summary_resp.output_tokens or 0}"
            f" cost=${summary_resp.total_cost_usd or 0:.3f}"
        )

        # Local --local-media file is the capture source; else the prefetch.
        prefetched_path = media_path
        if prefetch is not None:
            # The prefetch thread owns tmp/<video_id>.mp4. Block until it has
            # finished (success or failure) before Stage 03 runs. A fixed
            # timeout here is unsafe: when --download-concurrency throttles the
            # prefetch, it can sit queued on the download semaphore past the
            # timeout while still alive, and run_stage_capture would then fall
            # back to a second _download_video() on the same path — the two
            # downloads race on unlink/overwrite. Waiting to completion makes
            # the prefetch the single writer; on failure the thread has exited,
            # so the capture fallback re-downloads safely. The wait is finite:
            # the download is bounded by the backend subprocess timeout and the
            # download-concurrency semaphore.
            err = prefetch.wait(timeout=None)
            if err is None and prefetch.path.exists():
                prefetched_path = prefetch.path

        click.echo("  [03] capture...", nl=False)
        capture_result = run_stage_capture(
            video,
            paths["summary"],
            paths["capture"],
            capture_format=capture_format,  # type: ignore[arg-type]
            dry_run=dry_run,
            prefetched_video_path=prefetched_path,
            backend=capture_backend,
            # Local media is offline-only: if the file is missing, fail closed
            # rather than silently downloading from YouTube.
            allow_download=media_path is None,
            # Never delete the user's --local-media source file.
            delete_video=media_path is None,
            # Forward the injected cache (None ⇒ run_stage_capture falls back to
            # the process-global get_cache(), preserving prior behavior).
            cache=cache,
        )
        if capture_result.error and not capture_result.outcomes:
            click.echo(f" FAILED: {capture_result.error}")
        else:
            click.echo(
                f" {capture_result.success_count}/{len(capture_result.ranges)} ranges"
                f" fmt={capture_result.capture_format}"
            )

        if stop_after_capture:
            click.echo(
                "  [stop-after-capture] review 02_Summary.md then re-run with --resume-reviewed"
            )
            return VideoRunResult(
                video=video,
                learning_md_body=None,
                transcript_cost_usd=transcript.correction_cost_usd,
                transcript_model=correct_model,
                summary_cost_usd=summary_resp.total_cost_usd,
                summary_model=summary_resp.model,
                confirmed_terms=transcript.confirmed_terms,
            )

        click.echo(f"  [04] learning (model={models['stage_04']})...", nl=False)
        learning_resp = run_stage_learning(
            video,
            paths["summary"],
            paths["capture"],
            paths["learning"],
            run_time=run_time,
            model=models["stage_04"],
            dry_run=dry_run,
            code_bearing=code_bearing,
        )
        click.echo(
            f" in={learning_resp.input_tokens or 0}"
            f" out={learning_resp.output_tokens or 0}"
            f" cost=${learning_resp.total_cost_usd or 0:.3f}"
        )

        if dry_run:
            body = learning_resp.text.strip()
        else:
            body = _strip_frontmatter(paths["learning"].read_text(encoding="utf-8"))

        return VideoRunResult(
            video=video,
            learning_md_path=paths["learning"],
            learning_md_body=body,
            transcript_cost_usd=transcript.correction_cost_usd,
            transcript_model=correct_model,
            summary_cost_usd=summary_resp.total_cost_usd,
            summary_model=summary_resp.model,
            learning_cost_usd=learning_resp.total_cost_usd,
            learning_model=learning_resp.model,
            confirmed_terms=transcript.confirmed_terms,
        )
    except Exception as e:
        traceback.print_exc()
        return VideoRunResult(video=video, error=f"{type(e).__name__}: {e}")


async def _run_videos_concurrent(
    videos: list[VideoMeta],
    run_time: datetime,
    *,
    concurrency: int,
    dry_run: bool,
    capture_format: str,
    models: dict[str, str],
    filler_words: tuple[str, ...] = (),
    stop_after_capture: bool = False,
    capture_backend: Any = None,
    code_bearing: bool = False,
    glossary: Glossary | None = None,
    media_map: dict[str, Path] | None = None,
    correct_transcript: bool = False,
    known_terms: list[tuple[str, str]] | None = None,
    use_innertube: bool = True,
    cache: Cache | None = None,
) -> list[VideoRunResult]:
    """Process multiple videos concurrently with bounded parallelism."""
    sem = asyncio.Semaphore(concurrency)
    media = media_map or {}

    async def _task(i: int, video: VideoMeta) -> VideoRunResult:
        async with sem:
            click.echo(f"\n[{i}/{len(videos)}] {video.video_id} {video.title}")
            return await asyncio.to_thread(
                _process_video,
                video,
                run_time,
                dry_run=dry_run,
                capture_format=capture_format,
                models=models,
                filler_words=filler_words,
                stop_after_capture=stop_after_capture,
                capture_backend=capture_backend,
                code_bearing=code_bearing,
                glossary=glossary,
                media_path=media.get(video.video_id),
                correct_transcript=correct_transcript,
                known_terms=known_terms,
                use_innertube=use_innertube,
                cache=cache,
            )

    tasks = [_task(i, v) for i, v in enumerate(videos, 1)]
    return list(await asyncio.gather(*tasks))
