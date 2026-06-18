"""Click CLI 定義 (引数・オプション・ヘルプ)。

ここはコマンドラインの「受付」: オプションを定義し、入力を ``CliRequest`` へ
詰め替えて ``command.run`` に渡すだけ。処理の HOW は持たない。
"""

from __future__ import annotations

from pathlib import Path

import click

from .cli_config import _SYNTHESIS_PROFILE_CHOICES
from .command import CliRequest, run
from .stages.synthesis import MIN_PLAYLIST_SIZE


@click.command()
@click.argument("url", required=False)
@click.option("--dry-run", is_flag=True, help="Do not write to vault; print to stdout only.")
@click.option(
    "--local-media",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help=(
        "Build hands-on from a LOCAL folder of the playlist's video files "
        "(fully offline — no YouTube access). Stage 01 transcribes each file "
        "with Whisper (run with --extra mlx / --extra whisper); Stage 03 "
        "captures from it. URL becomes optional. Name files with the 11-char "
        "video_id, e.g. yt-dlp '%(id)s.%(ext)s'. See docs/local-media.md."
    ),
)
@click.option(
    "--concurrency",
    type=click.IntRange(1, 8),
    default=3,
    show_default=True,
    help="Videos in parallel (1-8). Higher is faster but raises API-rate/CPU load.",
)
@click.option(
    "--sub-agents",
    type=click.IntRange(1, 8),
    default=1,
    show_default=True,
    help=(
        "Run stages 01-04 as N parallel sub-agent processes, each owning a "
        "contiguous slice of the playlist (remainder on the last shard); Stage "
        "05 runs once afterwards. Default 1 = original single-process flow. "
        "Opt in with e.g. --sub-agents 3. See docs/sub-agents.md."
    ),
)
@click.option(
    "--video-range",
    default=None,
    hidden=True,
    help="Internal: 0-based half-open 'start:end' slice processed by one sub-agent shard.",
)
@click.option(
    "--run-timestamp",
    default=None,
    hidden=True,
    help="Internal: ISO run_time shared across sub-agent shards so they write one playlist folder.",
)
@click.option(
    "--code-bearing/--no-code-bearing",
    "code_bearing_override",
    default=None,
    hidden=True,
    help="Internal: parent-pinned genre decision passed to sub-agent shards (skips the router).",
)
@click.option(
    "--transcript-concurrency",
    type=click.IntRange(1, 16),
    default=None,
    help=(
        "Fan-out for the upfront caption-transcript cache warm-up (1-16). "
        "Network-bound, so it can run higher than --concurrency. "
        "Default 8 (or config.json transcript_concurrency)."
    ),
)
@click.option(
    "--llm-concurrency",
    type=click.IntRange(1, 32),
    default=None,
    help=(
        "Cap concurrent LLM provider calls, independent of --concurrency. "
        "Lets --concurrency rise without over-subscribing the API rate budget. "
        "Default: unbounded (or config.json llm_concurrency)."
    ),
)
@click.option(
    "--download-concurrency",
    type=click.IntRange(1, 32),
    default=None,
    help=(
        "Cap concurrent video downloads (yt-dlp), independent of --concurrency. "
        "Default: unbounded (or config.json download_concurrency)."
    ),
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Persistent cache root for transcripts/videos/code/LLM output. "
        "Default ~/.cache/pipeline-youtube (or config.json cache_dir / "
        "$PIPELINE_YOUTUBE_CACHE)."
    ),
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Disable the persistent cache entirely (recompute everything).",
)
@click.option(
    "--cache-llm-synthesis",
    is_flag=True,
    help=(
        "Also cache Stage 05 synthesis LLM output (off by default so re-runs "
        "regenerate fresh synthesis). Stage 02/04/router output is cached either way."
    ),
)
@click.option("--skip-synthesis", is_flag=True, help="Skip stage 05 after 01-04 finish.")
@click.option(
    "--synthesis-only",
    is_flag=True,
    help="Skip stages 01-04 and re-run only stage 05 against existing 04 md files for today's date.",
)
@click.option(
    "--folder-name",
    default=None,
    help=(
        "[scaffolding — not yet functional] Explicit 04_Learning_Material/<NAME> "
        "playlist folder to resume from (e.g. 'YYYY-MM-DD-HHmm <playlist>'). "
        "Intended: with --synthesis-only, load that exact folder regardless of "
        "date and re-run 05→evaluation, writing into 05_Synthesis/<NAME>."
    ),
)
@click.option(
    "--eval-loop",
    type=click.IntRange(0, 2),
    default=0,
    show_default=True,
    help=(
        "[scaffolding — not yet functional] Run the evaluation feedback loop "
        "after synthesis, up to N iterations (max 2): 2 fixed-role evaluators "
        "(coverage + pedagogy) auto-route findings to regenerate Stage 04 or "
        "re-run Stage 05, stopping early when no blocking findings remain. "
        "0 = disabled."
    ),
)
@click.option(
    "--force-video",
    multiple=True,
    help="Force reprocess specific video IDs even if checkpoint shows complete. Repeatable.",
)
@click.option(
    "--capture-format",
    type=click.Choice(["auto", "webp", "gif"]),
    default="auto",
    help="Animated capture output format. Default auto picks WebP when possible.",
)
@click.option(
    "--model",
    default="sonnet",
    help="Claude model alias for stages 02/04/05 (sonnet, haiku, opus, or full ID).",
)
@click.option(
    "--min-playlist-size",
    type=click.IntRange(1, 100),
    default=MIN_PLAYLIST_SIZE,
    show_default=True,
    help="Skip stage 05 when fewer than N videos succeed (default 3).",
)
@click.option(
    "--max-chapters",
    type=click.IntRange(1, 30),
    default=None,
    help="Cap β's chapter count via prompt constraint. Unset = let β decide.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Override config.json path.",
)
@click.option(
    "--stop-after-capture",
    is_flag=True,
    help=(
        "Phase 1: run stages 01-03 and stop. User reviews Stage 02 in Obsidian, "
        "flips frontmatter `reviewed: true`, then runs again with --resume-reviewed."
    ),
)
@click.option(
    "--resume-reviewed",
    is_flag=True,
    help=(
        "Phase 3: skip stages 01-03; only process videos whose 02_Summary.md "
        "frontmatter has `reviewed: true`. Runs stages 04 and 05 on the filtered set."
    ),
)
@click.option(
    "--capture-backend",
    type=click.Choice(["host", "docker"]),
    default=None,
    help=(
        "Stage 03 execution backend. 'host' (default) runs yt-dlp/ffmpeg directly. "
        "'docker' runs them inside the hardened pipeline-youtube-capture image — "
        "build it first via `docker build -f docker/Dockerfile.capture -t "
        "pipeline-youtube-capture:latest .`. Overrides config.json."
    ),
)
@click.option(
    "--synthesis-timeout",
    type=click.IntRange(60),
    default=None,
    help=(
        "Per-agent timeout for Stage 05 in seconds. "
        "Default: auto (300 + 60 × video_count). Overrides config.json."
    ),
)
@click.option(
    "--synthesis-profile",
    type=click.Choice(list(_SYNTHESIS_PROFILE_CHOICES)),
    default=None,
    help=(
        "Agent Teams composition for Stage 05. 'auto' (default) picks "
        "'standard' (≤15 videos), 'parallel' (16-30), or 'parallel+full' "
        "(>30). 'full' adds a Reviewer pass. Overrides config.json."
    ),
)
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "ollama", "lmstudio"]),
    default=None,
    help=(
        "Runtime backend override: route EVERY stage to this provider's "
        "default model for this run, ignoring config.json's per-stage models. "
        "Omit to use config.json as-is (config is the source of truth). "
        "Forwarded automatically to --sub-agents workers."
    ),
)
@click.option(
    "--hybrid",
    is_flag=True,
    help=(
        "Keep the heavy stages (stage_04, leader) on Anthropic even when an "
        "open/local provider is selected. Requires the 'anthropic' provider "
        "in config.json. Pairs well with --provider ollama."
    ),
)
def cli(
    url: str | None,
    dry_run: bool,
    concurrency: int,
    sub_agents: int,
    video_range: str | None,
    run_timestamp: str | None,
    code_bearing_override: bool | None,
    transcript_concurrency: int | None,
    llm_concurrency: int | None,
    download_concurrency: int | None,
    cache_dir: Path | None,
    no_cache: bool,
    cache_llm_synthesis: bool,
    skip_synthesis: bool,
    synthesis_only: bool,
    folder_name: str | None,
    eval_loop: int,
    force_video: tuple[str, ...],
    capture_format: str,
    model: str,
    min_playlist_size: int,
    max_chapters: int | None,
    config_path: Path | None,
    stop_after_capture: bool,
    resume_reviewed: bool,
    capture_backend: str | None,
    synthesis_timeout: int | None,
    synthesis_profile: str | None,
    provider: str | None,
    hybrid: bool,
    local_media: Path | None,
) -> None:
    """Process a YouTube playlist or single-video URL end-to-end."""
    run(
        CliRequest(
            url=url,
            dry_run=dry_run,
            concurrency=concurrency,
            sub_agents=sub_agents,
            video_range=video_range,
            run_timestamp=run_timestamp,
            code_bearing_override=code_bearing_override,
            transcript_concurrency=transcript_concurrency,
            llm_concurrency=llm_concurrency,
            download_concurrency=download_concurrency,
            cache_dir=cache_dir,
            no_cache=no_cache,
            cache_llm_synthesis=cache_llm_synthesis,
            skip_synthesis=skip_synthesis,
            synthesis_only=synthesis_only,
            folder_name=folder_name,
            eval_loop=eval_loop,
            force_video=force_video,
            capture_format=capture_format,
            model=model,
            min_playlist_size=min_playlist_size,
            max_chapters=max_chapters,
            config_path=config_path,
            stop_after_capture=stop_after_capture,
            resume_reviewed=resume_reviewed,
            capture_backend=capture_backend,
            synthesis_timeout=synthesis_timeout,
            synthesis_profile=synthesis_profile,
            provider=provider,
            hybrid=hybrid,
            local_media=local_media,
        )
    )


if __name__ == "__main__":
    cli()
