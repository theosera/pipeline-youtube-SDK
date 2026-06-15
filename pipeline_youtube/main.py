"""CLI entry point for pipeline-youtube-sdk.

Orchestrates the full pipeline: fetch playlist metadata → process each
video through stages 01-04 → stage 05 synthesis if ≥3 succeed.

SDK version: uses multi-provider LLM backends (Ollama, LM Studio,
OpenAI, Anthropic, Gemini) via direct API calls instead of the
`claude -p` CLI subprocess.

Concurrency model: `--concurrency N` runs up to N videos in parallel
via `asyncio.to_thread` + `asyncio.Semaphore` (default 3). Whisper
(tier 3 fallback) has its own bounded semaphore (default 1) so it never
exceeds its GPU/RAM budget even under higher concurrency.

A content-addressed persistent cache (see `cache.py`) stores transcripts,
downloaded videos, fetched code, and Stage 02/04/router LLM output so
re-runs and `--synthesis-only` are near-instant. `--no-cache` disables it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from .cache import DEFAULT_MAX_VIDEO_BYTES
from .checkpoint import (
    extract_trusted_video_id,
    get_completed_video_ids,
    read_trusted_video_id,
)
from .config import VaultRootError, set_dry_run, set_vault_root
from .genres import CODE_BEARING_GENRES, classify_playlist_genre
from .glossary import (
    Glossary,
    GlossaryConflictError,
    GlossaryParseError,
    Normalizer,
    load_glossary,
)
from .obsidian import format_playlist_folder_name
from .parallel import orchestrate_sub_agents, parse_video_range, strip_cli_option
from .path_safety import ensure_safe_path
from .pipeline import LEARNING_BASE, UNIT_DIRS, compute_note_paths, create_placeholder_notes
from .playlist import VideoMeta, fetch_metadata, validate_youtube_url
from .providers.registry import (
    configure_llm_cache,
    configure_llm_concurrency,
    configure_providers,
    resolve_role,
)
from .providers.selection import apply_selection
from .sanitize import configure_alert_sink, sanitize_untrusted_text
from .stages.capture import (
    ASSETS_REL_PATH,
    DEFAULT_RESOLUTION,
    prefetch_video_download,
    run_stage_capture,
    sweep_stale_tmp,
)
from .stages.capture_backend import DockerBackendNotReady, DockerCaptureBackend
from .stages.learning import run_stage_learning
from .stages.scripts import (
    DEFAULT_TRANSCRIPT_CONCURRENCY,
    run_stage_scripts,
    warm_transcript_cache,
)
from .stages.summary import run_stage_summary
from .stages.synthesis import MIN_PLAYLIST_SIZE, log_synthesis_preflight, run_stage_synthesis
from .stats import record_transcript_stat
from .synthesis.agents import compute_synthesis_timeouts
from .transcript.whisper_fallback import configure_whisper, describe_whisper

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

_MODEL_KEYS = frozenset(
    {
        "router",
        "stage_01_correct",
        "stage_02",
        "stage_04",
        "alpha",
        "beta",
        "leader",
        "reviewer",
        "eval_coverage",
        "eval_pedagogy",
    }
)
# "gamma" accepted silently for backward-compat with existing config.json,
# but the γ LLM role has been replaced by a Python set diff — the value is ignored.
_DEPRECATED_MODEL_KEYS = frozenset({"gamma"})

_SYNTHESIS_PROFILE_CHOICES = ("auto", "standard", "parallel", "full", "parallel+full")


_CAPTURE_BACKENDS = frozenset({"host", "docker"})


@dataclass(frozen=True)
class CliConfig:
    vault_root: Path
    models: dict[str, str]
    filler_words: tuple[str, ...]
    # Stage 03 execution backend. "host" runs yt-dlp/ffmpeg directly;
    # "docker" isolates them in the hardened image built from
    # docker/Dockerfile.capture. See docs/docker.md.
    capture_backend: str = "host"
    capture_docker_image: str = "pipeline-youtube-capture:latest"
    synthesis_timeout: int | None = None
    synthesis_profile: str | None = None
    # Persistent cache (see cache.py). cache_dir=None → default ~/.cache root.
    cache_dir: Path | None = None
    cache_max_video_bytes: int = DEFAULT_MAX_VIDEO_BYTES
    # Max concurrent Whisper transcriptions (GPU/RAM bound). None → keep default.
    whisper_concurrency: int | None = None
    # Skip Whisper for audio longer than this (seconds). None → keep default
    # (DEFAULT_WHISPER_MAX_AUDIO_SECONDS). Guards against multi-hour CPU runs.
    whisper_max_audio_seconds: int | None = None
    # Fan-out for the upfront transcript cache warm-up (network-bound).
    # None → use scripts.DEFAULT_TRANSCRIPT_CONCURRENCY.
    transcript_concurrency: int | None = None
    # Resource-class caps (Phase 3 A), independent of --concurrency.
    # None → unbounded (prior behavior).
    llm_concurrency: int | None = None
    download_concurrency: int | None = None
    # Proper-noun normalization glossary (Stage 02). None → no normalization
    # (prior behavior); set via config.json "glossary_path".
    glossary: Glossary | None = None
    # Local-transcription backend/model (Stage 01 Whisper tier). backend:
    # "auto" (MLX on Apple Silicon, else openai), "mlx", or "openai".
    # model: None → backend default. Set via config.json
    # "whisper_backend"/"whisper_model".
    whisper_backend: str = "auto"
    whisper_model: str | None = None
    # Stage 01b: when True, run the chunked transcript through an LLM +
    # web-search correction pass (role stage_01_correct, default opus on
    # Anthropic) before rendering. Opt-in because it is a paid, slower call.
    transcript_correction: bool = False


def _load_config(config_path: Path, fallback_model: str) -> CliConfig:
    """Load config.json. Unknown keys are ignored; `models` is optional.

    Any missing model key falls back to `fallback_model` (CLI --model).
    Unrecognized model keys raise UsageError so typos are caught early.
    """
    if not config_path.exists():
        raise click.UsageError(
            f"config.json not found at {config_path}. "
            "Copy config.example.json to config.json and set vault_root."
        )
    data = json.loads(config_path.read_text(encoding="utf-8"))
    vault_root = data.get("vault_root")
    if not vault_root or vault_root == "/path/to/your/Obsidian Vault":
        raise click.UsageError("config.json vault_root is not configured.")
    path = Path(vault_root).expanduser()
    if not path.exists():
        raise click.UsageError(f"vault_root does not exist: {path}")

    models_raw = data.get("models") or {}
    if not isinstance(models_raw, dict):
        raise click.UsageError("config.json: 'models' must be an object")
    unknown = set(models_raw) - _MODEL_KEYS - _DEPRECATED_MODEL_KEYS
    if unknown:
        raise click.UsageError(
            f"config.json: unknown model keys {sorted(unknown)!r}; "
            f"expected any of {sorted(_MODEL_KEYS)!r}"
        )
    # Router defaults to haiku regardless of fallback_model — it's a single
    # cheap classification call where speed/cost beats reasoning depth.
    # Stage 01b transcript correction defaults to opus (fact-checks proper
    # nouns via web search; reasoning depth matters most).
    _per_key_default = {"router": "haiku", "stage_01_correct": "opus"}
    models = {
        key: models_raw.get(key, _per_key_default.get(key, fallback_model)) for key in _MODEL_KEYS
    }

    filler_raw = data.get("filler_words")
    if filler_raw is None:
        from .transcript.chunking import DEFAULT_FILLER_WORDS

        filler = DEFAULT_FILLER_WORDS
    else:
        if not isinstance(filler_raw, list) or not all(isinstance(x, str) for x in filler_raw):
            raise click.UsageError("config.json: 'filler_words' must be a list of strings")
        filler = tuple(filler_raw)

    capture_backend = str(data.get("capture_backend") or "host").lower()
    if capture_backend not in _CAPTURE_BACKENDS:
        raise click.UsageError(
            f"config.json: capture_backend must be one of {sorted(_CAPTURE_BACKENDS)!r}, "
            f"got {capture_backend!r}"
        )
    capture_docker_image = str(
        data.get("capture_docker_image") or "pipeline-youtube-capture:latest"
    )

    synthesis_timeout_raw = data.get("synthesis_timeout")
    if synthesis_timeout_raw is None or synthesis_timeout_raw == "auto":
        synthesis_timeout: int | None = None
    elif isinstance(synthesis_timeout_raw, int) and synthesis_timeout_raw > 0:
        synthesis_timeout = synthesis_timeout_raw
    else:
        raise click.UsageError(
            f'config.json: synthesis_timeout must be a positive integer or "auto", '
            f"got {synthesis_timeout_raw!r}"
        )

    synthesis_profile_raw = data.get("synthesis_profile")
    if synthesis_profile_raw is None:
        synthesis_profile: str | None = None
    elif (
        isinstance(synthesis_profile_raw, str)
        and synthesis_profile_raw in _SYNTHESIS_PROFILE_CHOICES
    ):
        synthesis_profile = synthesis_profile_raw
    else:
        raise click.UsageError(
            f"config.json: synthesis_profile must be one of "
            f"{list(_SYNTHESIS_PROFILE_CHOICES)!r}, got {synthesis_profile_raw!r}"
        )

    cache_dir_raw = data.get("cache_dir")
    cache_dir = Path(str(cache_dir_raw)).expanduser() if cache_dir_raw else None

    max_video_raw = data.get("cache_max_video_bytes")
    if max_video_raw is None:
        cache_max_video_bytes = DEFAULT_MAX_VIDEO_BYTES
    elif isinstance(max_video_raw, int) and max_video_raw > 0:
        cache_max_video_bytes = max_video_raw
    else:
        raise click.UsageError("config.json: cache_max_video_bytes must be a positive integer")

    whisper_conc_raw = data.get("whisper_concurrency")
    if whisper_conc_raw is None:
        whisper_concurrency: int | None = None
    elif isinstance(whisper_conc_raw, int) and whisper_conc_raw > 0:
        whisper_concurrency = whisper_conc_raw
    else:
        raise click.UsageError("config.json: whisper_concurrency must be a positive integer")

    transcript_conc_raw = data.get("transcript_concurrency")
    if transcript_conc_raw is None:
        transcript_concurrency: int | None = None
    elif isinstance(transcript_conc_raw, int) and transcript_conc_raw > 0:
        transcript_concurrency = transcript_conc_raw
    else:
        raise click.UsageError("config.json: transcript_concurrency must be a positive integer")

    def _positive_int_or_none(key: str) -> int | None:
        raw = data.get(key)
        if raw is None:
            return None
        if isinstance(raw, int) and raw > 0:
            return raw
        raise click.UsageError(f"config.json: {key} must be a positive integer")

    llm_concurrency = _positive_int_or_none("llm_concurrency")
    download_concurrency = _positive_int_or_none("download_concurrency")
    whisper_max_audio_seconds = _positive_int_or_none("whisper_max_audio_seconds")

    glossary = _load_glossary_from_config(data, config_path)

    whisper_backend = str(data.get("whisper_backend") or "auto").lower()
    if whisper_backend not in {"auto", "mlx", "openai"}:
        raise click.UsageError(
            "config.json: whisper_backend must be one of ['auto', 'mlx', 'openai'], "
            f"got {whisper_backend!r}"
        )
    whisper_model_raw = data.get("whisper_model")
    if whisper_model_raw is None or whisper_model_raw == "":
        whisper_model = None
    elif isinstance(whisper_model_raw, str):
        whisper_model = whisper_model_raw
    else:
        raise click.UsageError("config.json: whisper_model must be a string or null")

    transcript_correction_raw = data.get("transcript_correction", False)
    if not isinstance(transcript_correction_raw, bool):
        raise click.UsageError("config.json: transcript_correction must be a boolean")

    return CliConfig(
        vault_root=path,
        models=models,
        filler_words=filler,
        capture_backend=capture_backend,
        capture_docker_image=capture_docker_image,
        synthesis_timeout=synthesis_timeout,
        synthesis_profile=synthesis_profile,
        cache_dir=cache_dir,
        cache_max_video_bytes=cache_max_video_bytes,
        whisper_concurrency=whisper_concurrency,
        whisper_max_audio_seconds=whisper_max_audio_seconds,
        transcript_concurrency=transcript_concurrency,
        llm_concurrency=llm_concurrency,
        download_concurrency=download_concurrency,
        glossary=glossary,
        whisper_backend=whisper_backend,
        whisper_model=whisper_model,
        transcript_correction=transcript_correction_raw,
    )


def _load_glossary_from_config(data: dict[str, Any], config_path: Path) -> Glossary | None:
    """Load the optional proper-noun glossary referenced by ``glossary_path``.

    ``glossary_path`` is optional (absent → ``None`` → Stage 02 normalization
    disabled). A relative path resolves against config.json's directory so the
    glossary travels with the config. A malformed/missing file is a
    configuration error surfaced as ``UsageError`` (fail fast, not silently
    skipped).
    """
    raw = data.get("glossary_path")
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise click.UsageError("config.json: glossary_path must be a non-empty string")
    glossary_path = Path(raw).expanduser()
    if not glossary_path.is_absolute():
        glossary_path = (config_path.parent / glossary_path).resolve()
    try:
        glossary = load_glossary(glossary_path)
    except (GlossaryParseError, OSError) as exc:
        raise click.UsageError(f"config.json: glossary_path could not be loaded: {exc}") from exc
    # Fail fast on variant conflicts now (build the index once at startup)
    # rather than deep inside per-video Stage 02, after transcript + LLM work.
    try:
        Normalizer(glossary)
    except GlossaryConflictError as exc:
        raise click.UsageError(f"config.json: glossary_path has a variant conflict: {exc}") from exc
    return glossary


@dataclass
class VideoRunResult:
    video: VideoMeta
    learning_md_path: Path | None = None
    learning_md_body: str | None = None
    error: str | None = None
    # Per-stage cost tracking (populated by `_process_video`).
    transcript_cost_usd: float | None = None
    transcript_model: str | None = None
    summary_cost_usd: float | None = None
    summary_model: str | None = None
    learning_cost_usd: float | None = None
    learning_model: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.learning_md_body is not None


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return text.strip()
    end = text.find("\n---", 3)
    if end == -1:
        return text.strip()
    return text[end + 4 :].lstrip()


def _parse_run_timestamp(run_timestamp: str | None) -> datetime:
    """Resolve the shared run_time, surfacing a bad --run-timestamp as a clean CLI error."""
    if not run_timestamp:
        return datetime.now()
    try:
        return datetime.fromisoformat(run_timestamp)
    except ValueError as exc:
        raise click.UsageError(f"invalid --run-timestamp: {run_timestamp!r}") from exc


def _load_existing_04_body(video_id: str, playlist_title: str, run_date: datetime) -> str | None:
    """Read the stage 04 body for a checkpoint-skipped video.

    Returns the frontmatter-stripped body, or None if the file can't be found.
    Uses the same M3 hardened frontmatter validation as `is_video_complete`.
    """
    from .checkpoint import _find_learning_folder

    folder = _find_learning_folder(playlist_title, run_date)
    if folder is None:
        return None
    for md in folder.glob("*.md"):
        if read_trusted_video_id(md) != video_id:
            continue
        try:
            text = md.read_text(encoding="utf-8")
        except OSError:
            continue
        return _strip_frontmatter(text)
    return None


def _find_summary_md(video_id: str, playlist_title: str, run_date: datetime) -> Path | None:
    """Locate the existing 02_Summary.md for `video_id` within a given run date.

    Used by Phase 3 (`--resume-reviewed`) to look up summaries written
    in a prior Phase 1 run. Falls back across date-prefix matches so
    users can resume on a different clock day.
    """
    from .config import get_vault_root

    vault_root = get_vault_root()
    rel = f"{LEARNING_BASE}/{UNIT_DIRS['summary']}"
    safe_rel = ensure_safe_path(rel)
    base = vault_root / safe_rel
    if not base.exists():
        return None

    # Try today's canonical playlist folder, then any folder under base.
    for candidate_folder in _summary_folder_candidates(base, playlist_title, run_date):
        if not candidate_folder.exists():
            continue
        for md in candidate_folder.glob("*.md"):
            if read_trusted_video_id(md) == video_id:
                return md
    return None


def _print_cost_breakdown(
    video_results: list[VideoRunResult],
    synthesis_result: Any = None,
) -> None:
    """Print a per-stage / per-agent cost table summing across all videos."""
    stage_totals: dict[str, tuple[str, float]] = {}

    def _add(label: str, model: str | None, cost: float | None) -> None:
        if cost is None:
            return
        existing = stage_totals.get(label)
        prev_model = existing[0] if existing else (model or "?")
        prev_cost = existing[1] if existing else 0.0
        stage_totals[label] = (prev_model or model or "?", prev_cost + cost)

    for r in video_results:
        _add("stage_01", r.transcript_model, r.transcript_cost_usd)
        _add("stage_02", r.summary_model, r.summary_cost_usd)
        _add("stage_04", r.learning_model, r.learning_cost_usd)

    if synthesis_result is not None and getattr(synthesis_result, "agent_results", None):
        # With profile-aware orchestration, the agent_results sequence
        # varies (parallel α spawns multiple, reviewer adds one, etc.).
        # Aggregate by the prompt's system-prompt role rather than by
        # positional role labels.
        for agent_res in synthesis_result.agent_results:
            _add("synthesis", agent_res.response.model, agent_res.total_cost_usd)

    if not stage_totals:
        return

    click.echo("\n=== Cost breakdown ===")
    total = 0.0
    for label, (model, cost) in stage_totals.items():
        click.echo(f"  {label:<9} ({model:<7}) ${cost:>7.3f}")
        total += cost
    click.echo(f"  {'total':<9} {'':<9} ${total:>7.3f}")


def _filter_to_reviewed(
    to_process: list[tuple[int, VideoMeta]],
    playlist_title: str,
    run_time: datetime,
) -> list[tuple[int, VideoMeta]]:
    """Keep only videos whose 02_Summary.md frontmatter has `reviewed: true`."""
    from .obsidian import read_frontmatter_field

    kept: list[tuple[int, VideoMeta]] = []
    for i, video in to_process:
        summary_md = _find_summary_md(video.video_id, playlist_title, run_time)
        if summary_md is None:
            click.echo(f"  [skip] {video.video_id}: no 02_Summary.md found")
            continue
        value = read_frontmatter_field(summary_md, "reviewed")
        if value and value.lower() == "true":
            kept.append((i, video))
        else:
            click.echo(f"  [skip] {video.video_id}: reviewed={value!r}")
    return kept


def _summary_folder_candidates(
    base: Path, playlist_title: str, run_date: datetime
) -> Iterator[Path]:
    """Yield likely playlist folders holding 02_Summary files.

    Canonical first, then any date-prefixed folder that contains the
    sanitized title substring (mirrors `_find_learning_folder` heuristics).
    """
    from .obsidian import _strip_playlist_category_prefix, sanitize_title_for_filename

    canonical_name = format_playlist_folder_name(run_date, playlist_title)
    yield base / canonical_name

    date_prefix = run_date.strftime("%Y-%m-%d")
    display_title = _strip_playlist_category_prefix(playlist_title)
    title_needle = sanitize_title_for_filename(display_title)
    if not title_needle:
        return
    try:
        for child in base.iterdir():
            if (
                child.is_dir()
                and child.name.startswith(date_prefix)
                and title_needle in child.name
                and child.name != canonical_name
            ):
                yield child
    except OSError:
        return


def _videos_from_learning_folder(folder_name: str) -> list[VideoMeta]:
    """Reconstruct the video list from an explicit 04_Learning_Material folder.

    Used by URL-free resume (``--synthesis-only --folder-name <NAME>``).
    Reads each ``*.md``'s trusted frontmatter (video_id + title + url) so no
    playlist URL is required. Resolves ``base_dir / <validated folder_name>``
    exactly (no date derivation), enabling resume of a past-date folder.

    TODO(scaffold): validate ``folder_name`` (no path traversal — reuse
    ``ensure_safe_path``), iterate ``*.md``, build ``VideoMeta`` from
    frontmatter (``extract_trusted_video_id`` for the id), set
    ``playlist_title`` to the folder name as a fallback.
    """
    raise NotImplementedError("scaffold: URL-free resume video reconstruction TODO")


def _collect_existing_learning_bodies(
    videos: list[VideoMeta],
    playlist_title: str,
    run_time: datetime,
) -> tuple[list[VideoMeta], list[str], str]:
    """Scan the existing 04_Learning_Material folder for the given playlist date
    and return `(videos, bodies, folder_name)` aligned by input video_id order.

    Also returns the resolved folder name so stage 05 can reuse the exact
    legacy name instead of creating a new one next to it.
    """
    from .config import get_vault_root

    rel_base = f"{LEARNING_BASE}/{UNIT_DIRS['learning']}"
    safe_rel_base = ensure_safe_path(rel_base)
    base_dir = get_vault_root() / safe_rel_base

    preferred = format_playlist_folder_name(run_time, playlist_title)
    learning_dir = base_dir / preferred
    folder_name = preferred

    if not learning_dir.exists() and base_dir.exists():
        # Fallback: match any sibling folder that begins with today's YYYY-MM-DD
        # and contains the sanitized playlist title as a substring. Handles
        # both the new YYYY-MM-DD HHmm <title> format and the legacy
        # YYYY-MM-DD <title> format from runs before the HHmm fix.
        from .obsidian import sanitize_title_for_filename

        date_prefix = run_time.strftime("%Y-%m-%d")
        title_needle = sanitize_title_for_filename(playlist_title)
        candidates = [
            p
            for p in base_dir.iterdir()
            if p.is_dir() and p.name.startswith(date_prefix) and title_needle in p.name
        ]
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            learning_dir = candidates[0]
            folder_name = learning_dir.name
            click.echo(f"(fallback: using legacy folder {folder_name!r})")

    if not learning_dir.exists():
        raise click.UsageError(
            f"04 folder not found: {learning_dir}. "
            "--synthesis-only requires stage 04 files from a prior run on the same date."
        )

    by_video_id: dict[str, str] = {}
    for md in sorted(learning_dir.glob("*.md")):
        try:
            data = md.read_bytes()
        except OSError:
            continue
        vid = extract_trusted_video_id(data)
        if vid is None:
            continue
        text = data.decode("utf-8", errors="replace")
        by_video_id[vid] = _strip_frontmatter(text)

    matched_videos: list[VideoMeta] = []
    matched_bodies: list[str] = []
    for v in videos:
        body = by_video_id.get(v.video_id)
        if body:
            matched_videos.append(v)
            matched_bodies.append(body)
    return matched_videos, matched_bodies, folder_name


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
            from .cache import get_cache

            if get_cache().get_video(video.video_id, DEFAULT_RESOLUTION) is None:
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
            )

    tasks = [_task(i, v) for i, v in enumerate(videos, 1)]
    return list(await asyncio.gather(*tasks))


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
    # The evaluation phase and the explicit-folder resume flow are scaffolded
    # (schemas, fixed-role agents, package, and these flags exist) but their
    # execution paths are not implemented yet. Reject them up front so the CLI
    # never crashes mid-run with NotImplementedError; drop a gate when wiring
    # its real implementation.
    if eval_loop > 0:
        raise click.UsageError("--eval-loop is not implemented yet (scaffolding in progress).")
    if folder_name:
        raise click.UsageError(
            "--folder-name resume is not implemented yet (scaffolding in progress)."
        )

    if not url and not local_media:
        click.echo("Usage: pipeline-youtube <playlist-or-video-url> [options]")
        click.echo("   or: pipeline-youtube --local-media <dir>   (fully offline)")
        sys.exit(2)

    if url:
        try:
            validate_youtube_url(url)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc

    # Mutually-exclusive phase flags
    phase_flags = sum(bool(x) for x in (stop_after_capture, resume_reviewed, synthesis_only))
    if phase_flags > 1:
        raise click.UsageError(
            "--stop-after-capture, --resume-reviewed, and --synthesis-only are mutually exclusive."
        )

    # Sub-agent orchestration owns the full 01-04 → 05 flow; it is incompatible
    # with the alternate phase flags and with --dry-run (workers write the 04
    # files that the post-merge Stage 05 reads back).
    if sub_agents > 1 and (dry_run or synthesis_only or resume_reviewed or stop_after_capture):
        raise click.UsageError(
            "--sub-agents > 1 cannot be combined with --dry-run or the "
            "--synthesis-only / --resume-reviewed / --stop-after-capture phase flags."
        )

    cfg_path = config_path or DEFAULT_CONFIG_PATH
    cfg = _load_config(cfg_path, fallback_model=model)
    try:
        set_vault_root(cfg.vault_root, strict=True)
    except VaultRootError as exc:
        raise click.UsageError(str(exc)) from exc
    set_dry_run(dry_run)
    configure_whisper(backend=cfg.whisper_backend, model=cfg.whisper_model)
    vault_root = cfg.vault_root
    filler_words = cfg.filler_words

    project_root = Path(__file__).resolve().parent.parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    configure_alert_sink(logs_dir / "sanitize_alerts.jsonl")

    swept = sweep_stale_tmp(project_root / "tmp")
    if swept:
        click.echo(f"swept {swept} stale tmp video file(s)")

    # Initialize LLM providers from config.json, applying the runtime
    # --provider / --hybrid overrides (config is the source of truth when
    # neither is given). See providers/selection.py.
    config_data = json.loads(cfg_path.read_text(encoding="utf-8"))
    providers_raw = config_data.get("providers", {})
    if (provider == "anthropic" or hybrid) and "anthropic" not in providers_raw:
        raise click.UsageError(
            "--provider anthropic / --hybrid requires the 'anthropic' provider in config.json."
        )
    # Seed from cfg.models — the NORMALIZED map _load_config builds with the
    # per-stage fallbacks already applied (router→"haiku", other unspecified
    # stages→the --model value). Using it (not the raw config) keeps --model
    # and partial-config fallbacks honored for missing roles; resolve_role
    # handles both the object ({provider, model}) and legacy string forms.
    effective_models, model_warnings = apply_selection(
        cfg.models, providers_raw, _MODEL_KEYS, provider=provider, hybrid=hybrid
    )
    for warning in model_warnings:
        click.echo(warning)
    configure_providers(providers_raw, effective_models)
    # Resolve each stage's concrete model NAME from the SAME effective map that
    # drives provider resolution, and pass THAT as the explicit `model=` arg.
    # invoke_llm only substitutes the role-resolved model when the caller
    # passes "default", so a per-stage object config (`{provider, model}`) or a
    # --provider override must be flattened to a model-name string here — else
    # the dict / a mismatched config model name would reach the provider.
    models = {stage: resolve_role(stage)[1] for stage in _MODEL_KEYS}
    if provider or hybrid:
        click.echo(f"model selection: provider={provider or 'config'} hybrid={hybrid}")
    click.echo(
        f"providers: {', '.join(providers_raw.keys()) if providers_raw else 'default (ollama)'}"
    )
    click.echo("llm_backends: SDK mode (no claude CLI dependency)")

    # Persistent cache + per-role LLM cache policy. ``--no-cache`` is the
    # master off switch; otherwise deterministic artifacts (transcript/video/
    # code) and Stage 02/04/router LLM output are cached, while Stage 05
    # synthesis is opt-in via ``--cache-llm-synthesis``.
    from .cache import configure_cache
    from .stages.capture import configure_download_concurrency
    from .transcript.whisper_fallback import (
        configure_whisper_concurrency,
        configure_whisper_max_audio_seconds,
    )

    cache = configure_cache(
        cache_dir or cfg.cache_dir,
        enabled=not no_cache,
        max_video_bytes=cfg.cache_max_video_bytes,
    )
    configure_llm_cache(stages=True, synthesis=cache_llm_synthesis)
    if cfg.whisper_concurrency:
        configure_whisper_concurrency(cfg.whisper_concurrency)
    if cfg.whisper_max_audio_seconds is not None:
        configure_whisper_max_audio_seconds(cfg.whisper_max_audio_seconds)
    # Resource-class caps (Phase 3 A): CLI flag overrides config; None=unbounded.
    configure_llm_concurrency(llm_concurrency or cfg.llm_concurrency)
    configure_download_concurrency(download_concurrency or cfg.download_concurrency)
    click.echo(
        f"cache: {'disabled' if not cache.enabled else cache.root} "
        f"(llm synthesis cache: {'on' if cache_llm_synthesis else 'off'})"
    )

    # Resolve the Stage 03 capture backend. CLI flag beats config.json; both
    # default to "host". The preflight for Docker mode is deferred until we
    # know Stage 03 will actually run — workflows that skip capture
    # (`--synthesis-only`, `--resume-reviewed`) must not fail just because
    # the docker daemon happens to be unavailable at that moment.
    active_capture_backend: Any = None
    backend_choice = capture_backend or cfg.capture_backend
    # Capture runs in every mode except --synthesis-only (which only re-runs
    # Stage 05 over existing 04 md). In particular --resume-reviewed still calls
    # _process_video()/Stage 03, so it must run the docker preflight and be
    # subject to the local-media guard below.
    will_run_capture = not synthesis_only
    # --local-media files live outside the container's bind mounts (tmp/ + the
    # Vault assets folder), so the docker backend's ffmpeg can't read them.
    # Reject the combination up front instead of failing per-video deep inside
    # Stage 03.
    if local_media and backend_choice == "docker" and will_run_capture:
        raise click.UsageError(
            "--local-media is incompatible with the docker capture backend: the "
            "hardened container only mounts tmp/ and the Vault assets folder, so "
            "your media directory is not visible to ffmpeg. Re-run with the host "
            "backend (--capture-backend host)."
        )
    if backend_choice == "docker":
        assets_dir = vault_root / ASSETS_REL_PATH
        assets_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = project_root / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        active_capture_backend = DockerCaptureBackend(
            tmp_dir=tmp_dir,
            assets_dir=assets_dir,
            image=cfg.capture_docker_image,
        )
        if will_run_capture:
            try:
                active_capture_backend.preflight()
            except DockerBackendNotReady as exc:
                raise click.UsageError(str(exc)) from exc
            click.echo(f"capture_backend: docker ({cfg.capture_docker_image})")
        else:
            click.echo(
                f"capture_backend: docker ({cfg.capture_docker_image}) "
                "[preflight deferred: capture not needed this run]"
            )
    else:
        click.echo("capture_backend: host")

    effective_synthesis_timeout = synthesis_timeout or cfg.synthesis_timeout
    effective_synthesis_profile = synthesis_profile or cfg.synthesis_profile or "auto"

    click.echo(f"vault_root: {vault_root}")
    click.echo(f"dry_run: {dry_run}")
    click.echo(f"model: {model}")
    click.echo(f"whisper: {describe_whisper()}")
    click.echo(f"capture_format: {capture_format}")
    click.echo(f"concurrency: {concurrency}")
    click.echo(f"min_playlist_size: {min_playlist_size}")
    click.echo(f"max_chapters: {max_chapters if max_chapters is not None else 'auto'}")
    click.echo(
        f"synthesis_timeout: {effective_synthesis_timeout}s"
        if effective_synthesis_timeout
        else "synthesis_timeout: auto"
    )
    click.echo(f"synthesis_profile: {effective_synthesis_profile}")

    # --local-media: build the video list from a local folder (no YouTube).
    # media_map (video_id → file path) is threaded into stages 01/03 so they
    # transcribe/capture the local file instead of downloading. Empty otherwise.
    media_map: dict[str, Path] = {}
    if local_media:
        from .local_media import build_local_videos

        videos, media_map = build_local_videos(local_media)
        if not videos:
            click.echo(f"No media files found in {local_media}")
            sys.exit(1)
        click.echo(f"local-media: {len(videos)} file(s) from {local_media}")
    else:
        if url is None:
            raise click.UsageError(
                "A playlist/video URL is required unless --local-media is given."
            )
        click.echo("fetching metadata...")
        videos = fetch_metadata(url)
        if not videos:
            click.echo("No videos found.")
            sys.exit(1)

    playlist_title = videos[0].playlist_title or videos[0].title or "single video"
    click.echo(f"playlist: {playlist_title!r}")
    click.echo(f"videos: {len(videos)}")

    # Stage 00.5: Router. One cheap haiku call decides whether downstream
    # code-bearing features (GitHub URL extraction, concept/practice split)
    # apply. Errors collapse to Genre.OTHER → default behavior. The parent
    # classifies once and pins the result for every sub-agent shard (internal
    # --code-bearing/--no-code-bearing), so a transient router error on one
    # worker can't leave shards disagreeing on code_bearing.
    if code_bearing_override is not None:
        code_bearing = code_bearing_override
        click.echo(f"genre: (inherited from parent) code_bearing={code_bearing}")
    else:
        genre, genre_rationale = classify_playlist_genre(
            playlist_title, videos, model=models["router"]
        )
        code_bearing = genre in CODE_BEARING_GENRES
        click.echo(f"genre: {genre.value} (code_bearing={code_bearing}) — {genre_rationale[:120]}")

    # Sub-agent orchestration (opt-in via --sub-agents N; default 1 keeps the
    # original single-process flow). Splits the playlist into N contiguous
    # shards, runs stages 01-04 as independent parallel worker processes, then
    # runs Stage 05 once over the merged output. See docs/sub-agents.md.
    if sub_agents > 1:
        orchestrator_run_time = _parse_run_timestamp(run_timestamp)
        click.echo(f"run_time: {orchestrator_run_time.isoformat(timespec='seconds')}")
        exit_code = orchestrate_sub_agents(
            total_videos=len(videos),
            shard_count=sub_agents,
            run_time=orchestrator_run_time,
            logs_dir=logs_dir,
            base_argv=strip_cli_option(sys.argv[1:], "--sub-agents"),
            run_synthesis=not skip_synthesis,
            code_bearing=code_bearing,
        )
        sys.exit(exit_code)

    # Sub-agent shard slicing (internal --video-range). Restricts the per-video
    # work below to this shard's contiguous slice; genre above is already pinned.
    if video_range is not None:
        try:
            shard_start, shard_end = parse_video_range(video_range)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
        videos = videos[shard_start:shard_end]
        click.echo(
            f"sub-agent shard: videos [{shard_start}:{shard_end}] → {len(videos)} this shard"
        )
        if not videos:
            click.echo("No videos in this shard; nothing to do.")
            return

    est_timeouts = compute_synthesis_timeouts(len(videos), override=effective_synthesis_timeout)
    total_duration = sum(v.duration or 0 for v in videos)
    click.echo(
        f"synthesis_estimate: {len(videos)} videos"
        f" → timeout α={est_timeouts['alpha']}s β={est_timeouts['beta']}s"
        f" leader={est_timeouts['leader']}s"
        + (f", total_duration={total_duration // 60}min" if total_duration else "")
    )

    run_time = _parse_run_timestamp(run_timestamp)
    click.echo(f"run_time: {run_time.isoformat(timespec='seconds')}")

    folder_override: str | None = None
    if synthesis_only:
        click.echo("\n=== --synthesis-only: loading existing 04 md files ===")
        matched_videos, matched_bodies, folder_override = _collect_existing_learning_bodies(
            videos, playlist_title, run_time
        )
        click.echo(f"matched: {len(matched_videos)}/{len(videos)} videos")
        if len(matched_videos) < min_playlist_size:
            click.echo(
                f"[skip] only {len(matched_videos)} matched (< {min_playlist_size}), "
                "stage 05 skipped"
            )
            return
        synthesis_videos = matched_videos
        synthesis_bodies = matched_bodies
    else:
        # Checkpoint: detect already-completed videos in one pass
        force_set = set(force_video)
        completed_ids = get_completed_video_ids(playlist_title, run_time) if not dry_run else set()
        if completed_ids:
            skippable = completed_ids - force_set
            if skippable:
                click.echo(f"checkpoint: {len(skippable)} videos already complete, will skip")

        # Separate videos into skip (checkpoint) and process lists
        to_process: list[tuple[int, VideoMeta]] = []
        results: list[VideoRunResult] = []
        for i, video in enumerate(videos, 1):
            if video.video_id in completed_ids and video.video_id not in force_set:
                click.echo(f"\n[{i}/{len(videos)}] {video.video_id} {video.title}")
                click.echo("  [skip] checkpoint: stage 04 already exists")
                body = _load_existing_04_body(video.video_id, playlist_title, run_time)
                results.append(VideoRunResult(video=video, learning_md_body=body))
            else:
                to_process.append((i, video))

        if resume_reviewed:
            # Phase 3: filter to videos whose 02_Summary.md has `reviewed: true`.
            to_process = _filter_to_reviewed(to_process, playlist_title, run_time)

        # Warm the transcript cache for all to-be-processed videos up front,
        # at a higher fan-out than --concurrency. Skipped under
        # --resume-reviewed (Stage 01 doesn't run) and a no-op when the cache
        # is disabled or Whisper is the only available tier.
        # Skipped under --local-media (warm-up fetches YouTube captions, which
        # the offline path never uses).
        if to_process and not resume_reviewed and not local_media:
            warm_conc = (
                transcript_concurrency
                or cfg.transcript_concurrency
                or DEFAULT_TRANSCRIPT_CONCURRENCY
            )
            warmed = warm_transcript_cache([v for _, v in to_process], concurrency=warm_conc)
            if warmed:
                click.echo(f"transcript warm-up: cached {warmed}/{len(to_process)} captions")

        # Process remaining videos
        if to_process and concurrency > 1:
            process_videos = [v for _, v in to_process]
            concurrent_results = asyncio.run(
                _run_videos_concurrent(
                    process_videos,
                    run_time,
                    concurrency=concurrency,
                    dry_run=dry_run,
                    capture_format=capture_format,
                    models=models,
                    filler_words=filler_words,
                    stop_after_capture=stop_after_capture,
                    capture_backend=active_capture_backend,
                    code_bearing=code_bearing,
                    glossary=cfg.glossary,
                    media_map=media_map,
                    correct_transcript=cfg.transcript_correction,
                )
            )
            results.extend(concurrent_results)
        else:
            for i, video in to_process:
                click.echo(f"\n[{i}/{len(videos)}] {video.video_id} {video.title}")
                result = _process_video(
                    video,
                    run_time,
                    dry_run=dry_run,
                    capture_format=capture_format,
                    models=models,
                    filler_words=filler_words,
                    stop_after_capture=stop_after_capture,
                    capture_backend=active_capture_backend,
                    code_bearing=code_bearing,
                    glossary=cfg.glossary,
                    media_path=media_map.get(video.video_id),
                    correct_transcript=cfg.transcript_correction,
                )
                results.append(result)

        if stop_after_capture:
            click.echo(
                "\n[stop-after-capture] Phase 1 complete. Review 02_Summary.md, "
                "set `reviewed: true`, then re-run with --resume-reviewed."
            )
            return

        succeeded = [r for r in results if r.ok]
        failed = [r for r in results if not r.ok]

        click.echo("\n=== Video processing summary ===")
        click.echo(f"succeeded: {len(succeeded)}/{len(videos)}")
        for f in failed:
            click.echo(f"  FAIL {f.video.video_id}: {f.error}")

        if skip_synthesis:
            click.echo("[skip] --skip-synthesis: stage 05 bypassed")
            return

        if len(succeeded) < min_playlist_size:
            click.echo(
                f"[skip] only {len(succeeded)} videos succeeded (< {min_playlist_size}), "
                "stage 05 skipped"
            )
            return
        synthesis_videos = [r.video for r in succeeded]
        synthesis_bodies = [r.learning_md_body or "" for r in succeeded]

    click.echo("\n=== Stage 05 Synthesis (Agent Teams) ===")
    synth_timeouts = compute_synthesis_timeouts(
        len(synthesis_videos), override=effective_synthesis_timeout
    )
    click.echo(log_synthesis_preflight(len(synthesis_videos), synthesis_bodies, synth_timeouts))
    synthesis_result = run_stage_synthesis(
        synthesis_videos,
        synthesis_bodies,
        run_time=run_time,
        playlist_title=playlist_title,
        model=model,
        agent_models={k: models[k] for k in ("alpha", "beta", "leader", "reviewer")},
        min_playlist_size=min_playlist_size,
        max_chapters=max_chapters,
        dry_run=dry_run,
        folder_name_override=folder_override,
        synthesis_timeout=effective_synthesis_timeout,
        profile=effective_synthesis_profile,
    )

    if synthesis_result.skipped:
        click.echo(f"[skip] {synthesis_result.skip_reason}")
    elif synthesis_result.error:
        click.echo(f"[error] synthesis: {synthesis_result.error}")
    else:
        click.echo(f"MOC:       {synthesis_result.moc_path}")
        click.echo(f"chapters:  {len(synthesis_result.chapter_paths)}")
        for p in synthesis_result.chapter_paths:
            click.echo(f"  - {p.name}")
        click.echo(f"meta:      {synthesis_result.meta_path}")
        click.echo(
            f"tokens:    in={synthesis_result.total_input_tokens}"
            f" out={synthesis_result.total_output_tokens}"
            f" cache_read={synthesis_result.total_cache_read_tokens}"
            f" cache_create={synthesis_result.total_cache_creation_tokens}"
        )
        click.echo(f"cost:      ${synthesis_result.total_cost_usd:.3f}")
        click.echo(f"duration:  {synthesis_result.total_duration_ms / 1000:.1f}s")

    # NOTE: the evaluation phase (run_stage_evaluation, gated by --eval-loop)
    # hooks in here after Stage 05 once implemented. It is rejected up front
    # while scaffolded — see the --eval-loop guard at the top of cli().

    if not synthesis_only:
        _print_cost_breakdown(results, synthesis_result)


if __name__ == "__main__":
    cli()
