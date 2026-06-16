"""Lookup of prior-run output for the resume / synthesis-only flows.

Extracted from `main.py`. These helpers locate existing Stage 02/04 notes on
disk (by trusted frontmatter video_id) so `--synthesis-only`,
`--resume-reviewed`, and checkpoint-skip can rebuild their inputs without
reprocessing.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import click

from .checkpoint import extract_trusted_video_id, read_trusted_video_id
from .obsidian import format_playlist_folder_name
from .path_safety import ensure_safe_path
from .pipeline import LEARNING_BASE, UNIT_DIRS
from .playlist import VideoMeta
from .run_result import _strip_frontmatter


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
