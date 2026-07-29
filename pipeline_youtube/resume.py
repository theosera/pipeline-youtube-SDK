"""Lookup of prior-run output for the resume / synthesis-only flows.

Extracted from `main.py`. These helpers locate existing Stage 02/04 notes on
disk (by trusted frontmatter video_id) so `--synthesis-only`,
`--resume-reviewed`, and checkpoint-skip can rebuild their inputs without
reprocessing.
"""

from __future__ import annotations

import re
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

# Playlist folders are named "YYYY-MM-DD-HHmm <title>" (legacy runs omit HHmm).
# The match covers the date (and time when present); the rest is the title.
#
# The trailing lookahead is what makes this a *prefix* rather than a substring:
# without it `2026-04-18testlist` matches, and `_folder_title` then reports
# "testlist" for a directory the pipeline never created.
_DATED_FOLDER_RE = re.compile(r"(\d{4}-\d{2}-\d{2})(?:-(\d{4}))?(?=\s|$)")


def _dated_folder_match(folder_name: str) -> re.Match[str] | None:
    """Match a pipeline-created folder prefix, or None when the name is not one.

    The regex only fixes the *shape*; a folder called ``2026-00-00 testlist``
    still matches it and sorts before today, so it would reach the earlier-day
    tier. Parsing the captured date and time is what rejects that.
    """
    match = _DATED_FOLDER_RE.match(folder_name)
    if match is None:
        return None
    try:
        datetime.strptime(match.group(1), "%Y-%m-%d")
        if match.group(2) is not None:
            datetime.strptime(match.group(2), "%H%M")
    except ValueError:
        return None
    return match


def _parse_run_timestamp(run_timestamp: str | None) -> datetime:
    """Resolve the shared run_time, surfacing a bad --run-timestamp as a clean CLI error."""
    if not run_timestamp:
        return datetime.now()
    try:
        return datetime.fromisoformat(run_timestamp)
    except ValueError as exc:
        raise click.UsageError(f"invalid --run-timestamp: {run_timestamp!r}") from exc


def _unit_base_dir(unit_key: str, *, vault_root: Path) -> Path | None:
    """Resolve the vault dir holding a unit's playlist folders, or None if absent.

    ``vault_root`` is injected by the caller (``runtime.vault_root``); the
    relative path still goes through ``ensure_safe_path`` so a malformed
    ``UNIT_DIRS`` entry can never escape the vault.
    """
    if unit_key not in UNIT_DIRS:
        raise ValueError(f"unknown unit key: {unit_key!r}")
    rel = f"{LEARNING_BASE}/{UNIT_DIRS[unit_key]}"
    base = vault_root / ensure_safe_path(rel, vault_root=vault_root)
    return base if base.exists() else None


def _find_unit_md(
    video_id: str,
    playlist_title: str,
    run_date: datetime,
    unit_key: str,
    *,
    vault_root: Path,
    preferred_folder_name: str | None = None,
) -> Path | None:
    """Locate an existing unit md for `video_id` within a given run date.

    Used by Phase 3 (`--resume-reviewed`) to look up Stage 02/03 notes written
    in a prior Phase 1 run. Falls back across date-prefix matches — same day
    first, then earlier days — so users can review overnight and resume without
    re-passing ``--run-timestamp``.

    ``preferred_folder_name`` pins the search to the playlist folder already
    resolved for a sibling unit (e.g. reuse the Stage 02 folder when locating
    Stage 03), so 02/03/04 stay aligned even when several same-day folders exist.
    A pinned lookup that misses returns None rather than falling back — see below.
    """
    base = _unit_base_dir(unit_key, vault_root=vault_root)
    if base is None:
        return None

    if preferred_folder_name:
        # Fail closed. The caller pinned this folder so 02/03/04 all come from
        # one Phase 1 run; falling through to another same-day run would pair
        # the reviewed summary with unrelated captures, whose embeds are
        # path-qualified to *that* run's playlist folder. Returning None lets
        # the caller report reviewed_capture_not_found instead of mixing runs.
        preferred = base / preferred_folder_name
        if not preferred.exists():
            return None
        for md in preferred.glob("*.md"):
            if read_trusted_video_id(md) == video_id:
                return md
        return None

    for candidate_folder in _unit_folder_candidates(base, playlist_title, run_date):
        if not candidate_folder.exists():
            continue
        for md in candidate_folder.glob("*.md"):
            if read_trusted_video_id(md) == video_id:
                return md
    return None


def _find_summary_md(
    video_id: str, playlist_title: str, run_date: datetime, *, vault_root: Path
) -> Path | None:
    """Locate the existing 02_Summary.md for `video_id` within a given run date.

    Used by Phase 3 (`--resume-reviewed`) to look up summaries written
    in a prior Phase 1 run. Falls back across date-prefix matches so
    users can resume on a different clock day.

    ``vault_root`` is injected by the caller (``runtime.vault_root``).
    """
    return _find_unit_md(video_id, playlist_title, run_date, "summary", vault_root=vault_root)


def _find_reviewed_summary_md(
    video_id: str, playlist_title: str, run_date: datetime, *, vault_root: Path
) -> Path | None:
    """Locate a 02_Summary.md for ``video_id`` with frontmatter ``reviewed: true``.

    Phase 1 reruns create a newer folder whose summaries still have
    ``reviewed: false``. Looking up "newest note for video_id, then check
    reviewed" would skip the older folder the operator actually marked.
    Scan newest-first and return the first matching *reviewed* summary.

    Because candidates now extend to earlier days, this reviewed-first rule is
    what keeps the widening safe: an old folder is only ever reached when
    nothing newer holds an approved summary for this video.
    """
    from .obsidian import read_frontmatter_field

    base = _unit_base_dir("summary", vault_root=vault_root)
    if base is None:
        return None

    for candidate_folder in _unit_folder_candidates(base, playlist_title, run_date):
        if not candidate_folder.exists():
            continue
        for md in candidate_folder.glob("*.md"):
            if read_trusted_video_id(md) != video_id:
                continue
            value = read_frontmatter_field(md, "reviewed")
            if value and value.lower() == "true":
                return md
    return None


def _find_existing_04_md(
    video_id: str, playlist_title: str, run_date: datetime, *, vault_root: Path
) -> Path | None:
    """Locate the stage 04 md for a checkpoint-skipped video."""
    from .checkpoint import _find_learning_folder

    folder = _find_learning_folder(playlist_title, run_date, vault_root=vault_root)
    if folder is None:
        return None
    for md in folder.glob("*.md"):
        if read_trusted_video_id(md) == video_id:
            return md
    return None


def _load_existing_04_body(
    video_id: str, playlist_title: str, run_date: datetime, *, vault_root: Path
) -> str | None:
    """Read the stage 04 body for a checkpoint-skipped video.

    Returns the frontmatter-stripped body, or None if the file can't be found.
    Uses the same M3 hardened frontmatter validation as `is_video_complete`.

    ``vault_root`` is injected by the caller (``runtime.vault_root``).
    """
    md = _find_existing_04_md(video_id, playlist_title, run_date, vault_root=vault_root)
    if md is None:
        return None
    try:
        text = md.read_text(encoding="utf-8")
    except OSError:
        return None
    return _strip_frontmatter(text)


def _learning_path_for_reviewed_summary(
    summary_md: Path, video: VideoMeta, *, vault_root: Path
) -> Path:
    """Return the Stage 04 output path that matches a reviewed Stage 02 note.

    Phase 3 must write 04 into the same playlist-folder name as Phase 1's 02/03
    notes. Using ``compute_note_paths`` with Phase 3's wall-clock ``run_time``
    would create a sibling folder and orphan the reviewed inputs.
    """
    from .obsidian import resolve_unique_path

    rel_folder = f"{LEARNING_BASE}/{UNIT_DIRS['learning']}/{summary_md.parent.name}"
    safe_rel_folder = ensure_safe_path(rel_folder, vault_root=vault_root)
    folder = vault_root / safe_rel_folder
    candidate = folder / summary_md.name
    if not candidate.exists() or read_trusted_video_id(candidate) == video.video_id:
        return candidate
    return resolve_unique_path(folder, summary_md.stem, ".md")


def _filter_to_reviewed(
    to_process: list[tuple[int, VideoMeta]],
    playlist_title: str,
    run_time: datetime,
    *,
    vault_root: Path,
) -> list[tuple[int, VideoMeta]]:
    """Keep only videos that have a 02_Summary.md with `reviewed: true`.

    Searches every candidate playlist folder (newest first), not only the
    newest note for the video_id — a later unreviewed Phase 1 rerun must
    not hide an older summary the operator already approved.

    A match from a previous day is echoed rather than used silently: resuming
    across midnight is the normal case, but *which* run Stage 04 is about to
    consume is exactly the thing an operator needs to see confirmed.

    ``vault_root`` is injected by the caller (``runtime.vault_root``).
    """
    from .obsidian import read_frontmatter_field

    today = run_time.strftime("%Y-%m-%d")
    kept: list[tuple[int, VideoMeta]] = []
    for i, video in to_process:
        summary_md = _find_reviewed_summary_md(
            video.video_id, playlist_title, run_time, vault_root=vault_root
        )
        if summary_md is not None:
            folder_date = summary_md.parent.name[:10]
            if folder_date != today:
                click.echo(f"  [resume] {video.video_id}: reviewed on {folder_date}")
            kept.append((i, video))
            continue
        # Nothing reviewed matched. Re-scan without the reviewed filter so the
        # skip line names the real cause: "no summary written at all" and
        # "summary written but never approved" need different operator action.
        any_summary = _find_summary_md(
            video.video_id, playlist_title, run_time, vault_root=vault_root
        )
        if any_summary is None:
            click.echo(f"  [skip] {video.video_id}: no 02_Summary.md found")
        else:
            value = read_frontmatter_field(any_summary, "reviewed")
            click.echo(f"  [skip] {video.video_id}: reviewed={value!r}")
    return kept


def _unit_folder_candidates(base: Path, playlist_title: str, run_date: datetime) -> Iterator[Path]:
    """Yield likely playlist folders holding unit files.

    Order: canonical, then same-day folders (newest first), then folders from
    earlier days (newest first). Matching is by sanitized title substring
    (mirrors `_find_learning_folder` heuristics).

    The earlier-day tier is what makes Phase 3 work across midnight. The
    workflow is "Phase 1 → a human reads 02_Summary.md → Phase 3", and that
    review routinely happens the next morning; a same-day-only search then
    reports "no 02_Summary.md found" for work that is sitting right there.
    Same-day is offered first, and only folders *older* than ``run_date`` are
    considered, so a stray future-dated folder (clock skew, a hand-typed
    ``--run-timestamp``) can never outrank today's run. Historical folders must
    also match the title exactly — see the comment on that tier below.
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
        matches = [
            child
            for child in base.iterdir()
            if (
                child.is_dir()
                # Require a real YYYY-MM-DD prefix rather than any directory:
                # widening past today must not start matching unrelated folders
                # that merely share a word with the playlist title.
                and _dated_folder_match(child.name)
                and child.name != canonical_name
            )
        ]
    except OSError:
        return
    # The title test belongs to each tier, not to this pre-filter: the two tiers
    # compare differently (substring vs exact) and a shared raw-substring gate
    # here would reject a legacy folder before the exact tier could sanitize it.
    #
    # iterdir() order is filesystem-dependent, so with two Phase 1 runs on one
    # day the caller could silently get either. Folder names start with
    # YYYY-MM-DD-HHmm, so a descending name sort puts the newest run first —
    # the one the operator most likely just reviewed.
    matches.sort(key=lambda child: child.name, reverse=True)
    yield from (
        child
        for child in matches
        if child.name.startswith(date_prefix) and title_needle in child.name
    )
    # Earlier days last, so a same-day reviewed summary always wins. The date is
    # a fixed-width YYYY-MM-DD prefix, so a string compare orders it.
    #
    # Historical folders are held to an *exact* title match, unlike the same-day
    # tier's substring rule. Substring matching is safe within one day because a
    # run only just created those folders; across all of history it would admit
    # a different playlist whose title merely contains this one — resuming
    # "Python" would accept "2026-04-17-0900 Python Advanced", and if that run
    # covered the same video its reviewed summary (and, via the pinned lookup,
    # its captures) would be consumed instead.
    yield from (
        child
        for child in matches
        if child.name[:10] < date_prefix and _folder_title(child.name) == title_needle
    )


def _folder_title(folder_name: str) -> str:
    """Return a playlist folder's sanitized title, stripped of its date prefix.

    Folder names are ``YYYY-MM-DD-HHmm <title>``; runs from before the HHmm fix
    use ``YYYY-MM-DD <title>``. Sanitizing keeps this comparable with a needle
    built from a live playlist title even when the folder on disk predates the
    concealment defenses.
    """
    match = _dated_folder_match(folder_name)
    if match is None:
        return ""
    from .obsidian import sanitize_title_for_filename

    return sanitize_title_for_filename(folder_name[match.end() :].strip())


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
    *,
    vault_root: Path,
) -> tuple[list[VideoMeta], list[str], str]:
    """Scan the existing 04_Learning_Material folder for the given playlist date
    and return `(videos, bodies, folder_name)` aligned by input video_id order.

    Also returns the resolved folder name so stage 05 can reuse the exact
    legacy name instead of creating a new one next to it.

    ``vault_root`` is injected by the caller (``runtime.vault_root``).
    """
    rel_base = f"{LEARNING_BASE}/{UNIT_DIRS['learning']}"
    safe_rel_base = ensure_safe_path(rel_base, vault_root=vault_root)
    base_dir = vault_root / safe_rel_base

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
