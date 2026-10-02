"""Video-level pipeline orchestration.

Exposes path computation and placeholder creation for the 4 processing
units (01 Scripts / 02 Summary / 03 Capture / 04 Learning_Material).

Templater interaction note
--------------------------
Obsidian Templater's folder-template feature triggers on file-open for
**empty** files (only frontmatter, no body). Stages 01/02/03 each append
a substantial body to their placeholder as soon as their work completes,
so Templater never sees those files as "empty". Stage 04, however, only
runs after 02 and 03 complete — if we pre-created a 04 placeholder, it
would be empty for ~90 seconds, during which Templater can hijack it
(renaming the file and asking the user for a title, overwriting our
frontmatter).

Fix: `create_placeholder_notes` only creates **01, 02, 03** by default.
Stage 04's implementation writes the 04 md directly when it has content,
bypassing the empty-file window entirely. Callers that need the 04 path
ahead of time should use `compute_note_paths` (pure path calc, no write).
"""

from __future__ import annotations

import threading
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .obsidian import (
    build_frontmatter,
    format_playlist_folder_name,
    format_video_note_base,
    resolve_unique_path,
)
from .path_safety import ensure_safe_path
from .playlist import VideoMeta

# Base Obsidian folder for YouTube learning notes.
LEARNING_BASE = "Permanent Note/08_YouTube学習"
UNIT_DIRS: dict[str, str] = {
    "scripts": "01_Scripts_Processing_Unit",
    "summary": "02_Summary_Processing_Unit",
    "capture": "03_Capture_Processing_Unit",
    "learning": "04_Learning_Material",
}
# Historical (typo) folder name. Kept only for backward-compat lookups in
# `checkpoint._find_learning_folder`; new writes always use UNIT_DIRS.
LEGACY_LEARNING_DIR = "04_Lerning_Material"

# Units that get pre-created as empty placeholders before stages run.
# 'learning' is excluded — see the module docstring for why.
DEFAULT_PLACEHOLDER_UNITS: tuple[str, ...] = ("scripts", "summary", "capture")


def compute_note_paths(
    video: VideoMeta,
    run_time: datetime,
    *,
    units: tuple[str, ...] = ("scripts", "summary", "capture", "learning"),
    vault_root: Path,
) -> dict[str, Path]:
    """Return the target md path for each requested unit without writing.

    Use this when you need to know where a stage will write its output
    before the stage runs — e.g. stage 04 creating its own md file
    directly. The path is collision-resolved (-2, -3 suffix) against the
    existing filesystem state.

    ``vault_root`` is injected by the caller (``runtime.vault_root``).
    """
    playlist_folder = format_playlist_folder_name(run_time, video.playlist_title)
    note_base = format_video_note_base(run_time, video.title)

    paths: dict[str, Path] = {}
    for unit_key in units:
        if unit_key not in UNIT_DIRS:
            raise ValueError(f"unknown unit key: {unit_key!r}")
        rel_path = f"{LEARNING_BASE}/{UNIT_DIRS[unit_key]}/{playlist_folder}"
        safe_rel = ensure_safe_path(rel_path, vault_root=vault_root)
        folder = vault_root / safe_rel
        paths[unit_key] = resolve_unique_path(folder, note_base, ".md")
    return paths


@dataclass(eq=False)
class NoteReservations:
    """Note paths handed out by `reserve_note_paths` during one pipeline run.

    Keyed by `_reservation_key`. 04 is never written as an empty placeholder
    (and a dry run writes nothing), so the filesystem alone cannot tell a
    concurrent task of the same run that such a path is already taken.

    Create one per run and share it across that run's tasks. A registry that
    outlived the run would make a later run in the same process skip paths
    nothing occupies: a dry run followed by a real run picked ``-2``.
    """

    lock: threading.Lock = field(default_factory=threading.Lock)
    keys: set[str] = field(default_factory=set)


def _reservation_key(path: Path) -> str:
    """Fold a path the way a case- and normalization-insensitive volume does.

    APFS (the macOS default) treats ``Foo.md`` / ``foo.md`` and NFC / NFD
    spellings as one file. Comparing folded keys keeps two such titles from
    reserving the same file; on a case-sensitive volume they only move one of
    them to the next suffix.
    """
    return unicodedata.normalize("NFC", str(path)).casefold()


def reserve_note_paths(
    video: VideoMeta,
    run_time: datetime,
    *,
    reservations: NoteReservations,
    dry_run: bool = False,
    vault_root: Path,
) -> dict[str, Path]:
    """Reserve one path per unit for ``video`` and create the 01-03 placeholders.

    The whole set shares one suffix (``""``, ``-2``, ``-3`` ...): the first
    suffix that is free on disk and not in ``reservations`` (this run's other
    tasks) for every unit. Under ``--concurrency`` >= 2 two same-title videos
    otherwise pick the same stem and overwrite each other's notes (#182).

    ``vault_root`` is injected by the caller (``runtime.vault_root``).
    """
    playlist_folder = format_playlist_folder_name(run_time, video.playlist_title)
    note_base = format_video_note_base(run_time, video.title)
    folders: dict[str, Path] = {}
    for unit_key, unit_dir in UNIT_DIRS.items():
        rel_path = f"{LEARNING_BASE}/{unit_dir}/{playlist_folder}"
        folders[unit_key] = vault_root / ensure_safe_path(rel_path, vault_root=vault_root)

    with reservations.lock:
        i = 1
        while True:
            suffix = "" if i == 1 else f"-{i}"
            candidate = {k: f / f"{note_base}{suffix}.md" for k, f in folders.items()}
            if not any(
                p.exists() or _reservation_key(p) in reservations.keys for p in candidate.values()
            ):
                break
            i += 1
        reservations.keys.update(_reservation_key(p) for p in candidate.values())

        # Written before the lock is released, so a task that reserves next
        # sees these files on disk even under a spelling the registry key
        # would not fold.
        if not dry_run:
            for unit_key in DEFAULT_PLACEHOLDER_UNITS:
                path = candidate[unit_key]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    _placeholder_frontmatter(video, run_time, unit_key), encoding="utf-8"
                )
    return candidate


def _placeholder_frontmatter(video: VideoMeta, run_time: datetime, unit_key: str) -> str:
    extra: dict[str, str] = {
        "playlist": video.playlist_title or "",
        "video_id": video.video_id,
    }
    if unit_key == "summary":
        # `reviewed` flags Phase 3 (WS5) that the user has approved the
        # summary for downstream synthesis. User flips to `true` in
        # Obsidian after manual review.
        extra["reviewed"] = "false"
    return build_frontmatter(
        dt=run_time,
        title=video.title,
        url=video.watch_url,
        tags=["memo", "youtube"],
        extra=extra,
    )


def create_placeholder_notes(
    video: VideoMeta,
    run_time: datetime,
    *,
    units: tuple[str, ...] = DEFAULT_PLACEHOLDER_UNITS,
    dry_run: bool = False,
    vault_root: Path,
) -> dict[str, Path]:
    """Create empty md placeholders for the specified units.

    By default only 01/02/03 are created — 04 is skipped to avoid
    Templater folder-template interference on empty files. Pass
    `units=("scripts", "summary", "capture", "learning")` explicitly
    if all four are needed (e.g. legacy tests).

    Returns `{unit_key: absolute_path}` for whatever was created.

    ``vault_root`` is injected by the caller (``runtime.vault_root``).
    """
    playlist_folder = format_playlist_folder_name(run_time, video.playlist_title)
    note_base = format_video_note_base(run_time, video.title)

    paths: dict[str, Path] = {}
    for unit_key in units:
        if unit_key not in UNIT_DIRS:
            raise ValueError(f"unknown unit key: {unit_key!r}")
        unit_dir = UNIT_DIRS[unit_key]
        rel_path = f"{LEARNING_BASE}/{unit_dir}/{playlist_folder}"
        safe_rel = ensure_safe_path(rel_path, vault_root=vault_root)
        folder = vault_root / safe_rel

        if not dry_run:
            folder.mkdir(parents=True, exist_ok=True)

        path = resolve_unique_path(folder, note_base, ".md")
        paths[unit_key] = path

        extra: dict[str, str] = {
            "playlist": video.playlist_title or "",
            "video_id": video.video_id,
        }
        if unit_key == "summary":
            # `reviewed` flags Phase 3 (WS5) that the user has approved the
            # summary for downstream synthesis. User flips to `true` in
            # Obsidian after manual review.
            extra["reviewed"] = "false"
        fm = build_frontmatter(
            dt=run_time,
            title=video.title,
            url=video.watch_url,
            tags=["memo", "youtube"],
            extra=extra,
        )

        if not dry_run:
            path.write_text(fm, encoding="utf-8")

    return paths
