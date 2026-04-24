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

from datetime import datetime
from pathlib import Path

from .config import get_vault_root
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
) -> dict[str, Path]:
    """Return the target md path for each requested unit without writing.

    Use this when you need to know where a stage will write its output
    before the stage runs — e.g. stage 04 creating its own md file
    directly. The path is collision-resolved (-2, -3 suffix) against the
    existing filesystem state.
    """
    vault_root = get_vault_root()
    playlist_folder = format_playlist_folder_name(run_time, video.playlist_title)
    note_base = format_video_note_base(run_time, video.title)

    paths: dict[str, Path] = {}
    for unit_key in units:
        if unit_key not in UNIT_DIRS:
            raise ValueError(f"unknown unit key: {unit_key!r}")
        rel_path = f"{LEARNING_BASE}/{UNIT_DIRS[unit_key]}/{playlist_folder}"
        safe_rel = ensure_safe_path(rel_path)
        folder = vault_root / safe_rel
        paths[unit_key] = resolve_unique_path(folder, note_base, ".md")
    return paths


def create_placeholder_notes(
    video: VideoMeta,
    run_time: datetime,
    *,
    units: tuple[str, ...] = DEFAULT_PLACEHOLDER_UNITS,
    dry_run: bool = False,
) -> dict[str, Path]:
    """Create empty md placeholders for the specified units.

    By default only 01/02/03 are created — 04 is skipped to avoid
    Templater folder-template interference on empty files. Pass
    `units=("scripts", "summary", "capture", "learning")` explicitly
    if all four are needed (e.g. legacy tests).

    Returns `{unit_key: absolute_path}` for whatever was created.
    """
    vault_root = get_vault_root()
    playlist_folder = format_playlist_folder_name(run_time, video.playlist_title)
    note_base = format_video_note_base(run_time, video.title)

    paths: dict[str, Path] = {}
    for unit_key in units:
        if unit_key not in UNIT_DIRS:
            raise ValueError(f"unknown unit key: {unit_key!r}")
        unit_dir = UNIT_DIRS[unit_key]
        rel_path = f"{LEARNING_BASE}/{unit_dir}/{playlist_folder}"
        safe_rel = ensure_safe_path(rel_path)
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
