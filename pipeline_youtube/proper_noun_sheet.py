"""Per-playlist proper-noun sheet maintenance (Stage 01b byproduct).

Extracted from `main.py`. Locates the shared sheet under the 01_Scripts folder,
promotes user corrections back into the configured glossary, and merges each
video's confirmed terms into the sheet under a cross-process write lock.
"""

from __future__ import annotations

import contextlib
from datetime import datetime
from pathlib import Path
from typing import Any

from .glossary import (
    SHEET_FILENAME,
    Glossary,
    GlossaryParseError,
    ProperNounSheet,
    correction_entries,
    load_glossary,
    load_sheet,
    merge_glossary,
    upsert_video_terms,
    write_glossary,
    write_sheet,
)
from .pipeline import compute_note_paths
from .playlist import VideoMeta
from .run_result import VideoRunResult


def _proper_noun_sheet_path(video: VideoMeta, run_time: datetime) -> Path:
    """Path of the per-playlist proper-noun sheet (under the 01_Scripts folder).

    Every video in a playlist shares the same 01_Scripts playlist folder, so the
    sheet's parent is stable regardless of which video is used to derive it.
    """
    scripts_path = compute_note_paths(video, run_time, units=("scripts",))["scripts"]
    return scripts_path.parent / SHEET_FILENAME


def _promote_corrections_to_glossary(sheet: ProperNounSheet, glossary_path: Path) -> int:
    """Promote user-corrected sheet rows into ``glossary.json``; return # added.

    Only rows the user actually corrected (right column filled) are promoted —
    the correction becomes the ``canonical`` and the system spelling its
    ``alias``. The merge is non-destructive and conflict-tolerant; the file is
    rewritten only on a real change. All I/O is best-effort (a broken/locked
    glossary never aborts the run).
    """
    new_entries = correction_entries(sheet)
    if not new_entries:
        return 0
    try:
        base = load_glossary(glossary_path) if glossary_path.exists() else Glossary()
    except (GlossaryParseError, OSError):
        return 0
    merged = merge_glossary(base, new_entries)
    if merged == base:
        return 0
    try:
        write_glossary(glossary_path, merged)
    except OSError:
        return 0
    return len(merged.entries) - len(base.entries)


def _sheet_write_lock(sheet_path: Path) -> contextlib.AbstractContextManager[Any]:
    """Cross-process lock guarding the shared proper-noun sheet.

    Mirrors the whisper-fallback idiom: use ``filelock`` when installed, else a
    no-op (single-process runs are unaffected either way).
    """
    try:
        import filelock  # type: ignore[import-untyped]
    except ImportError:
        return contextlib.nullcontext()
    return filelock.FileLock(str(sheet_path) + ".lock", timeout=-1)


def _update_proper_noun_sheet(sheet_path: Path, results: list[VideoRunResult]) -> None:
    """Merge each video's confirmed terms into the on-disk sheet and rewrite it.

    Existing user corrections are preserved (``upsert_video_terms``). The file is
    left untouched when no video contributed new terms, so a user's hand edits
    are never clobbered by an empty rewrite.

    Concurrency: with ``--sub-agents > 1`` several worker processes share one
    playlist sheet. The read-modify-write therefore runs under a cross-process
    file lock and re-reads the on-disk sheet *inside* the lock, so each shard
    merges only its own videos and never clobbers sections another shard wrote.
    """
    contributing = [r for r in results if r.confirmed_terms]
    if not contributing:
        return
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    with _sheet_write_lock(sheet_path):
        sheet = load_sheet(sheet_path)
        for result in contributing:
            sheet = upsert_video_terms(
                sheet,
                video_id=result.video.video_id,
                title=result.video.title or "",
                terms=list(result.confirmed_terms),
            )
        write_sheet(sheet_path, sheet)
