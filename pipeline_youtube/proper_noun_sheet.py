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


def _proper_noun_sheet_path(video: VideoMeta, run_time: datetime, *, vault_root: Path) -> Path:
    """Path of the per-playlist proper-noun sheet (under the 01_Scripts folder).

    Every video in a playlist shares the same 01_Scripts playlist folder, so the
    sheet's parent is stable regardless of which video is used to derive it.

    ``vault_root`` is injected by the caller (``runtime.vault_root``).
    """
    scripts_path = compute_note_paths(video, run_time, units=("scripts",), vault_root=vault_root)[
        "scripts"
    ]
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

    # Sub-agent workers can all promote into the same glossary before their
    # shard starts. Serialize the read-merge-write and re-read inside the lock
    # so a later worker never overwrites corrections promoted by an earlier one.
    try:
        with _sheet_write_lock(glossary_path):
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
    except OSError:
        return 0
    return len(merged.entries) - len(base.entries)


def _sheet_write_lock(sheet_path: Path) -> contextlib.AbstractContextManager[Any]:
    """Cross-process lock guarding the shared proper-noun sheet.

    Prefer POSIX ``fcntl.flock``. A flock is owned by the open file description,
    so a forked ``--sub-agents`` worker that opens its own fd correctly blocks
    on a sibling's lock. ``filelock`` instead keeps thread-local bookkeeping that
    a fork inherits verbatim, which either misfires as a false "deadlock" or
    (with ``is_singleton``) silently skips the lock and drops a concurrent
    worker's promoted corrections. Use ``filelock`` only where ``fcntl`` is
    unavailable (Windows); fall back to no-op locking only when neither exists.
    """
    lock_path = Path(str(sheet_path) + ".lock")
    try:
        import fcntl  # noqa: F401  (availability probe; _posix_file_lock re-imports it)

        has_fcntl = True
    except ImportError:
        has_fcntl = False
    if has_fcntl:
        return _posix_file_lock(lock_path)
    try:
        import filelock  # type: ignore[import-untyped]
    except ImportError:
        return _posix_file_lock(lock_path)  # neither fcntl nor filelock → no-op
    return filelock.FileLock(str(lock_path), timeout=-1)


@contextlib.contextmanager
def _posix_file_lock(lock_path: Path) -> Any:
    try:
        import fcntl
    except ImportError:
        yield
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


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
