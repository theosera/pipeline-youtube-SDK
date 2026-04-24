"""Video-level checkpoint: skip already-processed videos.

Checks whether a stage 04 md file with a matching `video_id` frontmatter
field exists in the expected playlist folder. If so, the video can be
skipped entirely (stages 01-04 all write to the same playlist folder,
and 04 is the last to complete, so its presence implies 01-03 are also
done).

Design note (ミノ駆動本 ch8 単一責任):
    This module does ONE thing — answer "is this video already done?"
    It does not decide what to do about it; that's the caller's job.

Trust model (M3 hardening)
--------------------------
The vault directory is treated as semi-trusted: a user with write access
to the vault could in principle plant a crafted md file to make the
pipeline skip real videos (self-DoS). `extract_trusted_video_id` applies
three defensive layers before accepting a checkpoint marker:

  1. Requires a well-formed YAML frontmatter block (`---...---`) at the
     top of the file. `video_id:` anywhere else is ignored.
  2. Validates the video_id matches YouTube's canonical format
     (11 chars from `[A-Za-z0-9_-]`). Arbitrary strings, partial IDs,
     and path-like values are rejected.
  3. If a `URL:` field is also present in the frontmatter, enforces
     that it references the same video_id (integrity cross-check).

On failure the file is silently skipped — same safe fallback as
"no matching file found".
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .config import get_vault_root
from .obsidian import format_playlist_folder_name, sanitize_title_for_filename
from .path_safety import ensure_safe_path
from .pipeline import LEARNING_BASE, LEGACY_LEARNING_DIR, UNIT_DIRS

# YouTube video IDs are always 11 chars from [A-Za-z0-9_-].
_YT_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Match a YAML frontmatter block that opens the file: `---\n...\n---`.
_FRONTMATTER_BLOCK_RE = re.compile(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)

# Inside-frontmatter field lines.
_VIDEO_ID_LINE_RE = re.compile(r'^video_id:\s*"([^"\n]+)"\s*$', re.MULTILINE)
_URL_LINE_RE = re.compile(r'^URL:\s*"([^"\n]+)"\s*$', re.MULTILINE)

# Cap the frontmatter scan — long titles/playlists still fit comfortably.
_FRONTMATTER_SCAN_BYTES = 2048


def extract_trusted_video_id(md_bytes: bytes) -> str | None:
    """Extract a validated video_id from a pipeline-produced md file.

    Returns the 11-char YouTube video_id on success, else `None`. See
    module docstring for the three defense layers this enforces.
    Never raises — any I/O or parse failure returns `None`.
    """
    try:
        head = md_bytes[:_FRONTMATTER_SCAN_BYTES].decode("utf-8", errors="replace")
    except Exception:
        return None

    fm_match = _FRONTMATTER_BLOCK_RE.match(head)
    if fm_match is None:
        return None
    block = fm_match.group(1)

    vid_match = _VIDEO_ID_LINE_RE.search(block)
    if vid_match is None:
        return None
    video_id = vid_match.group(1)

    if _YT_VIDEO_ID_RE.match(video_id) is None:
        return None

    # Integrity cross-check: if URL is present, it must embed the same video_id.
    url_match = _URL_LINE_RE.search(block)
    if url_match is not None:
        url = url_match.group(1)
        if f"v={video_id}" not in url and f"/{video_id}" not in url:
            return None

    return video_id


def read_trusted_video_id(md_path: Path) -> str | None:
    """Read an md file and return its validated video_id, or None on any failure."""
    try:
        data = md_path.read_bytes()
    except OSError:
        return None
    return extract_trusted_video_id(data)


def _find_learning_folder(playlist_title: str, run_date: datetime) -> Path | None:
    """Locate the 04_Learning_Material playlist folder for a given date.

    Tries the canonical name first (`YYYY-MM-DD-HHmm <title>`), then
    falls back to any folder starting with today's date prefix and
    containing the sanitized playlist title. Returns None if nothing
    matches.

    Historical `04_Lerning_Material` (typo) folders are also searched
    so existing vaults continue to work without renaming. See
    `pipeline.LEGACY_LEARNING_DIR`.
    """
    vault_root = get_vault_root()
    bases = [
        vault_root / ensure_safe_path(f"{LEARNING_BASE}/{UNIT_DIRS['learning']}"),
        vault_root / ensure_safe_path(f"{LEARNING_BASE}/{LEGACY_LEARNING_DIR}"),
    ]
    bases = [b for b in bases if b.exists()]
    if not bases:
        return None
    base = bases[0]

    # Canonical name
    canonical = base / format_playlist_folder_name(run_date, playlist_title)
    if canonical.exists():
        return canonical

    # Fallback: date prefix + title substring (handles legacy folder names)
    date_prefix = run_date.strftime("%Y-%m-%d")
    title_needle = sanitize_title_for_filename(playlist_title)
    if not title_needle:
        return None

    # Also handle `/`-separated playlist titles (take last segment)
    from .obsidian import _strip_playlist_category_prefix

    display_title = _strip_playlist_category_prefix(playlist_title)
    title_needle = sanitize_title_for_filename(display_title)

    for b in bases:
        for child in b.iterdir():
            if child.is_dir() and child.name.startswith(date_prefix) and title_needle in child.name:
                return child
    return None


def is_video_complete(
    video_id: str,
    playlist_title: str,
    run_date: datetime,
) -> bool:
    """Return True if a stage 04 md with matching video_id already exists.

    Scans the 04_Learning_Material playlist folder for any .md file whose
    YAML frontmatter contains `video_id: "<video_id>"`.
    """
    folder = _find_learning_folder(playlist_title, run_date)
    if folder is None or not folder.exists():
        return False

    return any(read_trusted_video_id(md) == video_id for md in folder.glob("*.md"))


def get_completed_video_ids(
    playlist_title: str,
    run_date: datetime,
) -> set[str]:
    """Return the set of video_ids that have completed stage 04.

    Useful for batch skip decisions without calling is_video_complete
    in a loop (one folder scan instead of N).
    """
    folder = _find_learning_folder(playlist_title, run_date)
    if folder is None or not folder.exists():
        return set()

    ids: set[str] = set()
    for md in folder.glob("*.md"):
        vid = read_trusted_video_id(md)
        if vid is not None:
            ids.add(vid)
    return ids
