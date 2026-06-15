"""Build pipeline inputs from a local folder of video files (fully offline).

Lets stages 01-05 run with **no YouTube access**: instead of fetching a
playlist over the network, enumerate media files in a directory and synthesize
the `VideoMeta` the pipeline needs. Stage 01 transcribes each file locally with
Whisper; Stage 03 captures frames from it directly (no download). Pair with
``main.py --local-media DIR``.

Filename → video_id: yt-dlp's two common shapes are recognized — bare
``<id>`` and ``Title [<id>]`` (the 11-char YouTube id). When a file has no
id-shaped token, a deterministic 11-char id is synthesized from the filename so
it still satisfies the canonical id format the checkpoint/frontmatter layer
validates against (``[A-Za-z0-9_-]{11}``).
"""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path

from .playlist import VideoMeta

# Container formats ffmpeg/Whisper can decode directly (audio is extracted
# internally, so the full video file is fine as a transcription source).
MEDIA_EXTENSIONS = frozenset({".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi"})

# A YouTube id is exactly 11 chars of [A-Za-z0-9_-]. Match either a stem that
# *is* the id, or a bracketed `[id]` suffix (yt-dlp `%(title)s [%(id)s]`).
# The bracket form is anchored to the end of the stem so an id-shaped token
# *inside* a title (e.g. "01 [abcdefghijk] Talk [realId11chr]") can't be
# mistaken for the id — yt-dlp always puts the real id last.
_BARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_BRACKET_ID_RE = re.compile(r"\[([A-Za-z0-9_-]{11})\]\s*$")


def extract_video_id(stem: str) -> str | None:
    """Return the YouTube 11-char id embedded in a filename stem, or None.

    Only the two unambiguous yt-dlp shapes are accepted (bare id, or a
    bracketed ``[id]``) to avoid matching a random 11-char run inside a title.
    """
    if _BARE_ID_RE.match(stem):
        return stem
    match = _BRACKET_ID_RE.search(stem)
    return match.group(1) if match else None


def synthesize_video_id(name: str) -> str:
    """Derive a stable canonical-format id from a filename (for non-yt-dlp files).

    base64-urlsafe of the name's SHA-256 yields only ``[A-Za-z0-9_-]`` (plus
    trailing ``=`` padding, which never lands in the first 11 chars), so the
    11-char slice always matches ``[A-Za-z0-9_-]{11}`` and the checkpoint layer
    treats it as a trusted id. Deterministic, so re-runs reuse the same id.

    ``name`` is hashed verbatim, so callers that want two identically-named
    files in *different* folders to get distinct ids must qualify it with the
    containing folder's resolved path (see ``build_local_videos``); hashing
    ``path.name`` — or even ``folder_basename/path.name`` — alone would collide
    across same-named folders and silently reuse another folder's transcript
    cache / checkpoint.
    """
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii")[:11]


def title_from_filename(stem: str, fallback: str) -> str:
    """Human title from a filename stem: drop a bracketed ``[id]`` and trim.

    Falls back to ``fallback`` (the id) when nothing readable remains, e.g. a
    file literally named ``<id>.mp4``.
    """
    # A bare-id stem has no human title — return the id directly. Trimming it
    # would mangle ids that legitimately start/end with '-' or '_'.
    if _BARE_ID_RE.fullmatch(stem):
        return fallback
    cleaned = _BRACKET_ID_RE.sub("", stem).strip(" -_·–—").strip()
    return cleaned or fallback


def build_local_videos(media_dir: Path) -> tuple[list[VideoMeta], dict[str, Path]]:
    """Enumerate ``media_dir`` and return ``(videos, video_id -> path)``.

    Files are taken in sorted filename order (so a ``NN_`` / playlist-index
    prefix preserves playlist order). The directory name becomes
    ``playlist_title`` for output-folder naming. Duplicate ids keep the first
    file. Raises ``ValueError`` if the path is not a directory.
    """
    if not media_dir.is_dir():
        raise ValueError(f"--local-media is not a directory: {media_dir}")

    playlist_title = media_dir.name
    files = sorted(
        p for p in media_dir.iterdir() if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS
    )

    videos: list[VideoMeta] = []
    media_map: dict[str, Path] = {}
    media_dir_key = str(media_dir.resolve())
    for path in files:
        # Qualify the synthesized id with the *resolved* media-dir path (not just
        # its basename) so two identically-named files under different folders —
        # even folders that share a basename, e.g. /a/media/x.mp4 vs /b/media/
        # x.mp4 — don't collide and cross-wire their transcript cache /
        # checkpoints. yt-dlp-style ids are already globally unique, so only the
        # synthesized branch needs this. Re-runs from the same path stay stable.
        video_id = extract_video_id(path.stem) or synthesize_video_id(
            f"{media_dir_key}/{path.name}"
        )
        if video_id in media_map:
            continue
        videos.append(
            VideoMeta(
                video_id=video_id,
                title=title_from_filename(path.stem, video_id),
                url=f"https://www.youtube.com/watch?v={video_id}",
                duration=None,
                channel=None,
                upload_date=None,
                playlist_title=playlist_title,
            )
        )
        media_map[video_id] = path
    return videos, media_map
