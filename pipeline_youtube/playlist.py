"""Playlist and video metadata fetching via yt-dlp's Python API.

Uses `extract_flat='in_playlist'` so only metadata is fetched (no
actual video downloads). No YouTube Data API key is required —
yt-dlp scrapes the public playlist page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import yt_dlp  # type: ignore[import-untyped]

_ALLOWED_HOSTS = frozenset(
    {
        "www.youtube.com",
        "youtube.com",
        "m.youtube.com",
        "youtu.be",
    }
)

# Max URL length. YouTube URLs with playlist + timestamp params fit in
# well under 500 chars; anything longer is almost certainly attacker
# padding or an accidental paste of an entirely different payload.
_MAX_URL_LEN = 500

# Per-host path whitelist. `youtu.be` uses short form `/<VIDEO_ID>`;
# canonical hosts use `/watch`, `/playlist`, `/shorts/...`, `/live/...`,
# or a bare `/` when all parameters live in the query string.
_VIDEO_ID_RE = re.compile(r"^/[A-Za-z0-9_-]{6,20}/?$")
_CANONICAL_PATHS = frozenset({"", "/", "/watch", "/watch/", "/playlist", "/playlist/"})
_CANONICAL_PREFIXES = ("/shorts/", "/live/", "/embed/")


def _validate_path(hostname: str, path: str) -> None:
    if hostname == "youtu.be":
        if path in ("", "/"):
            return
        if _VIDEO_ID_RE.match(path):
            return
        raise ValueError(f"youtu.be path must be `/VIDEO_ID`: {path!r}")
    # canonical hosts (www/m youtube.com, youtube.com)
    if path in _CANONICAL_PATHS:
        return
    if any(path.startswith(prefix) for prefix in _CANONICAL_PREFIXES):
        return
    raise ValueError(
        f"URL path must be one of /watch, /playlist, /shorts/..., /live/..., /embed/...; got {path!r}"
    )


def validate_youtube_url(url: str) -> str:
    """Validate a URL as a YouTube playlist or video URL.

    Guards yt-dlp from file://, http://internal-host, arbitrary
    third-party extractors, and malformed YouTube-look-alike URLs.
    Defence layers:

      1. Type + non-empty check
      2. Length cap (reject unusually long payloads)
      3. Scheme whitelist (http / https only)
      4. Hostname whitelist (YouTube domains only)
      5. Path pattern per-host (reject unknown surface like `/api/*`)

    Returns the URL on success; raises ValueError otherwise.
    """
    if not isinstance(url, str) or not url:
        raise ValueError("URL is empty or not a string")
    if len(url) > _MAX_URL_LEN:
        raise ValueError(f"URL exceeds {_MAX_URL_LEN} chars: {len(url)}")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"URL scheme must be http/https: {parsed.scheme!r}")
    if parsed.hostname not in _ALLOWED_HOSTS:
        raise ValueError(f"URL host must be a YouTube domain: {parsed.hostname!r}")
    _validate_path(parsed.hostname, parsed.path)
    return url


@dataclass(frozen=True)
class VideoMeta:
    video_id: str
    title: str
    url: str
    duration: int | None  # seconds, may be None on flat-playlist mode
    channel: str | None
    upload_date: str | None  # YYYYMMDD format from yt-dlp
    playlist_title: str | None

    @property
    def watch_url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"

    @property
    def timestamp_url(self) -> str:
        """Base URL suitable for appending &t=<seconds>."""
        return self.watch_url


_BASE_OPTS: dict[str, Any] = {
    "quiet": True,
    "no_warnings": True,
    "skip_download": True,
}


def fetch_metadata(url: str) -> list[VideoMeta]:
    """Fetch metadata for a playlist URL or a single-video URL.

    Returns a list of VideoMeta (single entry for a video URL, multiple
    for a playlist). On a playlist URL, the playlist title is propagated
    to every VideoMeta.playlist_title for downstream folder naming.
    """
    validate_youtube_url(url)
    opts = {**_BASE_OPTS, "extract_flat": "in_playlist"}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info: dict[str, Any] = ydl.extract_info(url, download=False)  # type: ignore[assignment]

    if info is None:
        return []

    playlist_title: str | None = None
    if info.get("_type") == "playlist":
        playlist_title = info.get("title")

    entries = info.get("entries")
    if entries is None:
        entries = [info]

    videos: list[VideoMeta] = []
    for entry in entries:
        if entry is None:
            continue
        video_id = entry.get("id") or ""
        if not video_id:
            continue
        videos.append(
            VideoMeta(
                video_id=video_id,
                title=entry.get("title") or "",
                url=entry.get("url") or f"https://www.youtube.com/watch?v={video_id}",
                duration=entry.get("duration"),
                channel=entry.get("channel") or entry.get("uploader"),
                upload_date=entry.get("upload_date"),
                playlist_title=playlist_title,
            )
        )
    return videos
