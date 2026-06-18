"""Domain entity: a single video's metadata (pure contract, no behavior deps)."""

from __future__ import annotations

from dataclasses import dataclass


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
