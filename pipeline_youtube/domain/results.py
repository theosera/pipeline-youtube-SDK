"""Domain contract: the per-video stages 01-04 run result (pure data)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .video import VideoMeta


@dataclass
class VideoRunResult:
    video: VideoMeta
    learning_md_path: Path | None = None
    learning_md_body: str | None = None
    error: str | None = None
    # Per-stage cost tracking (populated by `_process_video`).
    transcript_cost_usd: float | None = None
    transcript_model: str | None = None
    summary_cost_usd: float | None = None
    summary_model: str | None = None
    learning_cost_usd: float | None = None
    learning_model: str | None = None
    # Proper nouns Stage 01b confirmed for this video, written into the
    # per-playlist proper-noun sheet after all videos are processed.
    confirmed_terms: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.error is None and self.learning_md_body is not None
