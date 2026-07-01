"""Domain contracts: transcript value types (pure, provider-agnostic).

The fallback-chain orchestration (``build_result`` / ``fetch_with_fallback``)
lives in ``transcript/base.py``; only the data/contract types live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class TranscriptSource(StrEnum):
    OFFICIAL = "official"
    AUTO = "auto-generated"
    WHISPER = "whisper"
    ERROR = "error"


class TranscriptNotAvailable(Exception):
    """Raised by a tier fetcher when it cannot produce a transcript.

    The fallback chain catches this and moves to the next tier. The
    message is recorded in `fallback_reason` for observability.
    """


@dataclass(frozen=True)
class TranscriptSnippet:
    """A single transcript segment.

    Mirrors youtube-transcript-api's FetchedTranscriptSnippet but stays
    provider-agnostic so the chunker/stage code doesn't depend on the
    library shape.
    """

    text: str
    start: float  # seconds
    duration: float  # seconds

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass(frozen=True)
class VideoChapter:
    """One YouTube-declared chapter, from the video's own metadata."""

    title: str
    start_seconds: float

    @property
    def mmss(self) -> str:
        total = int(self.start_seconds)
        mm, ss = divmod(total, 60)
        return f"{mm:02d}:{ss:02d}"


@dataclass(frozen=True)
class TranscriptResult:
    video_id: str
    source: TranscriptSource
    language: str | None
    snippets: list[TranscriptSnippet] = field(default_factory=list)
    retrieved_at: str = ""
    fallback_reason: str | None = None
    error: str | None = None
    # Billed cost of the Stage 01b LLM correction pass, if it ran. ``None`` when
    # no correction was attempted (Stage 01a only); a float (possibly 0.0) once
    # correction ran so Stage 01 can echo ``cost=$...`` like Stage 02/04.
    correction_cost_usd: float | None = None
    # Proper nouns Stage 01b confirmed for this video (deduped). Written into the
    # per-playlist proper-noun sheet for human review and next-run reuse.
    confirmed_terms: tuple[str, ...] = ()
    # Video description + declared chapters (Stage 01a, a single best-effort
    # yt-dlp extract — ``None``/``()`` when skipped under ``--local-media`` or
    # the fetch failed). Feeds Stage 01b's known-context block (fewer web
    # searches) and Stage 02's content-mode diagnosis.
    description: str | None = None
    chapters: tuple[VideoChapter, ...] = ()
