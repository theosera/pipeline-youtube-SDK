"""Shared transcript types and fallback chain orchestration.

The fallback hierarchy is deterministic:
  1. Official (manually-created captions via youtube-transcript-api)
  2. Auto (YouTube auto-generated captions via youtube-transcript-api)
  3. Whisper (local openai-whisper, optional dependency)

Each tier raises `TranscriptNotAvailable` on failure so the chain can
move to the next tier. The returned `TranscriptResult` always includes
a `fallback_reason` string describing why earlier tiers were skipped —
this feeds into `stats.py` for the genre-level statistics described in
the plan (decision 1).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
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
class TranscriptResult:
    video_id: str
    source: TranscriptSource
    language: str | None
    snippets: list[TranscriptSnippet] = field(default_factory=list)
    retrieved_at: str = ""
    fallback_reason: str | None = None
    error: str | None = None


Fetcher = Callable[[str, list[str]], TranscriptResult]


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def build_result(
    video_id: str,
    source: TranscriptSource,
    language: str | None,
    snippets: list[TranscriptSnippet],
    fallback_reason: str | None = None,
) -> TranscriptResult:
    """Helper so individual tier fetchers don't duplicate timestamp logic."""
    return TranscriptResult(
        video_id=video_id,
        source=source,
        language=language,
        snippets=snippets,
        retrieved_at=_iso_now(),
        fallback_reason=fallback_reason,
    )


def fetch_with_fallback(
    video_id: str,
    languages: list[str],
    fetchers: list[tuple[str, Fetcher | None]],
) -> TranscriptResult:
    """Try each tier in order; return the first successful result.

    `fetchers` is an ordered list of (tier_name, fetcher_callable). A
    `None` fetcher is skipped (used when the Whisper extra is not
    installed or disabled for cost reasons).
    """
    fallback_reasons: list[str] = []
    for tier_name, fetcher in fetchers:
        if fetcher is None:
            fallback_reasons.append(f"{tier_name}:disabled")
            continue
        try:
            result = fetcher(video_id, languages)
        except TranscriptNotAvailable as e:
            fallback_reasons.append(f"{tier_name}:{e}")
            continue
        # Success: annotate with accumulated fallback history
        joined = "; ".join(fallback_reasons) if fallback_reasons else None
        return replace(result, fallback_reason=joined)

    # All tiers failed — return an error result (no exception, so the
    # outer pipeline can log and continue to the next video)
    return TranscriptResult(
        video_id=video_id,
        source=TranscriptSource.ERROR,
        language=None,
        snippets=[],
        retrieved_at=_iso_now(),
        fallback_reason="; ".join(fallback_reasons),
        error="all transcript tiers failed",
    )
