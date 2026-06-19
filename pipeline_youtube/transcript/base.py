"""Shared transcript types and fallback chain orchestration.

The fallback hierarchy is deterministic:
  0. InnerTube (captions via the iOS YouTube client — bypasses bot/PO-token
     blocks that hit youtube-transcript-api; best-effort, optional)
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
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from ..domain.transcript import (
    TranscriptNotAvailable,
    TranscriptResult,
    TranscriptSnippet,
    TranscriptSource,
)

# Re-exported here for backward compatibility: the value types moved to
# ``domain/transcript.py`` but historically lived in this module.
__all__ = [
    "Fetcher",
    "TranscriptNotAvailable",
    "TranscriptResult",
    "TranscriptSnippet",
    "TranscriptSource",
    "build_result",
    "fetch_with_fallback",
    "result_from_cache_dict",
    "result_to_cache_dict",
]

Fetcher = Callable[[str, list[str]], TranscriptResult]


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def result_to_cache_dict(result: TranscriptResult) -> dict[str, object]:
    """Serialize a successful result for the persistent cache.

    ``fallback_reason``/``error`` are intentionally omitted — they describe
    *this* run's tier history, which is re-derived on every fetch.
    """
    return {
        "video_id": result.video_id,
        "source": str(result.source),
        "language": result.language,
        "retrieved_at": result.retrieved_at,
        "snippets": [
            {"text": s.text, "start": s.start, "duration": s.duration} for s in result.snippets
        ],
    }


def result_from_cache_dict(data: dict[str, Any]) -> TranscriptResult:
    """Rebuild a TranscriptResult from its cached form."""
    raw_snippets: list[dict[str, Any]] = data.get("snippets") or []
    snippets = [
        TranscriptSnippet(
            text=str(s["text"]), start=float(s["start"]), duration=float(s["duration"])
        )
        for s in raw_snippets
    ]
    language = data.get("language")
    return TranscriptResult(
        video_id=str(data["video_id"]),
        source=TranscriptSource(str(data["source"])),
        language=language if language is None else str(language),
        snippets=snippets,
        retrieved_at=str(data.get("retrieved_at", "")),
    )


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
    from ..cache import get_cache

    cache = get_cache()
    lang_key = languages[0] if languages else "none"

    fallback_reasons: list[str] = []
    for tier_name, fetcher in fetchers:
        if fetcher is None:
            fallback_reasons.append(f"{tier_name}:disabled")
            continue
        # Cache hit: a prior run already produced this tier's transcript.
        # Whisper (the slowest tier) benefits most. Tier ordering is
        # preserved — a missed earlier tier still re-attempts as before.
        cached = cache.get_transcript(video_id, tier_name, lang_key)
        if cached is not None:
            result = result_from_cache_dict(cached)
            joined = "; ".join(fallback_reasons) if fallback_reasons else None
            return replace(result, fallback_reason=joined)
        try:
            result = fetcher(video_id, languages)
        except TranscriptNotAvailable as e:
            fallback_reasons.append(f"{tier_name}:{e}")
            continue
        cache.put_transcript(video_id, tier_name, lang_key, result_to_cache_dict(result))
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
