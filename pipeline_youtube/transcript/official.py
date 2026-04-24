"""Tier 1: manually-created YouTube captions via youtube-transcript-api.

Uses `TranscriptList.find_manually_created_transcript()` so auto-generated
captions are explicitly excluded from this tier — they're handled by
`auto.py` as tier 2. This matches decision (1) in the plan.
"""

from __future__ import annotations

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    IpBlocked,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from .base import (
    TranscriptNotAvailable,
    TranscriptResult,
    TranscriptSnippet,
    TranscriptSource,
    build_result,
)

# Module-level cache so repeated calls in a run share the HTTP session.
_api: YouTubeTranscriptApi | None = None


def _get_api() -> YouTubeTranscriptApi:
    global _api
    if _api is None:
        _api = YouTubeTranscriptApi()
    return _api


def fetch_official(video_id: str, languages: list[str]) -> TranscriptResult:
    """Fetch manually-created captions for a video.

    Raises `TranscriptNotAvailable` with a short reason string when the
    video has no manually-created captions in any of the requested
    languages. Re-raises nothing else — IpBlocked and other fatal
    conditions are also wrapped so the fallback chain can move on.
    """
    if not video_id:
        raise TranscriptNotAvailable("empty video_id")

    api = _get_api()
    try:
        transcript_list = api.list(video_id)
        transcript = transcript_list.find_manually_created_transcript(languages)
        fetched = transcript.fetch()
    except NoTranscriptFound as e:
        raise TranscriptNotAvailable("no_manual_transcript_in_languages") from e
    except TranscriptsDisabled as e:
        raise TranscriptNotAvailable("transcripts_disabled") from e
    except VideoUnavailable as e:
        raise TranscriptNotAvailable("video_unavailable") from e
    except IpBlocked as e:
        raise TranscriptNotAvailable("ip_blocked") from e
    except CouldNotRetrieveTranscript as e:
        raise TranscriptNotAvailable(f"retrieve_failed:{type(e).__name__}") from e
    except Exception as e:  # catch-all so the chain continues
        raise TranscriptNotAvailable(f"unexpected:{type(e).__name__}:{e}") from e

    snippets = [
        TranscriptSnippet(text=s.text, start=float(s.start), duration=float(s.duration))
        for s in fetched
    ]
    if not snippets:
        raise TranscriptNotAvailable("empty_transcript")

    return build_result(
        video_id=video_id,
        source=TranscriptSource.OFFICIAL,
        language=transcript.language_code,
        snippets=snippets,
    )
