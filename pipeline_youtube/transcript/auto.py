"""Tier 2: auto-generated YouTube captions via youtube-transcript-api.

Uses `TranscriptList.find_generated_transcript()` to explicitly target
YouTube's ASR output. Only reached when `official.py` raised
`TranscriptNotAvailable`.
"""

from __future__ import annotations

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
from .official import _get_api


def fetch_auto(video_id: str, languages: list[str]) -> TranscriptResult:
    """Fetch auto-generated captions for a video.

    Raises `TranscriptNotAvailable` when no auto-generated captions
    exist in any of the requested languages.
    """
    if not video_id:
        raise TranscriptNotAvailable("empty video_id")

    api = _get_api()
    try:
        transcript_list = api.list(video_id)
        transcript = transcript_list.find_generated_transcript(languages)
        fetched = transcript.fetch()
    except NoTranscriptFound as e:
        raise TranscriptNotAvailable("no_auto_transcript_in_languages") from e
    except TranscriptsDisabled as e:
        raise TranscriptNotAvailable("transcripts_disabled") from e
    except VideoUnavailable as e:
        raise TranscriptNotAvailable("video_unavailable") from e
    except IpBlocked as e:
        raise TranscriptNotAvailable("ip_blocked") from e
    except CouldNotRetrieveTranscript as e:
        raise TranscriptNotAvailable(f"retrieve_failed:{type(e).__name__}") from e
    except Exception as e:
        raise TranscriptNotAvailable(f"unexpected:{type(e).__name__}:{e}") from e

    snippets = [
        TranscriptSnippet(text=s.text, start=float(s.start), duration=float(s.duration))
        for s in fetched
    ]
    if not snippets:
        raise TranscriptNotAvailable("empty_transcript")

    return build_result(
        video_id=video_id,
        source=TranscriptSource.AUTO,
        language=transcript.language_code,
        snippets=snippets,
    )
