"""Tier 0: fast caption fetch via YouTube's private InnerTube player API.

Why this exists: ``youtube-transcript-api`` (tiers 1/2) is increasingly met with
bot challenges / PO-token requirements, which pushes runs onto the slow Whisper
fallback (the dominant cost — see ``docs/parallelization-effectiveness.md``).
Posing as the **iOS YouTube client** sidesteps that: as of early 2026 the iOS
InnerTube client returns caption-track URLs *without* a PO token (the trick the
``obsidian-yt-transcript`` plugin relies on). Two HTTP round-trips — POST
``youtubei/v1/player`` to get ``captionTracks[]``, then GET the chosen track's
timedtext XML — and a regex parse. No audio download, no inference.

Best-effort by construction: any failure raises ``TranscriptNotAvailable`` so
the existing ``youtube-transcript-api`` and Whisper tiers still run. This tier
never moves a timestamp or changes the ``TranscriptResult`` contract.

Brittleness / scope (read before relying on it): InnerTube is undocumented and
YouTube can change it; the embedded API key and client version below are public
iOS-app values that may need bumping over time. It is meant for **residential
IP, low request volume** (mirroring the plugin running inside Obsidian). On
datacenter/cloud IPs or at high request rates YouTube returns 403/bot
challenges, so this stays *behind the fallback chain*, never a sole dependency.
"""

from __future__ import annotations

import html
import random
import re
import time
from collections.abc import Callable
from typing import Any

from .base import (
    TranscriptNotAvailable,
    TranscriptResult,
    TranscriptSnippet,
    TranscriptSource,
    build_result,
)

# Public iOS-app InnerTube key + client identity. These are not secrets (they
# ship in the iOS app); they are here so the request looks like the iOS client,
# which is the part that avoids PO tokens. Bump CLIENT_VERSION if YouTube starts
# rejecting it.
INNERTUBE_API_KEY = "AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8"
CLIENT_NAME = "IOS"
CLIENT_VERSION = "20.10.38"
IOS_USER_AGENT = "com.google.ios.youtube/20.10.38 (iPhone16,2; U; CPU iOS 17_5_1 like Mac OS X)"

# Player API hosts, tried in order. The googleapis host is the canonical
# InnerTube endpoint and tends to serve fewer bot interstitials than the
# www.youtube.com web host; the web host is kept as a fallback. (The caption
# track ``baseUrl`` YouTube returns always points at www.youtube.com/api/
# timedtext regardless of which host answered the player call.)
PLAYER_HOSTS = (
    "https://youtubei.googleapis.com/youtubei/v1/player",
    "https://www.youtube.com/youtubei/v1/player",
)

DEFAULT_TIMEOUT = 20.0

# The timedtext endpoint rate-limits per IP: niche (non-CDN-cached) videos hit
# 429 easily, especially in bursts. Retry these transient statuses with
# exponential backoff + jitter, honoring Retry-After. Keep request volume low
# (low --concurrency / paced warm-up) to avoid tripping it in the first place.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 1.0

# Minimal country hint per UI language so the player call looks plausible. An
# unmapped language just defaults to a US gl — YouTube does not require a match.
_GL_BY_LANG = {"ja": "JP", "en": "US", "ko": "KR", "zh": "CN", "fr": "FR", "de": "DE"}

# Two timedtext shapes: the classic ``<text start dur>`` (seconds) and the srv3
# ``<p t d>`` (milliseconds). We try the first, then the second.
_TEXT_RE = re.compile(r'<text\s+start="([\d.]+)"\s+dur="([\d.]+)"[^>]*>(.*?)</text>', re.DOTALL)
_P_RE = re.compile(r'<p\s+t="(\d+)"\s+d="(\d+)"[^>]*>(.*?)</p>', re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# Injection seams so tests exercise selection/parsing without touching the
# network. Defaults below do the real httpx calls.
PlayerJsonFetcher = Callable[[str, list[str], float], dict[str, Any]]
TrackTextFetcher = Callable[[str, float], str]


def _gl_for(languages: list[str]) -> str:
    return _GL_BY_LANG.get(languages[0].lower()[:2], "US") if languages else "US"


def _default_fetch_player_json(
    video_id: str, languages: list[str], timeout: float
) -> dict[str, Any]:
    """POST the iOS InnerTube ``player`` request and return the parsed JSON."""
    import httpx

    hl = languages[0] if languages else "en"
    body = {
        "context": {
            "client": {
                "clientName": CLIENT_NAME,
                "clientVersion": CLIENT_VERSION,
                "hl": hl,
                "gl": _gl_for(languages),
                "deviceModel": "iPhone16,2",
            }
        },
        "videoId": video_id,
    }
    headers = {"Content-Type": "application/json", "User-Agent": IOS_USER_AGENT}
    last_reason = "innertube_no_host"
    for host in PLAYER_HOSTS:
        try:
            resp = httpx.post(
                f"{host}?key={INNERTUBE_API_KEY}", json=body, headers=headers, timeout=timeout
            )
        except httpx.HTTPError as e:
            last_reason = f"innertube_post_failed:{type(e).__name__}"
            continue
        if resp.status_code != 200:
            last_reason = f"innertube_http_{resp.status_code}"
            continue
        try:
            payload = resp.json()
        except ValueError:
            last_reason = "innertube_bad_json"
            continue
        if not isinstance(payload, dict):
            last_reason = "innertube_unexpected_payload"
            continue
        return payload
    raise TranscriptNotAvailable(last_reason)


def _retry_after_seconds(headers: Any) -> float | None:
    """Parse a numeric ``Retry-After`` header (seconds), if present and valid."""
    value = headers.get("Retry-After")
    if isinstance(value, str) and value.isdigit():
        return float(value)
    return None


def _default_fetch_track_text(
    url: str, timeout: float, *, max_retries: int = DEFAULT_MAX_RETRIES
) -> str:
    """GET a caption track's timedtext XML, retrying transient 429/5xx.

    The timedtext endpoint throttles per IP; a 429 here is usually transient.
    Back off exponentially with jitter (honoring ``Retry-After``) before giving
    up so the chain falls through to youtube-transcript-api / Whisper.
    """
    import httpx

    for attempt in range(max_retries + 1):
        try:
            resp = httpx.get(url, headers={"User-Agent": IOS_USER_AGENT}, timeout=timeout)
        except httpx.HTTPError as e:
            raise TranscriptNotAvailable(f"innertube_track_failed:{type(e).__name__}") from e
        if resp.status_code == 200:
            return resp.text
        if resp.status_code in RETRYABLE_STATUS and attempt < max_retries:
            base = DEFAULT_BACKOFF_BASE * (2**attempt)
            time.sleep((_retry_after_seconds(resp.headers) or base) + random.uniform(0, 0.5))
            continue
        raise TranscriptNotAvailable(f"innertube_track_http_{resp.status_code}")
    raise TranscriptNotAvailable("innertube_track_retries_exhausted")


def _extract_caption_tracks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull ``captions.playerCaptionsTracklistRenderer.captionTracks`` safely."""
    captions = payload.get("captions")
    if not isinstance(captions, dict):
        return []
    renderer = captions.get("playerCaptionsTracklistRenderer")
    if not isinstance(renderer, dict):
        return []
    tracks = renderer.get("captionTracks")
    if not isinstance(tracks, list):
        return []
    return [t for t in tracks if isinstance(t, dict)]


def _lang_of(track: dict[str, Any]) -> str:
    code = track.get("languageCode")
    return code.lower() if isinstance(code, str) else ""


def _select_track(tracks: list[dict[str, Any]], languages: list[str]) -> dict[str, Any] | None:
    """Pick the best caption track: language order, manual preferred over ASR.

    For each requested language (in order) try exact, then prefix
    (``en`` matches ``en-US``), then reverse-prefix (``en-US`` matches ``en``).
    Within a language a manually-created track wins over an ``asr`` one — this
    preserves the existing "manual > auto" tier preference.

    Language is **strict** when ``languages`` is non-empty: if none of the
    requested languages matches, return ``None`` so the chain falls through to
    the youtube-transcript-api tiers (also language-strict) and then Whisper,
    rather than silently transcribing an unrequested language. Only when no
    language preference is given do we fall back to the first available track.
    """

    def manual_first(candidates: list[dict[str, Any]]) -> dict[str, Any]:
        manual = [t for t in candidates if t.get("kind") != "asr"]
        return (manual or candidates)[0]

    for lang in languages or []:
        code = lang.lower()
        exact = [t for t in tracks if _lang_of(t) == code]
        if exact:
            return manual_first(exact)
        prefix = [t for t in tracks if _lang_of(t).startswith(code + "-")]
        if prefix:
            return manual_first(prefix)
        reverse = [t for t in tracks if _lang_of(t) and code.startswith(_lang_of(t) + "-")]
        if reverse:
            return manual_first(reverse)
    if languages:
        return None  # requested languages unavailable — defer to later tiers
    return tracks[0] if tracks else None


def _clean_text(raw: str) -> str:
    """Strip inner tags, unescape entities, collapse whitespace."""
    return _WS_RE.sub(" ", html.unescape(_TAG_RE.sub("", raw))).strip()


def _parse_timedtext(xml: str) -> list[TranscriptSnippet]:
    """Parse timedtext XML into snippets, handling both ``<text>`` and ``<p>``."""
    snippets: list[TranscriptSnippet] = []
    text_matches = list(_TEXT_RE.finditer(xml))
    if text_matches:
        for m in text_matches:
            text = _clean_text(m.group(3))
            if text:
                snippets.append(
                    TranscriptSnippet(
                        text=text, start=float(m.group(1)), duration=float(m.group(2))
                    )
                )
        return snippets
    for m in _P_RE.finditer(xml):
        text = _clean_text(m.group(3))
        if text:
            snippets.append(
                TranscriptSnippet(
                    text=text,
                    start=int(m.group(1)) / 1000.0,
                    duration=int(m.group(2)) / 1000.0,
                )
            )
    return snippets


def fetch_innertube(
    video_id: str,
    languages: list[str],
    *,
    timeout: float = DEFAULT_TIMEOUT,
    fetch_player_json: PlayerJsonFetcher = _default_fetch_player_json,
    fetch_track_text: TrackTextFetcher = _default_fetch_track_text,
) -> TranscriptResult:
    """Tier 0 fetcher: caption via the iOS InnerTube player API.

    Raises ``TranscriptNotAvailable`` (caught by ``fetch_with_fallback``) on any
    failure — unreachable host, bot block, no captions, unparsable XML — so the
    youtube-transcript-api and Whisper tiers still get their turn. A
    manually-created track maps to ``OFFICIAL``; an ``asr`` track to ``AUTO``,
    keeping the existing source semantics that ``stats.py`` records.
    """
    if not video_id:
        raise TranscriptNotAvailable("empty video_id")

    payload = fetch_player_json(video_id, languages, timeout)

    status = payload.get("playabilityStatus")
    if isinstance(status, dict):
        state = status.get("status")
        if isinstance(state, str) and state != "OK":
            raise TranscriptNotAvailable(f"playability:{state}")

    tracks = _extract_caption_tracks(payload)
    if not tracks:
        raise TranscriptNotAvailable("no_caption_tracks")

    track = _select_track(tracks, languages)
    if track is None:
        raise TranscriptNotAvailable("no_matching_track")
    base_url = track.get("baseUrl")
    if not isinstance(base_url, str) or not base_url:
        raise TranscriptNotAvailable("track_missing_baseurl")

    xml = fetch_track_text(base_url, timeout)
    snippets = _parse_timedtext(xml)
    if not snippets:
        raise TranscriptNotAvailable("empty_transcript")

    source = TranscriptSource.AUTO if track.get("kind") == "asr" else TranscriptSource.OFFICIAL
    return build_result(
        video_id=video_id,
        source=source,
        language=track.get("languageCode"),
        snippets=snippets,
    )
