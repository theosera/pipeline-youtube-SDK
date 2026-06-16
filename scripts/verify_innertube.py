#!/usr/bin/env python3
"""Field-verify the Tier-0 InnerTube caption fetch on a real video.

Run this on the machine that will actually do the fetching (a residential IP —
the same place the YTranscript Obsidian plugin works). It exercises the real
``fetch_innertube`` end to end: the iOS-client player POST, track selection,
the timedtext XML GET, and parsing — printing diagnostics at each step.

Usage:
    uv run python scripts/verify_innertube.py <video_id_or_url> [lang ...]

Examples:
    uv run python scripts/verify_innertube.py dQw4w9WgXcQ ja en
    uv run python scripts/verify_innertube.py https://youtu.be/dQw4w9WgXcQ

Exit code 0 on a successful fetch, 1 otherwise. This is a diagnostic tool, not
part of the pipeline; it only reads captions and prints them.
"""

from __future__ import annotations

import re
import sys

from pipeline_youtube.transcript.base import TranscriptNotAvailable
from pipeline_youtube.transcript.innertube import (
    _default_fetch_player_json,
    _extract_caption_tracks,
    _select_track,
    fetch_innertube,
)

_ID_RE = re.compile(r"(?:v=|/shorts/|/live/|youtu\.be/|/embed/|/v/)([A-Za-z0-9_-]{11})")


def _video_id(arg: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", arg):
        return arg
    m = _ID_RE.search(arg)
    if not m:
        sys.exit(f"could not extract a video id from: {arg!r}")
    return m.group(1)


def main(argv: list[str]) -> int:
    if not argv:
        sys.exit("usage: verify_innertube.py <video_id_or_url> [lang ...]")
    video_id = _video_id(argv[0])
    languages = argv[1:] or ["ja", "en"]
    print(f"video_id={video_id}  languages={languages}")

    # Step 1: player POST (shows the full track list before selection).
    try:
        payload = _default_fetch_player_json(video_id, languages, 20.0)
    except TranscriptNotAvailable as e:
        print(f"[player] FAILED: {e}")
        print("  → tier 0 would fall through to youtube-transcript-api, then Whisper.")
        return 1
    tracks = _extract_caption_tracks(payload)
    print(f"[player] OK — {len(tracks)} caption track(s):")
    for t in tracks:
        kind = "asr" if t.get("kind") == "asr" else "manual"
        print(f"    {t.get('languageCode'):<8} {kind}")
    if not tracks:
        print("  → no captions on this video; tier 0 yields to Whisper.")
        return 1
    chosen = _select_track(tracks, languages)
    if chosen is not None:
        kind = "asr" if chosen.get("kind") == "asr" else "manual"
        print(f"[select] -> {chosen.get('languageCode')} ({kind})")

    # Step 2: full fetch (timedtext GET + parse).
    try:
        result = fetch_innertube(video_id, languages)
    except TranscriptNotAvailable as e:
        print(f"[fetch] FAILED at timedtext/parse: {e}")
        return 1

    print(
        f"[fetch] OK — source={result.source} language={result.language} "
        f"snippets={len(result.snippets)}"
    )
    for s in result.snippets[:5]:
        mm, ss = divmod(int(s.start), 60)
        print(f"    [{mm:02d}:{ss:02d}] {s.text}")
    print("RESULT: tier-0 InnerTube works on this machine ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
