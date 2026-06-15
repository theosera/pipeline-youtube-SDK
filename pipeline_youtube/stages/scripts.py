"""Stage 01: timestamped transcript → N-second chunks → markdown.

Output format matches the dummy data in
`Permanent Note/08_YouTube学習/01_Scripts_Processing_Unit/`:

    [MM:SS](https://www.youtube.com/watch?v=<id>&t=<seconds>) chunk text...
    [MM:SS](https://www.youtube.com/watch?v=<id>&t=<seconds>) chunk text...

The frontmatter above the body is already written by the placeholder
step (`pipeline.create_placeholder_notes`), so this stage appends the
chunked body to the existing file.

When ``include_code_blocks=True`` is passed (set by the orchestrator
when the Router classifies the playlist as ``coding``), this stage
additionally fetches the video description, scrapes any GitHub
blob/Gist URLs, downloads their raw content (size-capped), and appends
a ``## 関連コード`` section after the transcript.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

from ..code_fetch import (
    extract_github_urls,
    fetch_snippets_for_urls,
    fetch_video_description,
    render_code_section,
)
from ..playlist import VideoMeta
from ..transcript.auto import fetch_auto
from ..transcript.base import Fetcher, TranscriptResult, fetch_with_fallback
from ..transcript.chunking import Chunk, chunk_by_window
from ..transcript.official import fetch_official

DEFAULT_LANGUAGES: list[str] = ["ja", "en"]

# Default fan-out for the upfront transcript warm-up. Higher than the
# pipeline's --concurrency because caption fetches are light network I/O,
# not CPU/GPU/LLM bound.
DEFAULT_TRANSCRIPT_CONCURRENCY = 8


def warm_transcript_cache(
    videos: list[VideoMeta],
    *,
    languages: list[str] | None = None,
    concurrency: int = DEFAULT_TRANSCRIPT_CONCURRENCY,
) -> int:
    """Pre-fetch caption transcripts for many videos concurrently.

    Phase 3 (C). Stage 01 fetches transcripts one video at a time, gated by
    the pipeline's (intentionally low) ``--concurrency`` because the heavy
    downstream stages share that budget. Caption fetches, though, are cheap
    network I/O and can run at a much higher fan-out. Warming the persistent
    transcript cache up front lets those fetches overlap maximally; the
    per-video Stage 01 then hits the cache instead of re-fetching serially.

    Only the official/auto tiers are warmed — Whisper is deliberately
    excluded because it is GPU/RAM-bound and governed by its own semaphore.
    Videos that have no captions simply miss here and fall through to
    Whisper later in Stage 01, exactly as before. Best-effort: any per-video
    error is swallowed so warming never blocks the run.

    Returns the number of videos for which a caption tier was cached.
    """
    if not videos:
        return 0

    from ..cache import get_cache

    # Without a persistent cache there is nothing to warm — Stage 01 would
    # re-fetch regardless, so skip the extra pass entirely.
    if not get_cache().enabled:
        return 0

    langs = languages or DEFAULT_LANGUAGES
    bound = max(1, concurrency)

    def _warm_one(video: VideoMeta) -> bool:
        result = fetch_with_fallback(
            video.video_id,
            langs,
            # Whisper intentionally omitted (heavy / separate budget).
            fetchers=[("official", fetch_official), ("auto", fetch_auto)],
        )
        return bool(result.snippets)

    async def _warm_all() -> int:
        sem = asyncio.Semaphore(bound)
        warmed = 0

        async def _task(video: VideoMeta) -> None:
            nonlocal warmed
            async with sem:
                ok = False
                with contextlib.suppress(Exception):
                    ok = await asyncio.to_thread(_warm_one, video)
                if ok:
                    warmed += 1

        await asyncio.gather(*(_task(v) for v in videos))
        return warmed

    return asyncio.run(_warm_all())


def run_stage_scripts(
    video: VideoMeta,
    scripts_md_path: Path,
    window_seconds: float = 30.0,
    languages: list[str] | None = None,
    dry_run: bool = False,
    include_code_blocks: bool = False,
    media_path: Path | None = None,
) -> TranscriptResult:
    """Fetch transcript, chunk it, and append the body to `scripts_md_path`.

    - Uses the tier 1 → tier 2 fallback chain (Whisper is added in a
      later step via a lazy import so the optional dependency stays
      optional).
    - When `media_path` is set (``--local-media`` / fully offline), skips the
      caption tiers entirely and transcribes that local file with Whisper —
      so YouTube is never contacted for this video.
    - Does NOT overwrite the frontmatter already present; appends below.
    - Returns the `TranscriptResult` so the caller can record stats and
      pass timing info to stages 02/03.
    """
    langs = languages or DEFAULT_LANGUAGES

    # Whisper is an optional dependency — import dynamically so the
    # fallback chain degrades gracefully when not installed.
    whisper_fetcher = None
    try:
        from ..transcript.whisper_fallback import fetch_whisper

        whisper_fetcher = fetch_whisper
    except ImportError:
        pass

    if media_path is not None:
        # Local-media mode: Whisper on the local file only (no YouTube).
        if whisper_fetcher is not None:
            captured = whisper_fetcher
            source = media_path

            def _local_whisper(video_id: str, langs_: list[str]) -> TranscriptResult:
                return captured(video_id, langs_, media_path=source)

            local_fetcher: Fetcher | None = _local_whisper
        else:
            local_fetcher = None
        result = fetch_with_fallback(
            video.video_id, langs, fetchers=[("whisper-local", local_fetcher)]
        )
    else:
        result = fetch_with_fallback(
            video.video_id,
            langs,
            fetchers=[
                ("official", fetch_official),
                ("auto", fetch_auto),
                ("whisper", whisper_fetcher),
            ],
        )

    body = _render_chunks(video, chunk_by_window(result.snippets, window_seconds))

    code_section = ""
    # Skip the description fetch under --local-media: it hits YouTube (yt-dlp),
    # defeating the fully-offline guarantee this mode provides.
    if include_code_blocks and media_path is None:
        # Fetching description + raw code is best-effort. If anything
        # fails, we silently skip — the transcript is the primary asset.
        description = fetch_video_description(video.video_id)
        if description:
            urls = extract_github_urls(description)
            snippets = fetch_snippets_for_urls(urls)
            code_section = render_code_section(snippets)

    full_body = body + code_section if code_section else body

    if not dry_run and full_body:
        _append_body(scripts_md_path, full_body)

    return result


def _render_chunks(video: VideoMeta, chunks: list[Chunk]) -> str:
    """Render chunks as markdown lines matching the dummy-data format.

    Each line: `[MM:SS](<watch_url>&t=<sec>) <text>`
    """
    lines: list[str] = []
    base_url = video.watch_url
    for chunk in chunks:
        link = f"{base_url}&t={chunk.start_int}"
        lines.append(f"[{chunk.mmss}]({link}) {chunk.text}")
    return "\n".join(lines)


def _append_body(path: Path, body: str) -> None:
    """Append body to the existing placeholder md.

    The placeholder ends with a trailing newline after `---`, so we can
    append directly. We ensure a blank line separator for readability.
    """
    if not path.exists():
        raise FileNotFoundError(f"placeholder md not found: {path}")
    existing = path.read_text(encoding="utf-8")
    # Ensure a blank line between frontmatter and body
    sep = "" if existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
    path.write_text(existing + sep + body + "\n", encoding="utf-8")
