"""Stage 01: timestamped transcript → N-second chunks → markdown.

Output format matches the dummy data in
`Permanent Note/08_YouTube学習/01_Scripts_Processing_Unit/`:

    [MM:SS](https://www.youtube.com/watch?v=<id>&t=<seconds>) chunk text...
    [MM:SS](https://www.youtube.com/watch?v=<id>&t=<seconds>) chunk text...

The frontmatter above the body is already written by the placeholder
step (`pipeline.create_placeholder_notes`), so this stage appends the
chunked body to the existing file.

The video's description + declared chapters are fetched once (best-effort,
skipped under ``--local-media``) and attached to the returned
``TranscriptResult`` for Stage 01b (known-context, fewer web searches) and
Stage 02 (Mode-diagnosis context) to consume. When ``include_code_blocks=True``
is also passed (set by the orchestrator when the Router classifies the
playlist as ``coding``), this stage additionally scrapes the fetched
description for GitHub blob/Gist URLs, downloads their raw content
(size-capped), and appends a ``## 関連コード`` section after the transcript.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from youtube_transcript_api import YouTubeTranscriptApi

from ..code_fetch import (
    extract_github_urls,
    fetch_snippets_for_urls,
    fetch_video_extra_metadata,
    render_code_section,
)
from ..playlist import VideoMeta
from ..synthesis.body_validator import validate_chapter_body
from ..transcript.auto import fetch_auto
from ..transcript.base import Fetcher, TranscriptResult, fetch_with_fallback
from ..transcript.chunking import Chunk, chunk_by_window
from ..transcript.correction import chunks_to_snippets, correct_chunks
from ..transcript.innertube import fetch_innertube
from ..transcript.official import fetch_official

if TYPE_CHECKING:
    from ..services.cache import Cache

_log = logging.getLogger(__name__)

DEFAULT_LANGUAGES: list[str] = ["ja", "en"]

# Default fan-out for the upfront transcript warm-up. Higher than the
# pipeline's --concurrency because caption fetches are light network I/O,
# not CPU/GPU/LLM bound.
DEFAULT_TRANSCRIPT_CONCURRENCY = 8

# Ceiling on the body handed to the validator and written to 01_Scripts.md.
#
# Deliberately *not* ``summary._MAX_OUTPUT_CHARS`` (50,000): that bounds LLM
# output, whereas this body is a full caption track plus fetched source, and
# 50,000 would reject ordinary long talks. Two independent derivations agree
# on ~400k:
#
#   - Measured: the longest real 01_Scripts.md body in the vault is 79,483
#     chars (sample of the four longest-plausible notes; a 2h34m webinar is
#     75,042). Five times that is 397k.
#   - Structural: transcript + ``render_code_section``, whose output cannot
#     exceed ``MAX_URLS_PER_VIDEO`` (5) * ``MAX_BYTES_PER_FILE`` (50,000)
#     plus per-snippet markup — so 79,483 + ~251,000 = ~331k.
#
# Passing it is fail-closed (``ScriptBodyTooLargeError``). Truncating instead
# would drop text that *looks* complete to every reader downstream, and the
# cut would land mid-construct.
MAX_SCRIPT_BODY_CHARS = 400_000


def warm_transcript_cache(
    videos: list[VideoMeta],
    *,
    languages: list[str] | None = None,
    concurrency: int = DEFAULT_TRANSCRIPT_CONCURRENCY,
    use_innertube: bool = True,
    cache: Cache,
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

    ``cache`` is the transcript cache to warm (injected by the caller; a
    disabled ``Cache`` makes this a no-op via the ``cache.enabled`` guard below).
    """
    if not videos:
        return 0

    # Without a persistent cache there is nothing to warm — Stage 01 would
    # re-fetch regardless, so skip the extra pass entirely.
    if not cache.enabled:
        return 0

    langs = languages or DEFAULT_LANGUAGES
    bound = max(1, concurrency)

    # InnerTube (iOS client) first — bypasses the bot/PO-token blocks that hit
    # youtube-transcript-api during a high fan-out warm-up. Honors use_innertube
    # so the warm-up and per-video Stage 01 share one on/off switch (disable on
    # datacenter IPs where tier 0 just 403s). Whisper intentionally omitted
    # (heavy / separate budget).
    innertube_tier: tuple[str, Fetcher | None] = (
        "innertube",
        fetch_innertube if use_innertube else None,
    )

    def _warm_one(video: VideoMeta) -> bool:
        # One YouTubeTranscriptApi (its own requests.Session) per worker task:
        # _warm_one runs under asyncio.to_thread at high fan-out, so a shared
        # instance would mean concurrent threads racing one mutable HTTP
        # session. official + auto within this task share it.
        api = YouTubeTranscriptApi()
        result = fetch_with_fallback(
            video.video_id,
            langs,
            fetchers=[
                innertube_tier,
                ("official", partial(fetch_official, api=api)),
                ("auto", partial(fetch_auto, api=api)),
            ],
            cache=cache,
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
    correct_model: str | None = None,
    known_terms: list[tuple[str, str]] | None = None,
    use_innertube: bool = True,
    *,
    cache: Cache,
) -> TranscriptResult:
    """Fetch transcript, chunk it, and append the body to `scripts_md_path`.

    - Uses the fallback chain tier 0 (InnerTube iOS-client captions —
      best-effort, on by default via `use_innertube`, skipped when False) →
      tier 1/2 (youtube-transcript-api manual/auto) → tier 3 (Whisper, added
      via a lazy import so the optional dependency stays optional).
    - When `media_path` is set (``--local-media`` / fully offline), skips the
      caption tiers entirely and transcribes that local file with Whisper —
      so YouTube is never contacted for this video.
    - When `correct_model` is set (Stage 01b), the chunked transcript is
      passed through an LLM + web-search correction pass (timestamps
      preserved) and the corrected text is folded back into the returned
      ``TranscriptResult.snippets`` so Stage 02/03/04 consume it. Skipped
      under `dry_run`. `known_terms` (the per-playlist confirmed vocabulary)
      is forwarded so already-known proper nouns skip the web search; the
      video's description (fetched once below) is forwarded too so the
      correction pass can resolve proper nouns from it before searching.
      The proper nouns the pass confirms are returned on
      ``TranscriptResult.confirmed_terms``.
    - The video's description + chapters are fetched once (best-effort,
      skipped under ``--local-media``) and attached to the returned
      ``TranscriptResult`` regardless of `correct_model`/`include_code_blocks`.
    - Does NOT overwrite the frontmatter already present; appends below.
    - Returns the `TranscriptResult` so the caller can record stats and
      pass timing info to stages 02/03.

    ``cache`` is injected by the caller and threaded into the transcript /
    code-fetch / correction calls below (a disabled ``Cache`` no-ops them).
    """
    langs = languages or DEFAULT_LANGUAGES

    # Whisper is an optional dependency — import dynamically so the
    # fallback chain degrades gracefully when not installed.
    whisper_fetcher = None
    # Qualify the whisper cache tier with the resolved backend+model so changing
    # whisper_backend/whisper_model re-transcribes instead of reusing a stale
    # cached transcript (Codex cache-key concern).
    whisper_tier = "whisper"
    whisper_local_tier = "whisper-local"
    try:
        from ..transcript.whisper_fallback import fetch_whisper, whisper_cache_tag

        whisper_fetcher = fetch_whisper
        tag = whisper_cache_tag()
        whisper_tier = f"whisper-{tag}"
        whisper_local_tier = f"whisper-local-{tag}"
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
            video.video_id,
            langs,
            fetchers=[(whisper_local_tier, local_fetcher)],
            cache=cache,
        )
    else:
        # Tier 0 (InnerTube iOS client) is tried first: it fetches existing
        # YouTube captions without the bot/PO-token challenges that increasingly
        # block youtube-transcript-api, keeping caption-bearing videos off the
        # slow Whisper path. Best-effort — on failure the chain falls through to
        # the youtube-transcript-api tiers and then Whisper.
        innertube_tier: tuple[str, Fetcher | None] = (
            "innertube",
            fetch_innertube if use_innertube else None,
        )
        # One YouTubeTranscriptApi per Stage-01 call, shared by the official +
        # auto tiers below (single-threaded within a video).
        api = YouTubeTranscriptApi()
        result = fetch_with_fallback(
            video.video_id,
            langs,
            fetchers=[
                innertube_tier,
                ("official", partial(fetch_official, api=api)),
                ("auto", partial(fetch_auto, api=api)),
                (whisper_tier, whisper_fetcher),
            ],
            cache=cache,
        )

    chunks = chunk_by_window(result.snippets, window_seconds)

    # Fetch description + chapters once, up front, so both Stage 01b (below)
    # and the returned TranscriptResult (Stage 02) can use them. Skipped
    # under --local-media: it hits YouTube (yt-dlp), defeating the fully-
    # offline guarantee that mode provides. Best-effort — a failed fetch
    # yields an empty VideoExtraMetadata, never raises.
    video_extra = None
    if media_path is None:
        video_extra = fetch_video_extra_metadata(video.video_id, cache=cache)

    # Stage 01b: repair ASR/caption errors with an LLM + web search. Best-effort
    # and timestamp-preserving — never blocks the run. Skipped on dry runs (it
    # is a paid LLM call) and when there is nothing to correct. The corrected
    # text is folded back into `result.snippets` so Stage 02/03/04 (which
    # re-chunk the TranscriptResult) consume the correction, not just the 01 md.
    if correct_model and not dry_run and chunks:
        correction = correct_chunks(
            chunks,
            model=correct_model,
            known_terms=known_terms,
            description=video_extra.description if video_extra else None,
            cache=cache,
        )
        chunks = correction.chunks
        last = result.snippets[-1]
        result = replace(
            result,
            snippets=chunks_to_snippets(chunks, last_end=last.start + last.duration),
            correction_cost_usd=correction.cost_usd,
            confirmed_terms=tuple(correction.confirmed_terms),
        )
    if video_extra is not None:
        result = replace(result, description=video_extra.description, chapters=video_extra.chapters)
    body = _render_chunks(video, chunks)

    code_section = ""
    if include_code_blocks and video_extra is not None and video_extra.description:
        urls = extract_github_urls(video_extra.description)
        snippets = fetch_snippets_for_urls(urls, cache=cache)
        code_section = render_code_section(snippets)

    full_body = body + code_section if code_section else body

    if not dry_run and full_body:
        full_body = _validate_script_body(full_body)
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


class ScriptBodyTooLargeError(ValueError):
    """Raised when the rendered Stage 01 body exceeds ``MAX_SCRIPT_BODY_CHARS``."""


def _validate_script_body(body: str) -> str:
    """Bound, then sanitize, the Stage 01 body before it reaches the vault.

    Stage 01 was the last writer reaching disk unvalidated. Its body is
    ``_render_chunks`` output — caption cues authored by whoever uploaded the
    video — concatenated with ``render_code_section`` output fetched from
    GitHub/Gist, so `<script>`, `<% … %>` and `![[…]]` all arrive verbatim
    from outside. ``frozenset()`` because this note legitimately embeds
    nothing; every ``![[…]]`` is a dropped-embed comment.

    Size is checked *before* validating, not after: the check exists to bound
    what the sanitize loop is handed, so running it second would defeat it.

    Two things this deliberately does not do:

    - No homoglyph fold. Stage 02/04/05 fold because they write LLM-generated
      Japanese prose; this note is a verbatim transcript, and folding
      Cyrillic/Greek into Latin would corrupt legitimate Russian or Greek
      captions. A Cyrillic-obfuscated ``<sсript>`` therefore survives here —
      inert, since Obsidian does not render it as a tag, and no downstream
      stage folds this body (Stage 02 re-chunks the ``TranscriptResult``,
      not this markdown).
    - No code-fence exemption. ``validate_chapter_body`` is fence-blind, so
      ``<script>`` inside a fetched snippet is stripped like any other. Kept
      uniform with every other writer on purpose — a per-stage fence policy
      would put two different rules in one validator.

    What it does not cover is pinned in ``TestCurrentValidatorLimits``: the
    validator strips five tag names and nothing else, and because it strips
    rather than escapes, ``<im<script>g … onerror=…>`` splices down to a live
    ``<img … onerror=…>``. Hardening that means rewriting the validator, which
    is a different review perspective from wiring this stage into it.
    """
    if len(body) > MAX_SCRIPT_BODY_CHARS:
        # No body text in the message: it is attacker-influenced and this
        # record reaches logs.
        _log.warning(
            "stage 01 body is %d chars, over the %d limit; refusing to write",
            len(body),
            MAX_SCRIPT_BODY_CHARS,
        )
        raise ScriptBodyTooLargeError(
            f"stage 01 body exceeds {MAX_SCRIPT_BODY_CHARS} chars: {len(body)}"
        )
    return validate_chapter_body(body, frozenset())


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
