"""Tier 3 transcript fetcher: local Whisper transcription.

Downloads the audio track via yt-dlp, runs openai-whisper, and returns
a TranscriptResult with word-level timestamps. This is the last resort
when YouTube provides neither official nor auto-generated captions.

Global lock
-----------
Whisper is GPU/memory intensive. A file-based lock at
`{project_root}/tmp/.whisper.lock` ensures only one Whisper process
runs at a time across all pipeline instances. Other videos queue up
behind the lock. The lock file is NOT deleted on release so the path
stays stable.

Optional dependency
-------------------
`openai-whisper` is declared under `[project.optional-dependencies]`
(`uv sync --extra whisper`). If not installed, `fetch_whisper` raises
`TranscriptNotAvailable("whisper_not_installed")` immediately so the
fallback chain terminates gracefully.

Model integrity (L2)
--------------------
`whisper.load_model()` verifies SHA256 **only during the initial
download** to `~/.cache/whisper/`. On every subsequent load it reads
the cached `.pt` directly with no integrity check — so if the cached
file is replaced after first use (e.g. by a co-located malicious
process), whisper happily loads tampered weights and runs arbitrary
PyTorch deserialization on them.

`verify_whisper_model_integrity()` closes this gap by recomputing the
SHA256 of the cached model before each load and comparing against the
expected hash extracted from `whisper._MODELS`. Each model URL embeds
its SHA256 as the penultimate path segment, so the expected hash is
self-contained and version-pinned by whatever whisper version is
installed. A mismatch raises `TranscriptNotAvailable` so the pipeline
fails loudly instead of silently trusting bad weights.

Missing cache file is treated as "first run" and skipped — whisper's
own download-plus-verify path then handles that cleanly.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
from pathlib import Path
from typing import Any

from .base import (
    TranscriptNotAvailable,
    TranscriptResult,
    TranscriptSnippet,
    TranscriptSource,
    build_result,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_TMP_DIR = _PROJECT_ROOT / "tmp"
_LOCK_PATH = _TMP_DIR / ".whisper.lock"

# Default whisper model — "small" balances speed and accuracy for most
# YouTube content. Override via config.json whisper_model field (future).
DEFAULT_WHISPER_MODEL = "small"

# Each whisper `_MODELS` URL is of the form:
#   https://openaipublic.azureedge.net/main/whisper/models/<sha256>/<name>.pt
# where `<sha256>` is the expected 64-char hex hash of the .pt file.
_SHA256_IN_MODEL_URL_RE = re.compile(r"/([0-9a-f]{64})/[^/]+\.pt$")

# 1 MiB chunks — large enough to amortize syscall overhead, small
# enough to keep peak memory bounded on low-RAM machines.
_SHA256_CHUNK_BYTES = 1 << 20


def _ensure_tmp() -> None:
    _TMP_DIR.mkdir(parents=True, exist_ok=True)


def _download_audio(video_id: str) -> Path:
    """Download audio-only track via yt-dlp as m4a/mp3.

    Returns the path to the downloaded file inside tmp/.
    Raises TranscriptNotAvailable on download failure.
    """
    _ensure_tmp()
    out_template = str(_TMP_DIR / f"whisper_{video_id}.%(ext)s")
    url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        import yt_dlp  # type: ignore[import-untyped]
    except ImportError as e:
        raise TranscriptNotAvailable("yt_dlp_not_installed") from e

    ydl_opts: dict[str, Any] = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise TranscriptNotAvailable(f"audio_download_failed: {e}") from e

    # Find the downloaded file (extension may vary)
    candidates = sorted(_TMP_DIR.glob(f"whisper_{video_id}.*"))
    candidates = [c for c in candidates if c.suffix != ".lock"]
    if not candidates:
        raise TranscriptNotAvailable("audio_file_not_found_after_download")
    return candidates[0]


def _whisper_cache_dir() -> Path:
    """Mirror whisper's own default cache directory resolution.

    Matches the logic in `whisper.load_model()`:
        $XDG_CACHE_HOME/whisper, else ~/.cache/whisper
    """
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "whisper"


def _expected_sha256_for_model(model_name: str) -> str | None:
    """Return the expected SHA256 hex for a whisper model, or None.

    Reads `whisper._MODELS` (a `{name: url}` dict baked into the whisper
    package) and extracts the hash from the URL path. Returns None if
    whisper is not importable or if the URL doesn't match the expected
    shape (e.g. future whisper versions change hosting layout).
    """
    try:
        import whisper  # type: ignore[import-untyped]
    except ImportError:
        return None

    url = getattr(whisper, "_MODELS", {}).get(model_name)
    if not isinstance(url, str):
        return None
    match = _SHA256_IN_MODEL_URL_RE.search(url)
    if match is None:
        return None
    return match.group(1)


def _sha256_of_file(path: Path) -> str:
    """Compute the SHA256 hex of a file, streaming in 1 MiB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(_SHA256_CHUNK_BYTES)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_whisper_model_integrity(model_name: str) -> None:
    """Verify that a cached whisper `.pt` matches the expected SHA256.

    - If whisper is not installed, or the model name is not in
      `whisper._MODELS`, or the URL format is unrecognized: skip silently
      (no integrity claim possible).
    - If the cache file does not yet exist: skip silently. whisper will
      download and verify on first `load_model()` call — that path
      already does SHA256 verification correctly.
    - If the cache file exists but its SHA256 does not match the
      expected value: raise `TranscriptNotAvailable` with a
      `whisper_model_integrity_mismatch:...` reason so the pipeline
      surfaces the tamper event instead of loading bad weights.
    """
    expected = _expected_sha256_for_model(model_name)
    if expected is None:
        return

    cache_path = _whisper_cache_dir() / f"{model_name}.pt"
    if not cache_path.is_file():
        return

    actual = _sha256_of_file(cache_path)
    if actual != expected:
        raise TranscriptNotAvailable(
            f"whisper_model_integrity_mismatch: {model_name} "
            f"expected={expected[:12]}... actual={actual[:12]}... "
            f"path={cache_path}"
        )


def _run_whisper(
    audio_path: Path,
    model_name: str = DEFAULT_WHISPER_MODEL,
    language: str | None = None,
) -> list[dict[str, Any]]:
    """Run openai-whisper on the audio file and return segments.

    Returns a list of segment dicts with keys: start, end, text.
    Raises TranscriptNotAvailable if whisper is not installed or fails,
    or if the cached model file has been tampered with (see
    `verify_whisper_model_integrity`).
    """
    try:
        import whisper  # type: ignore[import-untyped]
    except ImportError as e:
        raise TranscriptNotAvailable("whisper_not_installed") from e

    # L2: re-verify cached model before load (whisper skips this check
    # on cache hit, so a replaced .pt would otherwise be trusted).
    verify_whisper_model_integrity(model_name)

    try:
        model = whisper.load_model(model_name)
        result = model.transcribe(
            str(audio_path),
            language=language,
            verbose=False,
        )
    except Exception as e:
        raise TranscriptNotAvailable(f"whisper_transcribe_failed: {e}") from e

    return result.get("segments", [])


def _segments_to_snippets(segments: list[dict[str, Any]]) -> list[TranscriptSnippet]:
    """Convert whisper segments to TranscriptSnippet list."""
    snippets: list[TranscriptSnippet] = []
    for seg in segments:
        start = float(seg.get("start", 0))
        end = float(seg.get("end", start))
        text = str(seg.get("text", "")).strip()
        if text:
            snippets.append(TranscriptSnippet(text=text, start=start, duration=end - start))
    return snippets


def _detect_language(segments: list[dict[str, Any]]) -> str | None:
    """Best-effort language detection from whisper output."""
    # Whisper segments don't carry language, but the model result does.
    # We pass language through the caller chain instead.
    return None


def fetch_whisper(
    video_id: str,
    languages: list[str],
    *,
    model_name: str = DEFAULT_WHISPER_MODEL,
) -> TranscriptResult:
    """Tier 3 fetcher: download audio + Whisper transcribe.

    Acquires a file-based global lock before running. Only one Whisper
    instance runs at a time regardless of --concurrency.

    Parameters
    ----------
    video_id:
        YouTube video ID.
    languages:
        Preferred languages. The first entry is used as Whisper's
        `language` hint. If empty, Whisper auto-detects.
    model_name:
        Whisper model size (tiny/base/small/medium/large).
    """
    # Check whisper is importable before acquiring lock
    try:
        import whisper  # type: ignore[import-untyped]  # noqa: F401
    except ImportError as e:
        raise TranscriptNotAvailable("whisper_not_installed") from e

    # File-based lock — only one whisper process at a time
    try:
        import filelock  # type: ignore[import-untyped]
    except ImportError:
        # filelock not installed — fall back to no lock (still works,
        # just no cross-process protection)
        filelock = None  # type: ignore[assignment]

    _ensure_tmp()
    lock_ctx = filelock.FileLock(_LOCK_PATH, timeout=-1) if filelock else _noop_lock()

    with lock_ctx:
        audio_path: Path | None = None
        try:
            audio_path = _download_audio(video_id)
            lang_hint = languages[0] if languages else None
            segments = _run_whisper(audio_path, model_name=model_name, language=lang_hint)
            snippets = _segments_to_snippets(segments)

            if not snippets:
                raise TranscriptNotAvailable("whisper_produced_no_segments")

            return build_result(
                video_id=video_id,
                source=TranscriptSource.WHISPER,
                language=lang_hint,
                snippets=snippets,
            )
        finally:
            # Clean up audio file
            if audio_path is not None:
                with contextlib.suppress(OSError):
                    audio_path.unlink(missing_ok=True)


class _noop_lock:
    """No-op context manager when filelock is not installed."""

    def __enter__(self) -> _noop_lock:
        return self

    def __exit__(self, *args: object) -> None:
        pass
