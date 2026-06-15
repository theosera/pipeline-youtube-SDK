"""Tier 3 transcript fetcher: local Whisper transcription.

Downloads the audio track via yt-dlp, runs openai-whisper, and returns
a TranscriptResult with word-level timestamps. This is the last resort
when YouTube provides neither official nor auto-generated captions.

Bounded concurrency + model cache
----------------------------------
Whisper is GPU/memory intensive. An in-process ``BoundedSemaphore``
(default 1, see ``configure_whisper_concurrency``) caps how many
transcriptions run at once even under high ``--concurrency``; other
videos queue behind it. Loaded models are memoized in ``_model_cache``
so the multi-second ``load_model`` cost is paid once per process.

Optional dependency
-------------------
`openai-whisper` is declared under `[project.optional-dependencies]`
(`uv sync --extra whisper`). If not installed, `fetch_whisper` raises
`TranscriptNotAvailable("whisper_not_installed")` immediately so the
fallback chain terminates gracefully.

Long-form guard
---------------
Multi-hour audio (e.g. day-long livestream VODs) transcribed on CPU runs
for hours and, since Whisper concurrency is bounded (default 1), a single
such video blocks the whole transcription queue. ``fetch_whisper`` probes
the duration from yt-dlp metadata *before* downloading and, if it exceeds
``DEFAULT_WHISPER_MAX_AUDIO_SECONDS`` (override via
``configure_whisper_max_audio_seconds`` / ``config.json
whisper_max_audio_seconds``), bails out via the normal
``TranscriptNotAvailable("audio_too_long: ...")`` fallback path instead of
occupying a Whisper slot. Videos whose duration metadata is unavailable
(some live streams) are not blocked — they proceed as before.

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
import platform
import re
import threading
import warnings
from pathlib import Path
from typing import Any, cast

from .base import (
    TranscriptNotAvailable,
    TranscriptResult,
    TranscriptSnippet,
    TranscriptSource,
    build_result,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_TMP_DIR = _PROJECT_ROOT / "tmp"

# Whisper is GPU/RAM-heavy, so concurrency is bounded by an in-process
# semaphore (default 1) rather than a cross-process file lock — the
# pipeline runs videos in worker threads of a single process, so a
# threading primitive is the right tool and avoids filesystem-lock latency.
DEFAULT_WHISPER_CONCURRENCY = 1
_whisper_semaphore = threading.BoundedSemaphore(DEFAULT_WHISPER_CONCURRENCY)

# Loaded whisper models are cached in-process so we don't pay the
# multi-second `load_model` cost on every transcription.
_model_cache: dict[str, Any] = {}
_model_cache_lock = threading.Lock()


def configure_whisper_concurrency(n: int) -> None:
    """Set the max number of concurrent Whisper transcriptions."""
    global _whisper_semaphore
    _whisper_semaphore = threading.BoundedSemaphore(max(1, n))


# Upper bound on audio length Whisper will attempt. CPU transcription of
# multi-hour audio runs for hours and blocks the bounded Whisper queue, so
# anything longer is skipped via the normal fallback path. 2h is generous
# for typical YouTube content while still excluding day-long VODs.
DEFAULT_WHISPER_MAX_AUDIO_SECONDS = 2 * 60 * 60
_max_audio_seconds = DEFAULT_WHISPER_MAX_AUDIO_SECONDS


def configure_whisper_max_audio_seconds(seconds: int) -> None:
    """Set the max audio duration (seconds) Whisper will attempt.

    Non-positive values disable the guard (no upper bound).
    """
    global _max_audio_seconds
    _max_audio_seconds = seconds


# Default openai-whisper model — "small" balances speed and accuracy on CPU
# while staying memory-light. Override via config.json `whisper_model`.
DEFAULT_WHISPER_MODEL = "small"
# Default MLX model — on Apple Silicon the GPU runs large-v3-turbo fast at low
# memory, so we can afford near-large accuracy by default.
DEFAULT_MLX_MODEL = "large-v3-turbo"

# Logical model name → mlx-community HF repo. An unmapped value is passed
# through verbatim so a full repo id also works.
_MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base",
    "small": "mlx-community/whisper-small",
    "medium": "mlx-community/whisper-medium",
    "large-v3": "mlx-community/whisper-large-v3",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}

_WHISPER_BACKENDS = frozenset({"auto", "mlx", "openai"})

# Backend/model selection, set once at startup via configure_whisper(). Default
# "auto" → MLX on Apple Silicon when installed, else openai-whisper.
_BACKEND = "auto"
_MODEL: str | None = None


def configure_whisper(*, backend: str = "auto", model: str | None = None) -> None:
    """Select the transcription backend and model (call once from config)."""
    global _BACKEND, _MODEL
    if backend not in _WHISPER_BACKENDS:
        raise ValueError(
            f"whisper_backend must be one of {sorted(_WHISPER_BACKENDS)}, got {backend!r}"
        )
    _BACKEND = backend
    _MODEL = model or None


def _mlx_available() -> bool:
    """True only on Apple Silicon with `mlx_whisper` importable (GPU path)."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx_whisper  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        return False
    return True


def _resolve_backend() -> str:
    """Concrete backend: 'auto' resolves to mlx on Apple Silicon, else openai."""
    if _BACKEND == "auto":
        return "mlx" if _mlx_available() else "openai"
    return _BACKEND


def _resolve_mlx_repo() -> str:
    return _MLX_REPOS.get(_MODEL or DEFAULT_MLX_MODEL, _MODEL or DEFAULT_MLX_MODEL)


def _resolve_openai_model() -> str:
    return _MODEL or DEFAULT_WHISPER_MODEL


def describe_whisper() -> str:
    """One-line resolved backend+model, e.g. 'auto→mlx (large-v3-turbo)'.

    Lets startup logging confirm whether MLX (GPU) is actually in use vs a
    silent fall-back to openai-whisper (CPU).
    """
    backend = _resolve_backend()
    model = (_MODEL or DEFAULT_MLX_MODEL) if backend == "mlx" else _resolve_openai_model()
    prefix = f"auto→{backend}" if _BACKEND == "auto" else backend
    return f"{prefix} ({model})"


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


def _probe_duration_seconds(video_id: str) -> float | None:
    """Return the video duration in seconds from yt-dlp metadata.

    Uses ``extract_info(download=False)`` so the (potentially large) audio
    track is never fetched for a video we are going to reject. Returns None
    when the duration is unavailable (e.g. some live streams) or the probe
    fails — the caller then proceeds rather than blocking on a metadata gap.
    """
    try:
        import yt_dlp  # type: ignore[import-untyped]
    except ImportError as e:
        raise TranscriptNotAvailable("yt_dlp_not_installed") from e

    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "skip_download": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        return None

    if not isinstance(info, dict):
        return None
    duration = info.get("duration")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration > 0:
        return float(duration)
    return None


def _guard_audio_duration(video_id: str) -> None:
    """Raise if the video exceeds the configured Whisper duration cap.

    Probes the duration *before* download (see ``_probe_duration_seconds``)
    and raises ``TranscriptNotAvailable("audio_too_long: ...")`` so an
    over-long video ends the fallback chain gracefully without ever
    occupying a bounded Whisper slot. A non-positive cap disables the guard.
    """
    limit = _max_audio_seconds
    if limit <= 0:
        return
    duration = _probe_duration_seconds(video_id)
    if duration is not None and duration > limit:
        raise TranscriptNotAvailable(
            f"audio_too_long: {duration:.0f}s exceeds whisper limit {limit}s"
        )


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


def _load_model_cached(model_name: str) -> Any:
    """Load a whisper model once per process and reuse it.

    The integrity check runs before the *first* load of a given model name
    (whisper skips SHA256 verification on a cache hit, so a replaced ``.pt``
    would otherwise be trusted). Subsequent calls reuse the in-memory model.
    """
    cached = _model_cache.get(model_name)
    if cached is not None:
        return cached
    with _model_cache_lock:
        cached = _model_cache.get(model_name)
        if cached is not None:
            return cached
        import whisper  # type: ignore[import-untyped]

        # L2: re-verify cached model before load.
        verify_whisper_model_integrity(model_name)
        model = whisper.load_model(model_name)
        _model_cache[model_name] = model
        return model


def _run_whisper(
    audio_path: Path,
    language: str | None = None,
) -> list[dict[str, Any]]:
    """Transcribe `audio_path` with the resolved backend; return segment dicts.

    Dispatches to MLX (Apple-Silicon GPU) or openai-whisper (CPU) per
    `configure_whisper`. Segments have keys: start, end, text.
    """
    if _resolve_backend() == "mlx":
        return _run_whisper_mlx(audio_path, language)
    return _run_whisper_openai(audio_path, _resolve_openai_model(), language)


def _run_whisper_openai(
    audio_path: Path,
    model_name: str,
    language: str | None,
) -> list[dict[str, Any]]:
    """Run openai-whisper (CPU/PyTorch) with the in-process model cache."""
    try:
        import whisper  # type: ignore[import-untyped]  # noqa: F401
    except ImportError as e:
        raise TranscriptNotAvailable("whisper_not_installed") from e

    try:
        model = _load_model_cached(model_name)
        # verbose=None suppresses whisper's own tqdm progress bar (verbose=False
        # still draws it), matching the pipeline's noprogress policy. The FP16
        # UserWarning on CPU-only hosts is expected and silenced to keep logs
        # clean — CPU transcription correctly falls back to FP32 regardless.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="FP16 is not supported on CPU.*")
            result = model.transcribe(
                str(audio_path),
                language=language,
                verbose=None,
            )
    except Exception as e:
        raise TranscriptNotAvailable(f"whisper_transcribe_failed: {e}") from e

    return cast(list[dict[str, Any]], result.get("segments", []))


def _run_whisper_mlx(audio_path: Path, language: str | None) -> list[dict[str, Any]]:
    """Run mlx-whisper (Apple-Silicon GPU). Model weights come from HF cache."""
    try:
        import mlx_whisper  # type: ignore[import-untyped]
    except ImportError as e:
        raise TranscriptNotAvailable("mlx_whisper_not_installed") from e

    try:
        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=_resolve_mlx_repo(),
            language=language,
            verbose=False,
        )
    except Exception as e:
        raise TranscriptNotAvailable(f"mlx_transcribe_failed: {e}") from e

    return cast(list[dict[str, Any]], result.get("segments", []))


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
    media_path: Path | None = None,
) -> TranscriptResult:
    """Tier 3 fetcher: download audio + Whisper transcribe.

    Acquires a bounded semaphore (default 1) before running so Whisper
    concurrency stays within GPU/RAM limits even under high --concurrency.
    The backend/model are chosen by `configure_whisper` (MLX on Apple Silicon,
    else openai).

    Parameters
    ----------
    video_id:
        YouTube video ID.
    languages:
        Preferred languages. The first entry is used as Whisper's
        `language` hint. If empty, Whisper auto-detects.
    media_path:
        When given, transcribe this **local** file directly instead of
        downloading the audio (``--local-media`` / fully-offline mode). Skips
        the YouTube duration probe; the local file is never deleted.
    """
    # Check the resolved backend's runtime is importable before the semaphore.
    if _resolve_backend() == "mlx":
        try:
            import mlx_whisper  # type: ignore[import-untyped]  # noqa: F401
        except ImportError as e:
            raise TranscriptNotAvailable("mlx_whisper_not_installed") from e
    else:
        try:
            import whisper  # type: ignore[import-untyped]  # noqa: F401
        except ImportError as e:
            raise TranscriptNotAvailable("whisper_not_installed") from e

    _ensure_tmp()

    # Skip over-long audio before touching the bounded Whisper queue: a
    # multi-hour CPU transcription would block every other video behind it.
    # The probe hits YouTube, so it's skipped for a local --local-media file.
    if media_path is None:
        _guard_audio_duration(video_id)

    # Bounded in-process concurrency (default 1) instead of a cross-process
    # file lock: the pipeline is single-process with thread workers.
    with _whisper_semaphore:
        downloaded: Path | None = None
        try:
            if media_path is not None:
                source_path = media_path
            else:
                downloaded = _download_audio(video_id)
                source_path = downloaded
            lang_hint = languages[0] if languages else None
            segments = _run_whisper(source_path, language=lang_hint)
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
            # Only clean up a file we downloaded — never the user's local file.
            if downloaded is not None:
                with contextlib.suppress(OSError):
                    downloaded.unlink(missing_ok=True)


class _noop_lock:
    """No-op context manager when filelock is not installed."""

    def __enter__(self) -> _noop_lock:
        return self

    def __exit__(self, *args: object) -> None:
        pass
