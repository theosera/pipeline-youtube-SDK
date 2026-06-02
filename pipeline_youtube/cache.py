"""Content-addressed persistent cache for expensive, deterministic artifacts.

Stores transcripts, downloaded videos, fetched code snippets, and
(optionally) LLM outputs outside the repo and the Obsidian vault so that
re-runs and partial re-runs are near-instant. This is the single biggest
lever on wall-clock time for the common "run the same playlist again" /
``--synthesis-only`` workflows.

Security posture: video-derived artifacts never enter git or the vault.
The default root is ``~/.cache/pipeline-youtube/`` (honoring
``XDG_CACHE_HOME``), overridable via ``--cache-dir`` / config / the
``PIPELINE_YOUTUBE_CACHE`` env var, and fully disengaged by ``--no-cache``.

Namespaces
----------
``transcript/{video_id}/{tier}/{lang}``       JSON   deterministic, always cached
``video/{video_id}/{fmt}``                    bytes  deterministic, LRU/size-capped
``code_fetch/{sha256(url)}``                  JSON   deterministic, always cached
``llm/{sha256(provider+model+system+prompt)}`` JSON  per-role policy (see registry)

Thread-safety
-------------
Reads are lock-free. Writes are atomic (``os.replace`` of a same-dir temp
file), so concurrent writers are safe — values are content-addressed, so a
race just means two identical writes. Only the video-namespace eviction
bookkeeping takes a lock. This matters because videos are processed in
worker threads under ``--concurrency``.

When disabled (``--no-cache`` or never configured), every read misses and
every write is dropped, so callers degrade to "recompute every run" with no
branching at the call site.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

# Video binaries are the only large artifact and the only namespace subject
# to eviction. 10 GiB holds a few dozen 480p videos before LRU kicks in.
DEFAULT_MAX_VIDEO_BYTES = 10 * 1024 * 1024 * 1024

_VIDEO_NS = "video"


def resolve_cache_root(explicit: str | Path | None = None) -> Path:
    """Resolve the cache root with the documented precedence.

    ``--cache-dir`` / config (``explicit``) > ``PIPELINE_YOUTUBE_CACHE`` env
    > ``XDG_CACHE_HOME``/pipeline-youtube > ``~/.cache/pipeline-youtube``.
    Mirrors ``whisper_fallback._whisper_cache_dir`` for consistency.
    """
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("PIPELINE_YOUTUBE_CACHE")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "pipeline-youtube"


def llm_key(provider: str, model: str, system: str | None, prompt: str) -> str:
    """SHA256 key for an LLM call.

    Provider and model are part of the key so a model swap never collides.
    A NUL separator prevents boundary collisions between concatenated parts.
    """
    h = hashlib.sha256()
    for part in (provider, model, system or "", prompt):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def url_key(url: str) -> str:
    """SHA256 key for a code-fetch URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _safe_segment(seg: str) -> str:
    """Sanitize one path segment so cache keys can't escape the root."""
    seg = seg.replace("/", "_").replace("\\", "_").replace("\x00", "_")
    if seg in ("", ".", ".."):
        return "_"
    return seg


class Cache:
    """Content-addressed cache rooted at a directory.

    Construct via :func:`configure_cache` / :func:`get_cache`; a disabled
    instance (``enabled=False``) is a no-op used for ``--no-cache``, tests,
    and library callers that never opt in.
    """

    def __init__(
        self,
        root: Path | None,
        *,
        enabled: bool = True,
        max_video_bytes: int = DEFAULT_MAX_VIDEO_BYTES,
    ) -> None:
        self._enabled = enabled and root is not None
        self._root = root
        self._max_video_bytes = max_video_bytes
        self._evict_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def root(self) -> Path | None:
        return self._root

    # -- internal path helpers ------------------------------------------

    def _path(self, namespace: str, *segments: str, shard: bool = False) -> Path:
        assert self._root is not None  # guarded by self._enabled at call sites
        parts = [_safe_segment(namespace)]
        if shard and segments:
            # 2-char fan-out on a hash key to avoid one huge directory.
            key = _safe_segment(segments[0])
            parts.extend([key[:2], key])
        else:
            parts.extend(_safe_segment(s) for s in segments)
        return self._root.joinpath(*parts)

    def _get_json(self, path: Path) -> Any | None:
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _put_json(self, path: Path, obj: Any) -> None:
        try:
            data = json.dumps(obj, ensure_ascii=False)
        except (TypeError, ValueError):
            return  # value isn't serializable — silently skip caching
        self._atomic_write(path, data.encode("utf-8"))

    def _atomic_write(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, path)  # atomic on same filesystem
        except OSError:
            with _suppress_oserror():
                os.unlink(tmp)

    # -- transcript namespace (always cached) ---------------------------

    def get_transcript(self, video_id: str, tier: str, lang: str) -> Any | None:
        if not self._enabled:
            return None
        return self._get_json(self._path("transcript", video_id, tier, lang))

    def put_transcript(self, video_id: str, tier: str, lang: str, obj: Any) -> None:
        if not self._enabled:
            return
        self._put_json(self._path("transcript", video_id, tier, lang), obj)

    # -- code_fetch namespace (always cached) ---------------------------

    def get_code_fetch(self, key: str) -> Any | None:
        if not self._enabled:
            return None
        return self._get_json(self._path("code_fetch", key, shard=True))

    def put_code_fetch(self, key: str, obj: Any) -> None:
        if not self._enabled:
            return
        self._put_json(self._path("code_fetch", key, shard=True), obj)

    # -- llm namespace (per-role policy decided by the caller) ----------

    def get_llm(self, key: str) -> Any | None:
        if not self._enabled:
            return None
        return self._get_json(self._path("llm", key, shard=True))

    def put_llm(self, key: str, obj: Any) -> None:
        if not self._enabled:
            return
        self._put_json(self._path("llm", key, shard=True), obj)

    # -- video namespace (large; LRU/size-capped) -----------------------

    def get_video(self, video_id: str, fmt: str) -> Path | None:
        """Return the cached video path, refreshing its atime, or None."""
        if not self._enabled:
            return None
        path = self._path(_VIDEO_NS, video_id, fmt)
        if not path.exists():
            return None
        with _suppress_oserror():
            os.utime(path)  # bump atime so reused videos survive eviction
        return path

    def put_video(self, video_id: str, fmt: str, src: Path) -> None:
        """Copy ``src`` into the cache, then evict oldest videos if over cap."""
        if not self._enabled:
            return
        path = self._path(_VIDEO_NS, video_id, fmt)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-")
        os.close(fd)
        try:
            shutil.copy2(src, tmp)
            os.replace(tmp, path)
        except OSError:
            with _suppress_oserror():
                os.unlink(tmp)
            return
        self._evict_videos_if_needed()

    def _evict_videos_if_needed(self) -> None:
        video_root = self._path(_VIDEO_NS)
        if not video_root.exists():
            return
        with self._evict_lock:
            files: list[tuple[float, int, Path]] = []
            total = 0
            for f in video_root.rglob("*"):
                if not f.is_file() or f.name.startswith(".tmp-"):
                    continue
                try:
                    st = f.stat()
                except OSError:
                    continue
                files.append((st.st_atime, st.st_size, f))
                total += st.st_size
            if total <= self._max_video_bytes:
                return
            files.sort(key=lambda t: t[0])  # oldest atime first
            for _atime, size, f in files:
                if total <= self._max_video_bytes:
                    break
                with _suppress_oserror():
                    f.unlink()
                    total -= size


class _suppress_oserror:
    """Tiny contextlib.suppress(OSError) without importing contextlib here."""

    def __enter__(self) -> _suppress_oserror:
        return self

    def __exit__(self, exc_type: object, *_: object) -> bool:
        return exc_type is not None and issubclass(exc_type, OSError)  # type: ignore[arg-type]


# =====================================================
# Module-level singleton (mirrors config.py setter/getter pattern)
# =====================================================

_cache: Cache | None = None


def configure_cache(
    root: str | Path | None = None,
    *,
    enabled: bool = True,
    max_video_bytes: int = DEFAULT_MAX_VIDEO_BYTES,
) -> Cache:
    """Install the process-wide cache. Called once from ``main.cli()``."""
    global _cache
    if not enabled:
        _cache = Cache(None, enabled=False)
    else:
        _cache = Cache(resolve_cache_root(root), enabled=True, max_video_bytes=max_video_bytes)
    return _cache


def get_cache() -> Cache:
    """Return the configured cache, defaulting to a disabled no-op.

    Defaulting to disabled keeps tests and library callers from writing to
    the real ``~/.cache`` unless they explicitly opt in via ``configure_cache``.
    """
    global _cache
    if _cache is None:
        _cache = Cache(None, enabled=False)
    return _cache


def reset_cache() -> None:
    """Reset the singleton (test hook)."""
    global _cache
    _cache = None
