"""Backward-compat shim: moved to ``pipeline_youtube.services.cache``.

新規コードは ``services.cache`` を直接参照すること。
"""

from __future__ import annotations

from .services.cache import (
    DEFAULT_MAX_VIDEO_BYTES,
    Cache,
    configure_cache,
    get_cache,
    llm_key,
    reset_cache,
    resolve_cache_root,
    url_key,
)

__all__ = [
    "DEFAULT_MAX_VIDEO_BYTES",
    "Cache",
    "configure_cache",
    "get_cache",
    "llm_key",
    "reset_cache",
    "resolve_cache_root",
    "url_key",
]
