"""Backward-compat shim: moved to ``pipeline_youtube.services.path_safety``.

新規コードは ``services.path_safety`` を直接参照すること。
"""

from __future__ import annotations

from .services.path_safety import (
    FALLBACK_PATH,
    MAX_PATH_LENGTH,
    ensure_safe_path,
    safe_rename,
)

__all__ = [
    "FALLBACK_PATH",
    "MAX_PATH_LENGTH",
    "ensure_safe_path",
    "safe_rename",
]
