"""Backward-compat shim: moved to ``pipeline_youtube.services.checkpoint``.

新規コードは ``services.checkpoint`` を直接参照すること。``_find_learning_folder``
は ``resume`` が遅延 import するため re-export に含める。
"""

from __future__ import annotations

from .services.checkpoint import (
    _find_learning_folder,
    extract_trusted_video_id,
    get_completed_video_ids,
    is_video_complete,
    read_trusted_video_id,
)

__all__ = [
    "_find_learning_folder",
    "extract_trusted_video_id",
    "get_completed_video_ids",
    "is_video_complete",
    "read_trusted_video_id",
]
