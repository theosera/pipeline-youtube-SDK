"""Backward-compat shim: moved to ``pipeline_youtube.services.obsidian``.

Vault のノート名・フォルダ名整形 / frontmatter / collision 回避は共通基盤として
``services/`` 配下へ移設。旧 import (``from .obsidian import ...``) を壊さないよう
明示 re-export する。新規コードは ``services.obsidian`` を直接参照すること。
"""

from __future__ import annotations

from .services.obsidian import (
    _escape_yaml,
    _strip_playlist_category_prefix,
    build_frontmatter,
    format_playlist_folder_name,
    format_video_note_base,
    read_frontmatter_field,
    resolve_unique_path,
    sanitize_title_for_filename,
    upsert_frontmatter_field,
)

__all__ = [
    "_escape_yaml",
    "_strip_playlist_category_prefix",
    "build_frontmatter",
    "format_playlist_folder_name",
    "format_video_note_base",
    "read_frontmatter_field",
    "resolve_unique_path",
    "sanitize_title_for_filename",
    "upsert_frontmatter_field",
]
