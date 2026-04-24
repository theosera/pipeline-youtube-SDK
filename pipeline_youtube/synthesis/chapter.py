"""Write per-chapter md files for a Stage 05 synthesis result."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..obsidian import build_frontmatter, sanitize_title_for_filename
from ..path_safety import ensure_safe_path
from ..sanitize import sanitize_untrusted_text
from .body_validator import validate_chapter_body
from .scoring import SynthesisChapterBody

_MAX_FILENAME_BYTES = 200
_EXT = ".md"


def chapter_filename(index: int, label: str) -> str:
    """Build a safe chapter md filename: `{NN}_{sanitized-label}.md`.

    Defense layers:
      1. `sanitize_untrusted_text` strips zero-width / control chars
         (e.g. U+202E right-to-left override used for Obsidian filename
         spoofing) and caps the raw label at 150 chars.
      2. `sanitize_title_for_filename` replaces OS-unsafe chars
         (`\\ / : * ? " < > |`).
      3. UTF-8 boundary-safe truncation keeps the total filename under
         200 bytes while preserving the `.md` suffix.
    """
    cleaned = sanitize_untrusted_text(label, 150, context="synthesis.chapter.filename")
    safe_label = sanitize_title_for_filename(cleaned) or f"chapter-{index}"

    prefix = f"{index:02d}_"
    prefix_bytes = len(prefix.encode("utf-8"))
    ext_bytes = len(_EXT.encode("utf-8"))
    max_label_bytes = _MAX_FILENAME_BYTES - prefix_bytes - ext_bytes
    truncated_label = (
        safe_label.encode("utf-8")[:max_label_bytes].decode("utf-8", errors="ignore")
        or f"chapter-{index}"
    )
    return f"{prefix}{truncated_label}{_EXT}"


def write_chapter(
    chapter: SynthesisChapterBody,
    playlist_dir: Path,
    *,
    run_time: datetime,
    playlist_title: str,
    allowed_assets: frozenset[str] | set[str] = frozenset(),
) -> Path:
    """Write a single chapter md and return the absolute path.

    Filename is `{NN}_{label}.md` under `playlist_dir`. Body passes
    through `validate_chapter_body` to strip disallowed embeds, active
    HTML, and Templater tokens. Pass `allowed_assets` (image filenames
    present in source 04 md) to permit their embeds.
    """
    filename = chapter_filename(chapter.chapter_index, chapter.label)
    target = playlist_dir / filename

    fm = build_frontmatter(
        dt=run_time,
        title=chapter.label,
        url="",
        tags=["memo", "youtube", "synthesis"],
        extra={
            "playlist": playlist_title,
            "chapter": str(chapter.chapter_index),
            "category": chapter.category,
            "sources": ", ".join(chapter.source_video_ids),
        },
    )

    validated_body = validate_chapter_body(chapter.body_markdown, allowed_assets)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(fm + "\n" + validated_body.strip() + "\n", encoding="utf-8")
    return target


def validate_chapter_relative_path(relative_path: str) -> str:
    """Run a chapter output path through the 7-layer path-safety filter."""
    return ensure_safe_path(relative_path)
