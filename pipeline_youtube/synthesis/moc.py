"""Write the `00_MOC.md` hub note for a Stage 05 synthesis result."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..obsidian import build_frontmatter
from .body_validator import validate_chapter_body
from .scoring import SynthesisMoc


def write_moc(
    moc: SynthesisMoc,
    target_path: Path,
    *,
    run_time: datetime,
    playlist_title: str,
    allowed_assets: frozenset[str] | set[str] = frozenset(),
) -> None:
    """Write `00_MOC.md` with frontmatter + leader-produced body.

    Body passes through `validate_chapter_body` (same defense layer as
    chapter writes) to strip disallowed embeds, HTML, and Templater
    tokens. The file is written atomically in one `write_text` call.
    """
    fm = build_frontmatter(
        dt=run_time,
        title=moc.title or f"{playlist_title} ハンズオン",
        url="",
        tags=["memo", "youtube", "synthesis", "moc"],
        extra={"playlist": playlist_title},
    )
    validated_body = validate_chapter_body(moc.body_markdown, allowed_assets)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(fm + "\n" + validated_body.strip() + "\n", encoding="utf-8")
