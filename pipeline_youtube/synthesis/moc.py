"""Write the `00_MOC.md` hub note for a Stage 05 synthesis result."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..obsidian import build_frontmatter
from ..services.confusables import (
    fold_markdown_mixed_script_confusables,
    fold_mixed_script_confusables,
)
from .body_validator import validate_chapter_body
from .scoring import SynthesisMoc


def write_moc(
    moc: SynthesisMoc,
    target_path: Path,
    *,
    run_time: datetime,
    playlist_title: str,
    allowed_assets: frozenset[str] | set[str] = frozenset(),
    generated_chapter_link_targets: frozenset[str] | set[str] = frozenset(),
) -> None:
    """Write `00_MOC.md` with frontmatter + leader-produced body.

    Body passes through `validate_chapter_body` (same defense layer as
    chapter writes) to strip disallowed embeds, HTML, and Templater
    tokens. The file is written atomically in one `write_text` call.

    Homoglyphs are folded on the Leader-generated title and body UP FRONT —
    before the frontmatter and body sanitization are built — so concealment
    stays out of the `title` and, critically, the body is folded BEFORE
    ``validate_chapter_body`` strips active markup (a Cyrillic-obfuscated
    ``<sсript>`` folds to ``<script>`` and is then stripped, not written).
    Existing external wikilink/embed targets stay unchanged; generated chapter
    targets passed via ``generated_chapter_link_targets`` are folded with their
    filenames. ``playlist_title`` is external (YouTube) content and is left to
    the frontmatter's own invisible-char defense, not folded.
    """
    title = fold_mixed_script_confusables(moc.title) if moc.title else moc.title
    body = fold_markdown_mixed_script_confusables(
        moc.body_markdown,
        fold_wikilink_targets=generated_chapter_link_targets,
    )

    fm = build_frontmatter(
        dt=run_time,
        title=title or f"{playlist_title} ハンズオン",
        url="",
        tags=["memo", "youtube", "synthesis", "moc"],
        extra={"playlist": playlist_title},
    )
    validated_body = validate_chapter_body(body, allowed_assets)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(fm + "\n" + validated_body.strip() + "\n", encoding="utf-8")
