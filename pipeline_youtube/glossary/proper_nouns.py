"""Per-playlist, human-editable proper-noun sheet (TSV).

Distinct from the strict first-party ``glossary.json`` (``schema.py``): this TSV
is a *working sheet* that Stage 01b **writes** (the proper nouns it confirmed via
web search, one section per video) and the user **edits** before Stage 05
(filling the right column to override a spelling). Because a human edits it by
hand it is parsed leniently — malformed lines are skipped, never fatal — the
opposite of the strict JSON glossary.

It exists to do three things:

1. **Cut web-search cost** — on the next run, terms already in the sheet are fed
   to Stage 01b as a confirmed vocabulary so the model uses that spelling
   without searching again.
2. **Human-in-the-loop correction** — a spelling the user fixes in the right
   column becomes authoritative and is rewritten into the Stage 05 output.
3. **Feed ``glossary.json``** — user-corrected rows are promoted to permanent
   glossary entries (correction = canonical, system spelling = alias).

On-disk format (TAB-separated, ``#`` comments, ``## [video_id] title`` heads)::

    # 固有名詞辞書 ...
    ## [_h3decBW12Q] Anthropicが公開したハーネス設計
    Anthropic\t
    ヴァイブコーディング\tVibe Coding

The first column is the system-confirmed spelling (authoritative when the right
column is blank); the second is the user's override.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .normalizer import fold_term
from .schema import Glossary, GlossaryEntry

# Placed directly under the 01_Scripts playlist folder. The ``__`` prefix sorts
# it to the top of the folder and signals a system-owned (non-note) file.
SHEET_FILENAME = "__proper_nouns.tsv"

# Leading comment block rewritten on every render so the editing contract stays
# in front of the user. Kept as plain lines (no trailing newline handling here).
_HEADER_LINES: tuple[str, ...] = (
    "# 固有名詞辞書 (proper-noun glossary) — Stage 01b が自動生成します。",
    "# 各行: <システム確定語><TAB><ユーザー訂正>  （右が空ならシステム確定語を採用）",
    "# 右列に正しい表記を書くとそれを正とみなし、Stage 05 出力と次回以降の実行に反映します。",
    "# 訂正した行は glossary.json にも取り込まれます。Stage 05 へ進む前に保存してください。",
)

_SECTION_RE = re.compile(r"^##\s*\[(?P<vid>[^\]]+)\]\s*(?P<title>.*)$")


@dataclass(frozen=True)
class ProperNounRow:
    """One proper noun: the system spelling and an optional user override.

    ``resolved`` is the authoritative spelling (the override when present, else
    the system spelling). ``is_user_corrected`` is the signal that this row was
    edited by a human — it drives both Stage 05 rewriting and the
    ``glossary.json`` promotion.
    """

    system_term: str
    user_correction: str = ""

    @property
    def resolved(self) -> str:
        return self.user_correction or self.system_term

    @property
    def is_user_corrected(self) -> bool:
        return bool(self.user_correction) and fold_term(self.user_correction) != fold_term(
            self.system_term
        )


@dataclass(frozen=True)
class VideoSection:
    """One video's heading + rows in the sheet, keyed by ``video_id``."""

    video_id: str
    title: str = ""
    rows: tuple[ProperNounRow, ...] = ()


@dataclass(frozen=True)
class ProperNounSheet:
    """The whole per-playlist sheet: an ordered tuple of video sections."""

    sections: tuple[VideoSection, ...] = ()

    def section_for(self, video_id: str) -> VideoSection | None:
        for section in self.sections:
            if section.video_id == video_id:
                return section
        return None


def _dedup_terms(terms: list[str]) -> list[str]:
    """Strip, drop empties, and fold-dedup ``terms``, preserving first order."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in terms:
        term = raw.strip()
        key = fold_term(term)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(term)
    return out


def parse_sheet(text: str) -> ProperNounSheet:
    """Parse the TSV text into a ``ProperNounSheet`` (lenient: skip junk).

    Section heads (``## [id] title``) open a video section; ``#`` comments and
    blank lines are ignored; any other line is a row split on the first TAB.
    Rows appearing before the first section head (or with an empty left column)
    are dropped rather than raising — a human owns this file.
    """
    sections: list[VideoSection] = []
    cur_id: str | None = None
    cur_title = ""
    cur_rows: list[ProperNounRow] = []

    def flush() -> None:
        nonlocal cur_id, cur_title, cur_rows
        if cur_id is not None:
            sections.append(VideoSection(video_id=cur_id, title=cur_title, rows=tuple(cur_rows)))
        cur_id, cur_title, cur_rows = None, "", []

    for raw in text.splitlines():
        heading = _SECTION_RE.match(raw.strip())
        if heading:
            flush()
            cur_id = heading.group("vid").strip()
            cur_title = heading.group("title").strip()
            continue
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if cur_id is None:
            # Orphan row before any section head — cannot attribute it.
            continue
        system, _sep, correction = raw.partition("\t")
        system = system.strip()
        if not system:
            continue
        cur_rows.append(ProperNounRow(system_term=system, user_correction=correction.strip()))
    flush()
    return ProperNounSheet(sections=tuple(sections))


def render_sheet(sheet: ProperNounSheet) -> str:
    """Render a sheet back to TSV text (header + one block per section)."""
    lines: list[str] = list(_HEADER_LINES)
    for section in sheet.sections:
        lines.append("")
        lines.append(f"## [{section.video_id}] {section.title}".rstrip())
        for row in section.rows:
            lines.append(f"{row.system_term}\t{row.user_correction}")
    return "\n".join(lines) + "\n"


def load_sheet(path: str | Path) -> ProperNounSheet:
    """Read+parse the sheet, or return an empty sheet if the file is absent."""
    file = Path(path)
    if not file.exists():
        return ProperNounSheet()
    return parse_sheet(file.read_text(encoding="utf-8"))


def write_sheet(path: str | Path, sheet: ProperNounSheet) -> None:
    """Render and write the sheet to ``path`` (parent dirs must already exist)."""
    Path(path).write_text(render_sheet(sheet), encoding="utf-8")


def upsert_video_terms(
    sheet: ProperNounSheet, *, video_id: str, title: str, terms: list[str]
) -> ProperNounSheet:
    """Merge newly confirmed ``terms`` into ``video_id``'s section.

    Existing rows keep their user override and order; terms already present
    (folded) are not duplicated; brand-new terms are appended with an empty
    override. The section is created if absent and its title refreshed.
    """
    existing = sheet.section_for(video_id)
    rows: list[ProperNounRow] = list(existing.rows) if existing else []
    seen = {fold_term(r.system_term) for r in rows}
    for term in _dedup_terms(terms):
        key = fold_term(term)
        if key in seen:
            continue
        seen.add(key)
        rows.append(ProperNounRow(system_term=term))
    merged = VideoSection(video_id=video_id, title=title, rows=tuple(rows))

    out: list[VideoSection] = []
    replaced = False
    for section in sheet.sections:
        if section.video_id == video_id:
            out.append(merged)
            replaced = True
        else:
            out.append(section)
    if not replaced:
        out.append(merged)
    return ProperNounSheet(sections=tuple(out))


def known_pairs(sheet: ProperNounSheet) -> list[tuple[str, str]]:
    """Return ``(system_term, resolved)`` for every row, deduped by system term.

    Fed to Stage 01b as the confirmed vocabulary so the model reuses the
    resolved spelling instead of searching the web again.
    """
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for section in sheet.sections:
        for row in section.rows:
            key = fold_term(row.system_term)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append((row.system_term, row.resolved))
    return out


def correction_entries(sheet: ProperNounSheet) -> list[GlossaryEntry]:
    """Build ``GlossaryEntry`` objects from user-corrected rows only.

    Mapping (confirmed with the user): the user's correction is the
    ``canonical`` and the system spelling becomes an ``alias``, so the
    deterministic ``Normalizer`` can rewrite the mis-spelling everywhere. Rows
    sharing a canonical are merged so multiple bad spellings become aliases of
    one entry.
    """
    by_canonical: dict[str, list[str]] = {}
    order: list[str] = []
    for section in sheet.sections:
        for row in section.rows:
            if not row.is_user_corrected:
                continue
            canonical = row.user_correction
            if canonical not in by_canonical:
                by_canonical[canonical] = []
                order.append(canonical)
            aliases = by_canonical[canonical]
            if row.system_term not in aliases:
                aliases.append(row.system_term)
    return [GlossaryEntry(canonical=c, aliases=by_canonical[c]) for c in order]


def correction_glossary(sheet: ProperNounSheet) -> Glossary:
    """A ``Glossary`` of just the user-corrected rows (for Stage 05 rewriting)."""
    return Glossary(entries=tuple(correction_entries(sheet)))


__all__ = [
    "SHEET_FILENAME",
    "ProperNounRow",
    "ProperNounSheet",
    "VideoSection",
    "correction_entries",
    "correction_glossary",
    "known_pairs",
    "load_sheet",
    "parse_sheet",
    "render_sheet",
    "upsert_video_terms",
    "write_sheet",
]
