"""Data structures and strict JSON parsing for the proper-noun glossary.

Idiom note: this repo is Pydantic-first *in spirit* but ships no
``pydantic`` dependency — the actual local idiom (``synthesis/scoring.py``,
``evaluation/schemas.py``) is ``@dataclass(frozen=True)`` + stdlib
``json``. The glossary follows that idiom.

Parsing contract (deliberately the opposite of the LLM-output parsers in
``synthesis.scoring``): the glossary is a **first-party owned artifact**,
not model output. A malformed payload is a build/data bug we want to fail
loudly, so ``parse_glossary`` raises ``GlossaryParseError`` instead of
defensively swallowing bad entries. Internal inconsistency (one variant
key resolving to two different canonicals) is surfaced later, at
``Normalizer`` build time, as ``GlossaryConflictError``.

On-disk JSON shape::

    {
      "version": 1,
      "entries": [
        {"canonical": "Vibe Coding",
         "aliases": ["ビブコーディング", "バイブコーディング"],
         "reading": "ヴァイブコーディング",
         "category": "concept"}
      ]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..domain.errors import GlossaryConflictError as GlossaryConflictError
from ..domain.errors import GlossaryParseError as GlossaryParseError
from .fold import fold_term


@dataclass(frozen=True)
class GlossaryEntry:
    """One canonical proper noun plus its known variant spellings.

    ``aliases`` are the misrecognized / alternate forms that should be
    rewritten to ``canonical`` (e.g. ``"ビブコーディング"`` for
    ``"Vibe Coding"``). ``reading`` (yomi) and ``category`` are advisory
    metadata used by downstream evaluators; they never affect matching.
    """

    canonical: str
    aliases: list[str] = field(default_factory=list)
    reading: str = ""
    category: str = ""


@dataclass(frozen=True)
class Glossary:
    """An immutable collection of glossary entries plus a format version."""

    entries: tuple[GlossaryEntry, ...] = ()
    version: int = 1


def parse_glossary(data: object) -> Glossary:
    """Strictly parse a decoded JSON object into a ``Glossary``.

    Raises ``GlossaryParseError`` on any structural defect: non-dict
    root, non-list ``entries``, non-dict entry, or an entry whose
    ``canonical`` is missing/empty. ``aliases`` defaults to ``[]`` and
    must be a list of non-empty strings when present.
    """
    if not isinstance(data, dict):
        raise GlossaryParseError(f"glossary root must be a JSON object, got {type(data).__name__}")

    unknown_root = set(data) - {"version", "entries"}
    if unknown_root:
        raise GlossaryParseError(f"unknown top-level keys: {sorted(unknown_root)!r}")

    version_raw = data.get("version", 1)
    if not isinstance(version_raw, int) or isinstance(version_raw, bool):
        raise GlossaryParseError(f"version must be an integer, got {version_raw!r}")

    entries_raw = data.get("entries", [])
    if not isinstance(entries_raw, list):
        raise GlossaryParseError(f"entries must be a list, got {type(entries_raw).__name__}")

    entries: list[GlossaryEntry] = []
    for i, item in enumerate(entries_raw):
        entries.append(_parse_entry(item, index=i))

    return Glossary(entries=tuple(entries), version=version_raw)


def _parse_entry(item: object, *, index: int) -> GlossaryEntry:
    """Parse and validate one entry object; raise on any defect."""
    if not isinstance(item, dict):
        raise GlossaryParseError(f"entries[{index}] must be an object, got {type(item).__name__}")

    unknown_keys = set(item) - {"canonical", "aliases", "reading", "category"}
    if unknown_keys:
        raise GlossaryParseError(f"entries[{index}] has unknown keys: {sorted(unknown_keys)!r}")

    canonical = item.get("canonical")
    if not isinstance(canonical, str) or not canonical.strip():
        raise GlossaryParseError(f"entries[{index}].canonical must be a non-empty string")

    aliases = _parse_str_list(item.get("aliases", []), index=index, field_name="aliases")

    reading = item.get("reading", "")
    if not isinstance(reading, str):
        raise GlossaryParseError(f"entries[{index}].reading must be a string")

    category = item.get("category", "")
    if not isinstance(category, str):
        raise GlossaryParseError(f"entries[{index}].category must be a string")

    return GlossaryEntry(
        canonical=canonical,
        aliases=aliases,
        reading=reading,
        category=category,
    )


def _parse_str_list(value: object, *, index: int, field_name: str) -> list[str]:
    """Coerce a JSON value into a list of non-empty strings, strictly."""
    if not isinstance(value, list):
        raise GlossaryParseError(
            f"entries[{index}].{field_name} must be a list, got {type(value).__name__}"
        )
    result: list[str] = []
    for j, element in enumerate(value):
        if not isinstance(element, str) or not element.strip():
            raise GlossaryParseError(
                f"entries[{index}].{field_name}[{j}] must be a non-empty string"
            )
        result.append(element)
    return result


def load_glossary(path: str | Path) -> Glossary:
    """Read and strictly parse a glossary JSON file from disk.

    Raises ``GlossaryParseError`` if the file is not valid JSON or does
    not satisfy the glossary schema. The JSON decode error is chained so
    the original position info is preserved (errors-are-treasure).
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GlossaryParseError(f"glossary file is not valid JSON: {exc}") from exc
    return parse_glossary(data)


def dump_glossary(glossary: Glossary) -> dict[str, Any]:
    """Serialize a ``Glossary`` to the on-disk JSON shape.

    ``reading``/``category`` are emitted only when non-empty so promoted
    entries (which carry neither) stay terse.
    """
    entries: list[dict[str, Any]] = []
    for entry in glossary.entries:
        item: dict[str, Any] = {"canonical": entry.canonical}
        if entry.aliases:
            item["aliases"] = list(entry.aliases)
        if entry.reading:
            item["reading"] = entry.reading
        if entry.category:
            item["category"] = entry.category
        entries.append(item)
    return {"version": glossary.version, "entries": entries}


def write_glossary(path: str | Path, glossary: Glossary) -> None:
    """Write ``glossary`` to ``path`` as pretty UTF-8 JSON (CJK kept verbatim)."""
    text = json.dumps(dump_glossary(glossary), ensure_ascii=False, indent=2)
    Path(path).write_text(text + "\n", encoding="utf-8")


def _fold(term: str) -> str:
    return fold_term(term)


def merge_glossary(base: Glossary, new_entries: list[GlossaryEntry]) -> Glossary:
    """Non-destructively merge ``new_entries`` into ``base``.

    Existing entries are preserved verbatim and gain any genuinely new aliases;
    new canonicals are appended. The merge is **conflict-tolerant**: a canonical
    or alias whose folded spelling already belongs to a *different* canonical is
    skipped rather than added, so the result always builds a valid
    ``Normalizer`` (no ``GlossaryConflictError``). Returns the merged glossary;
    when nothing changed it is value-equal to ``base``.
    """
    canonicals: list[str] = [e.canonical for e in base.entries]
    aliases_by_canonical: dict[str, list[str]] = {
        e.canonical: list(e.aliases) for e in base.entries
    }
    meta_by_canonical: dict[str, GlossaryEntry] = {e.canonical: e for e in base.entries}

    # folded surface -> owning canonical, across every key currently in the set.
    owner: dict[str, str] = {}
    for entry in base.entries:
        for surface in (entry.canonical, *entry.aliases):
            key = _fold(surface)
            if key:
                owner.setdefault(key, entry.canonical)

    for entry in new_entries:
        canonical = entry.canonical
        ckey = _fold(canonical)
        if not ckey:
            continue
        owned = owner.get(ckey)
        if owned is not None and owned != canonical:
            # Spelling already belongs to a different canonical — skip the entry.
            continue
        if owned is None:
            canonicals.append(canonical)
            aliases_by_canonical[canonical] = []
            meta_by_canonical[canonical] = entry
            owner[ckey] = canonical
        for alias in entry.aliases:
            akey = _fold(alias)
            if not akey:
                continue
            if akey in owner:
                continue  # duplicate or conflicting alias — leave as-is
            aliases_by_canonical[canonical].append(alias)
            owner[akey] = canonical

    merged = tuple(
        GlossaryEntry(
            canonical=c,
            aliases=aliases_by_canonical[c],
            reading=meta_by_canonical[c].reading if c in meta_by_canonical else "",
            category=meta_by_canonical[c].category if c in meta_by_canonical else "",
        )
        for c in canonicals
    )
    return Glossary(entries=merged, version=base.version)


__all__ = [
    "Glossary",
    "GlossaryConflictError",
    "GlossaryEntry",
    "GlossaryParseError",
    "dump_glossary",
    "load_glossary",
    "merge_glossary",
    "parse_glossary",
    "write_glossary",
]
