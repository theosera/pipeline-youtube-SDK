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


class GlossaryParseError(ValueError):
    """Raised when glossary JSON is structurally malformed.

    Strict by design: unlike the advisory LLM-output parsers, the
    glossary is first-party data, so a bad payload should fail the build
    rather than be silently dropped.
    """


class GlossaryConflictError(ValueError):
    """Raised when two entries claim the same variant for different canonicals.

    Surfaced at ``Normalizer`` build time (see ``glossary.normalizer``).
    Carries the offending key so the data error is actionable.
    """


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


__all__ = [
    "Glossary",
    "GlossaryConflictError",
    "GlossaryEntry",
    "GlossaryParseError",
    "load_glossary",
    "parse_glossary",
]
