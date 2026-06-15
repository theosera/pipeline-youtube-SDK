"""Strict-parsing behavior for the proper-noun glossary schema.

The glossary is first-party data, so parsing must FAIL LOUDLY on any
structural defect rather than defensively drop entries (the opposite of
the advisory LLM-output parsers in ``synthesis.scoring``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline_youtube.glossary.schema import (
    Glossary,
    GlossaryEntry,
    GlossaryParseError,
    load_glossary,
    parse_glossary,
)

_VALID = {
    "version": 1,
    "entries": [
        {
            "canonical": "Vibe Coding",
            "aliases": ["ビブコーディング", "バイブコーディング"],
            "reading": "ヴァイブコーディング",
            "category": "concept",
        }
    ],
}


def test_parse_valid_returns_glossary() -> None:
    glossary = parse_glossary(_VALID)
    assert isinstance(glossary, Glossary)
    assert glossary.version == 1
    assert len(glossary.entries) == 1
    entry = glossary.entries[0]
    assert isinstance(entry, GlossaryEntry)
    assert entry.canonical == "Vibe Coding"
    assert entry.aliases == ["ビブコーディング", "バイブコーディング"]
    assert entry.reading == "ヴァイブコーディング"
    assert entry.category == "concept"


def test_entries_is_tuple_for_immutability() -> None:
    glossary = parse_glossary(_VALID)
    assert isinstance(glossary.entries, tuple)


def test_defaults_when_optional_fields_absent() -> None:
    glossary = parse_glossary({"entries": [{"canonical": "Obsidian"}]})
    entry = glossary.entries[0]
    assert entry.aliases == []
    assert entry.reading == ""
    assert entry.category == ""
    assert glossary.version == 1  # default when "version" omitted


def test_empty_entries_is_valid() -> None:
    glossary = parse_glossary({"entries": []})
    assert glossary.entries == ()


def test_root_must_be_object() -> None:
    with pytest.raises(GlossaryParseError, match="root must be a JSON object"):
        parse_glossary([{"canonical": "x"}])


def test_entries_must_be_list() -> None:
    with pytest.raises(GlossaryParseError, match="entries must be a list"):
        parse_glossary({"entries": {"canonical": "x"}})


def test_entry_must_be_object() -> None:
    with pytest.raises(GlossaryParseError, match=r"entries\[0\] must be an object"):
        parse_glossary({"entries": ["Vibe Coding"]})


def test_missing_canonical_raises() -> None:
    with pytest.raises(GlossaryParseError, match=r"entries\[0\].canonical"):
        parse_glossary({"entries": [{"aliases": ["x"]}]})


def test_blank_canonical_raises() -> None:
    with pytest.raises(GlossaryParseError, match=r"entries\[0\].canonical"):
        parse_glossary({"entries": [{"canonical": "   "}]})


def test_alias_must_be_non_empty_string() -> None:
    with pytest.raises(GlossaryParseError, match=r"aliases\[1\]"):
        parse_glossary({"entries": [{"canonical": "x", "aliases": ["ok", ""]}]})


def test_aliases_must_be_list() -> None:
    with pytest.raises(GlossaryParseError, match="aliases must be a list"):
        parse_glossary({"entries": [{"canonical": "x", "aliases": "ビブコーディング"}]})


def test_bool_version_rejected() -> None:
    # bool is an int subclass; the schema must not accept True as version 1.
    with pytest.raises(GlossaryParseError, match="version must be an integer"):
        parse_glossary({"version": True, "entries": []})


def test_load_glossary_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "glossary.json"
    path.write_text(json.dumps(_VALID, ensure_ascii=False), encoding="utf-8")
    glossary = load_glossary(path)
    assert glossary.entries[0].canonical == "Vibe Coding"


def test_load_glossary_invalid_json_chains_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(GlossaryParseError, match="not valid JSON") as exc_info:
        load_glossary(path)
    assert exc_info.value.__cause__ is not None  # original decode error preserved
