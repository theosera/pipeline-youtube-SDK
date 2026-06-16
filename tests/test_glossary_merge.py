"""Tests for glossary serialization + conflict-tolerant merge (schema.py)."""

from __future__ import annotations

from pathlib import Path

from pipeline_youtube.glossary.normalizer import Normalizer
from pipeline_youtube.glossary.schema import (
    Glossary,
    GlossaryEntry,
    dump_glossary,
    load_glossary,
    merge_glossary,
    write_glossary,
)


class TestDump:
    def test_omits_empty_optional_fields(self) -> None:
        g = Glossary(entries=(GlossaryEntry(canonical="Google", aliases=["ぐぐる"]),))
        assert dump_glossary(g) == {
            "version": 1,
            "entries": [{"canonical": "Google", "aliases": ["ぐぐる"]}],
        }

    def test_includes_reading_and_category(self) -> None:
        g = Glossary(
            entries=(GlossaryEntry(canonical="X", reading="えっくす", category="concept"),)
        )
        item = dump_glossary(g)["entries"][0]
        assert item == {"canonical": "X", "reading": "えっくす", "category": "concept"}


class TestMerge:
    def test_appends_new_entry(self) -> None:
        base = Glossary(entries=(GlossaryEntry(canonical="A"),))
        merged = merge_glossary(base, [GlossaryEntry(canonical="Google", aliases=["ぐぐる"])])
        assert [e.canonical for e in merged.entries] == ["A", "Google"]

    def test_unions_aliases_into_existing_canonical(self) -> None:
        base = Glossary(entries=(GlossaryEntry(canonical="Google", aliases=["ぐぐる"]),))
        merged = merge_glossary(base, [GlossaryEntry(canonical="Google", aliases=["グーグル"])])
        assert merged.entries[0].aliases == ["ぐぐる", "グーグル"]

    def test_preserves_existing_metadata(self) -> None:
        base = Glossary(
            entries=(GlossaryEntry(canonical="Google", reading="ぐーぐる", category="company"),)
        )
        merged = merge_glossary(base, [GlossaryEntry(canonical="Google", aliases=["ぐぐる"])])
        assert merged.entries[0].reading == "ぐーぐる"
        assert merged.entries[0].category == "company"
        assert merged.entries[0].aliases == ["ぐぐる"]

    def test_skips_conflicting_canonical(self) -> None:
        # "ぐぐる" is already an alias of Google; an entry trying to make it a
        # canonical must be skipped so the merged glossary stays loadable.
        base = Glossary(entries=(GlossaryEntry(canonical="Google", aliases=["ぐぐる"]),))
        merged = merge_glossary(base, [GlossaryEntry(canonical="ぐぐる", aliases=["x"])])
        assert [e.canonical for e in merged.entries] == ["Google"]
        Normalizer(merged)  # must not raise GlossaryConflictError

    def test_skips_conflicting_alias(self) -> None:
        base = Glossary(entries=(GlossaryEntry(canonical="Google", aliases=["ぐぐる"]),))
        # New entry tries to claim "ぐぐる" as an alias of a different canonical.
        merged = merge_glossary(base, [GlossaryEntry(canonical="Bing", aliases=["ぐぐる"])])
        bing = next(e for e in merged.entries if e.canonical == "Bing")
        assert bing.aliases == []
        Normalizer(merged)

    def test_no_change_is_value_equal(self) -> None:
        base = Glossary(entries=(GlossaryEntry(canonical="Google", aliases=["ぐぐる"]),))
        assert merge_glossary(base, [GlossaryEntry(canonical="Google", aliases=["ぐぐる"])]) == base


class TestWriteRoundtrip:
    def test_write_then_load(self, tmp_path: Path) -> None:
        g = Glossary(
            entries=(GlossaryEntry(canonical="Vibe Coding", aliases=["ビブコーディング"]),)
        )
        path = tmp_path / "glossary.json"
        write_glossary(path, g)
        assert load_glossary(path) == g
