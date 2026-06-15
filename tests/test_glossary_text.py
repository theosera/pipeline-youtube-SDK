"""Golden-set behavior for deterministic text normalization (Phase C).

``normalize_text`` is what Stage 02 applies to rewrite known proper-noun
mis-transcriptions to canonical form without hallucinating. The detector
(Phase B) and this rewriter must agree, so these tests pin the same
``"ビブコーディング" → "Vibe Coding"`` oracle behavior.
"""

from __future__ import annotations

import pytest

from pipeline_youtube.glossary.schema import Glossary, GlossaryConflictError, GlossaryEntry
from pipeline_youtube.glossary.text import normalize_text, variant_surfaces

_VIBE = GlossaryEntry(canonical="Vibe Coding", aliases=["ビブコーディング", "バイブコーディング"])
_GLOSSARY = Glossary(entries=(_VIBE,))


def test_rewrites_variant_to_canonical_in_context() -> None:
    assert normalize_text("本日はビブコーディングを学ぶ", _GLOSSARY) == "本日はVibe Codingを学ぶ"


def test_rewrites_all_occurrences_and_all_aliases() -> None:
    text = "ビブコーディングとバイブコーディング、再びビブコーディング"
    expected = "Vibe CodingとVibe Coding、再びVibe Coding"
    assert normalize_text(text, _GLOSSARY) == expected


def test_canonical_text_is_unchanged() -> None:
    assert normalize_text("Vibe Coding は良い", _GLOSSARY) == "Vibe Coding は良い"


def test_unknown_text_is_left_verbatim() -> None:
    assert normalize_text("無関係なObsidianの話", _GLOSSARY) == "無関係なObsidianの話"


def test_empty_or_variantless_glossary_is_noop() -> None:
    assert normalize_text("何でも", Glossary()) == "何でも"
    only_canonical = Glossary(entries=(GlossaryEntry(canonical="Obsidian"),))
    assert normalize_text("Obsidian は良い", only_canonical) == "Obsidian は良い"


def test_case_insensitive_latin_variant_rewrites() -> None:
    glossary = Glossary(entries=(GlossaryEntry(canonical="Vibe Coding", aliases=["vibecoding"]),))
    assert normalize_text("I use VibeCoding daily", glossary) == "I use Vibe Coding daily"


def test_longest_surface_matches_first() -> None:
    # "Vibe Coding" must win over a shorter overlapping variant so the
    # replacement is whole, not partial.
    glossary = Glossary(
        entries=(
            GlossaryEntry(canonical="Vibe Coding", aliases=["ビブコーディング"]),
            GlossaryEntry(canonical="Coding", aliases=["コーディング"]),
        )
    )
    # The long katakana variant is rewritten as one unit.
    assert normalize_text("ビブコーディング入門", glossary) == "Vibe Coding入門"


def test_ascii_alias_does_not_corrupt_substrings() -> None:
    # Short ASCII aliases must use word boundaries so they never rewrite
    # letters inside unrelated words (the "AI" in "said"/"maintain").
    glossary = Glossary(
        entries=(GlossaryEntry(canonical="Artificial Intelligence", aliases=["AI"]),)
    )
    assert normalize_text("He said we maintain it", glossary) == "He said we maintain it"
    assert normalize_text("Use AI now", glossary) == "Use Artificial Intelligence now"


def test_normalization_is_idempotent() -> None:
    once = normalize_text("ビブコーディングとバイブコーディング", _GLOSSARY)
    assert normalize_text(once, _GLOSSARY) == once


def test_conflicting_glossary_raises() -> None:
    a = GlossaryEntry(canonical="Vibe Coding", aliases=["バイブコーディング"])
    b = GlossaryEntry(canonical="Bibe Coding", aliases=["バイブコーディング"])
    with pytest.raises(GlossaryConflictError):
        normalize_text("バイブコーディング", Glossary(entries=(a, b)))


def test_variant_surfaces_excludes_case_or_width_only_aliases() -> None:
    glossary = Glossary(
        entries=(
            GlossaryEntry(canonical="Vibe Coding", aliases=["VIBE CODING", "ビブコーディング"]),
        )
    )
    # "VIBE CODING" folds to the canonical -> excluded; only the real variant remains.
    assert variant_surfaces(glossary) == ["ビブコーディング"]
