"""Golden-set behavior for the deterministic proper-noun normalizer.

These are LLM-free, fully reproducible unit tests — the foundation the
whole fidelity chain (Stage 06 evaluator, Stage 02 cleansing) is later
verified against. The canonical motivating case is the
``"ビブコーディング" → "Vibe Coding"`` mis-transcription.
"""

from __future__ import annotations

import pytest

from pipeline_youtube.glossary.normalizer import Normalizer
from pipeline_youtube.glossary.schema import (
    Glossary,
    GlossaryConflictError,
    GlossaryEntry,
)


def _normalizer(*entries: GlossaryEntry) -> Normalizer:
    return Normalizer(Glossary(entries=tuple(entries)))


_VIBE = GlossaryEntry(
    canonical="Vibe Coding",
    aliases=["ビブコーディング", "バイブコーディング"],
)


def test_alias_resolves_to_canonical() -> None:
    norm = _normalizer(_VIBE)
    assert norm.normalize("ビブコーディング") == "Vibe Coding"
    assert norm.normalize("バイブコーディング") == "Vibe Coding"


def test_canonical_is_idempotent() -> None:
    norm = _normalizer(_VIBE)
    assert norm.normalize("Vibe Coding") == "Vibe Coding"


def test_matching_is_width_and_case_insensitive() -> None:
    norm = _normalizer(_VIBE)
    # Full-width latin (NFKC-folded) and arbitrary casing both resolve.
    assert norm.normalize("Ｖｉｂｅ　Ｃｏｄｉｎｇ") == "Vibe Coding"
    assert norm.normalize("VIBE CODING") == "Vibe Coding"
    assert norm.normalize("  vibe coding  ") == "Vibe Coding"


def test_unknown_term_passes_through_unchanged() -> None:
    norm = _normalizer(_VIBE)
    assert norm.normalize("Obsidian") == "Obsidian"
    assert norm.normalize("") == ""


def test_returned_canonical_is_verbatim_not_folded() -> None:
    # Folding builds only the key; the output keeps the glossary spelling.
    norm = _normalizer(_VIBE)
    assert norm.normalize("ビブコーディング") == "Vibe Coding"  # not "vibe coding"


def test_canonical_for_distinguishes_known_from_unknown() -> None:
    norm = _normalizer(_VIBE)
    assert norm.canonical_for("ビブコーディング") == "Vibe Coding"
    assert norm.canonical_for("Vibe Coding") == "Vibe Coding"
    assert norm.canonical_for("Unrelated") is None


def test_is_known_and_is_variant_semantics() -> None:
    norm = _normalizer(_VIBE)
    # alias: known AND a variant (mis-transcription to be rewritten)
    assert norm.is_known("ビブコーディング") is True
    assert norm.is_variant("ビブコーディング") is True
    # canonical: known but NOT a variant
    assert norm.is_known("Vibe Coding") is True
    assert norm.is_variant("Vibe Coding") is False
    # unknown: neither
    assert norm.is_known("Obsidian") is False
    assert norm.is_variant("Obsidian") is False


def test_canonical_listed_in_own_aliases_is_not_a_conflict() -> None:
    # Re-registering the same (key -> same canonical) pair is harmless.
    entry = GlossaryEntry(canonical="Vibe Coding", aliases=["Vibe Coding", "ビブコーディング"])
    norm = _normalizer(entry)
    assert norm.normalize("Vibe Coding") == "Vibe Coding"
    assert norm.normalize("ビブコーディング") == "Vibe Coding"


def test_conflicting_variant_raises_at_build_time() -> None:
    a = GlossaryEntry(canonical="Vibe Coding", aliases=["バイブコーディング"])
    b = GlossaryEntry(canonical="Bibe Coding", aliases=["バイブコーディング"])
    with pytest.raises(GlossaryConflictError, match="maps to both"):
        _normalizer(a, b)


def test_case_only_conflict_is_detected_via_folded_key() -> None:
    # Two canonicals that fold to the same key are an integrity error.
    a = GlossaryEntry(canonical="Vibe Coding")
    b = GlossaryEntry(canonical="vibe coding")
    with pytest.raises(GlossaryConflictError):
        _normalizer(a, b)


def test_len_counts_distinct_keys() -> None:
    norm = _normalizer(_VIBE)
    # 1 canonical + 2 aliases = 3 distinct folded keys
    assert len(norm) == 3
