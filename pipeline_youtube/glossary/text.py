"""Deterministic text-level proper-noun normalization (no LLM).

The inverse of ``evaluation.fidelity.scan_fidelity``: where the fidelity
scan *detects* known mis-transcriptions, ``normalize_text`` *rewrites*
them to the canonical spelling (``"ビブコーディングを学ぶ"`` →
``"Vibe Codingを学ぶ"``). This is what Stage 02 applies to its cleansed
output so the "no hallucination" rule is preserved — only spellings that
already exist in the glossary are ever substituted; unknown text is left
verbatim.

Matching uses a single longest-surface-first alternation regex with
``re.IGNORECASE`` over the original text (positions preserved), and each
match is resolved through ``Normalizer`` so detection (Phase B) and
rewriting (Phase C) share one canonical-resolution path. Constructing the
``Normalizer`` also revalidates the glossary, so a conflicting glossary
raises ``GlossaryConflictError`` here exactly as it does for the scanner.

Scope note: variants that differ from their canonical only by case/width
are intentionally NOT rewritten (they fold to the canonical and are not
defects). Width-divergent occurrences in the body (e.g. half-width
katakana) are left to the fidelity scan's folded detection rather than
risking position-shifting whole-text NFKC rewrites here.
"""

from __future__ import annotations

import re

from .normalizer import Normalizer, fold_term
from .schema import Glossary


def variant_surfaces(glossary: Glossary) -> list[str]:
    """Return the deduped variant spellings worth rewriting/detecting.

    A surface qualifies when its folded form is non-empty and differs
    from its entry's canonical folded form (i.e. it is a genuine
    mis-spelling, not a case/width restatement of the canonical).
    Insertion order is preserved for determinism.
    """
    seen: set[str] = set()
    surfaces: list[str] = []
    for entry in glossary.entries:
        canonical_fold = fold_term(entry.canonical)
        for alias in entry.aliases:
            folded = fold_term(alias)
            if not folded or folded == canonical_fold or alias in seen:
                continue
            seen.add(alias)
            surfaces.append(alias)
    return surfaces


# Only ASCII alphanumerics (+ underscore) can form a larger corrupted word
# around an ASCII alias; CJK neighbors cannot. Guards therefore reject only
# adjacent ASCII word chars, NOT Python's Unicode ``\b`` (which counts kana
# as word chars and would block legitimate CJK-adjacent matches such as the
# trailing digit in "へんかん0へんかん1").
_ASCII_WORD = "A-Za-z0-9_"


def _is_ascii_word_char(ch: str) -> bool:
    return bool(ch) and ch.isascii() and (ch.isalnum() or ch == "_")


def _bounded(surface: str) -> str:
    """Escape ``surface`` and guard its ASCII-word edges with lookarounds.

    Short ASCII aliases (``"AI"``, ``"Go"``) must NOT match inside larger
    ASCII words (``"said"``, ``"Google"``) — raw substring replacement
    there corrupts prose. A negative ASCII-word lookbehind/lookahead is
    added only when the corresponding edge character is itself ASCII, so a
    CJK alias (or an ASCII-ending alias sitting next to kana) still matches
    by substring.
    """
    esc = re.escape(surface)
    lead = rf"(?<![{_ASCII_WORD}])" if _is_ascii_word_char(surface[:1]) else ""
    trail = rf"(?![{_ASCII_WORD}])" if _is_ascii_word_char(surface[-1:]) else ""
    return f"{lead}{esc}{trail}"


def compile_variant_pattern(surfaces: list[str]) -> re.Pattern[str] | None:
    """Compile the shared variant matcher used by BOTH the rewriter and scanner.

    Longest surfaces first (so an overlapping shorter variant never
    partially clobbers a longer one), ``re.IGNORECASE``, with per-surface
    ASCII word-boundary guards (see ``_bounded``). Returns ``None`` when
    there are no variants. Keeping detection (``evaluation.fidelity``) and
    rewriting (``normalize_text``) on the *same* pattern guarantees they
    agree on what counts as a match.
    """
    if not surfaces:
        return None
    ordered = sorted(surfaces, key=len, reverse=True)
    return re.compile("|".join(_bounded(s) for s in ordered), re.IGNORECASE)


def normalize_text(text: str, glossary: Glossary) -> str:
    """Rewrite every known variant spelling in ``text`` to its canonical.

    Non-destructive: only glossary-known variants are touched (with ASCII
    word-boundary guards so short aliases never corrupt larger words); all
    other text is returned unchanged. Idempotent for a conflict-free
    glossary whose canonicals are not themselves variants of other
    entries.

    Raises ``GlossaryConflictError`` (via ``Normalizer``) if the glossary
    maps one variant to two canonicals.
    """
    normalizer = Normalizer(glossary)  # revalidate + single resolution path
    pattern = compile_variant_pattern(variant_surfaces(glossary))
    if pattern is None:
        return text
    return pattern.sub(lambda m: normalizer.normalize(m.group(0)), text)


__all__ = ["compile_variant_pattern", "normalize_text", "variant_surfaces"]
