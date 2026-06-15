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


def normalize_text(text: str, glossary: Glossary) -> str:
    """Rewrite every known variant spelling in ``text`` to its canonical.

    Non-destructive: only glossary-known variants are touched; all other
    text is returned unchanged. Longest surfaces are matched first so an
    overlapping shorter variant never partially clobbers a longer one.
    Idempotent for a conflict-free glossary whose canonicals are not
    themselves variants of other entries.

    Raises ``GlossaryConflictError`` (via ``Normalizer``) if the glossary
    maps one variant to two canonicals.
    """
    normalizer = Normalizer(glossary)  # revalidate + single resolution path
    surfaces = variant_surfaces(glossary)
    if not surfaces:
        return text
    surfaces.sort(key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(s) for s in surfaces), re.IGNORECASE)
    return pattern.sub(lambda m: normalizer.normalize(m.group(0)), text)


__all__ = ["normalize_text", "variant_surfaces"]
