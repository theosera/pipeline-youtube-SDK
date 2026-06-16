"""Deterministic, pure proper-noun normalizer (no LLM).

``Normalizer`` builds an immutable variant→canonical index from a
``Glossary`` once, then resolves terms in O(1). It is the deterministic
oracle the whole fidelity chain is verified against, so it has two
hard properties:

1. **Idempotent / non-destructive** — a canonical term resolves to
   itself, and an *unknown* term passes through unchanged. The
   normalizer never invents a canonical that is not already in the
   glossary (matches the pipeline's "no hallucination" rule).
2. **Conflict-detecting** — if two entries map the same variant to two
   different canonicals, the index cannot be trusted, so construction
   raises ``GlossaryConflictError`` instead of silently picking one.

Matching is width/case-insensitive: keys are folded with Unicode NFKC
+ ``casefold`` + strip, so half/full-width and upper/lower variants of
the same spelling resolve identically (``"Ｖｉｂｅ Ｃｏｄｉｎｇ"`` →
``"Vibe Coding"``). Folding only builds the lookup *key*; the returned
canonical is always the glossary's verbatim spelling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .fold import fold_term
from .schema import GlossaryConflictError

if TYPE_CHECKING:
    from .schema import Glossary, GlossaryEntry


class Normalizer:
    """Resolves proper-noun variants to their canonical spelling.

    Construct from a parsed ``Glossary``. Both each entry's ``canonical``
    and every ``alias`` become lookup keys pointing at that canonical, so
    canonicals are idempotent and aliases are rewritten.
    """

    def __init__(self, glossary: Glossary) -> None:
        self._index: dict[str, str] = {}
        for entry in glossary.entries:
            self._register(entry)

    def _register(self, entry: GlossaryEntry) -> None:
        """Index an entry's canonical and aliases; raise on conflict.

        A key that already resolves to a *different* canonical is a data
        integrity violation (``GlossaryConflictError``). Re-registering
        the same (key → same canonical) pair is harmless and ignored, so
        a canonical that also appears in its own alias list is fine.
        """
        for surface in (entry.canonical, *entry.aliases):
            key = fold_term(surface)
            if not key:
                continue
            existing = self._index.get(key)
            if existing is not None and existing != entry.canonical:
                raise GlossaryConflictError(
                    f"variant {surface!r} (key {key!r}) maps to both "
                    f"{existing!r} and {entry.canonical!r}"
                )
            self._index[key] = entry.canonical

    def normalize(self, term: str) -> str:
        """Return the canonical spelling for ``term``, else ``term`` unchanged.

        Non-destructive: unknown terms are returned verbatim so the
        normalizer can be applied anywhere without risk of inventing
        content.
        """
        return self._index.get(fold_term(term), term)

    def canonical_for(self, term: str) -> str | None:
        """Return the canonical for ``term`` if known, else ``None``.

        Unlike ``normalize``, this distinguishes "known and already
        canonical / aliased" from "unknown" — useful for the fidelity
        evaluator, which must tell whether a term is in scope at all.
        """
        return self._index.get(fold_term(term))

    def is_known(self, term: str) -> bool:
        """True iff ``term`` (folded) appears in the glossary."""
        return fold_term(term) in self._index

    def is_variant(self, term: str) -> bool:
        """True iff ``term`` is a known *non-canonical* spelling.

        This is the fidelity signal: a term that is known but whose
        canonical differs from the term itself is a mis-transcription the
        evaluator should flag and Stage 02 should rewrite.
        """
        canonical = self.canonical_for(term)
        return canonical is not None and canonical != term

    def __len__(self) -> int:
        """Number of distinct lookup keys (canonicals + aliases)."""
        return len(self._index)


__all__ = ["Normalizer", "fold_term"]
