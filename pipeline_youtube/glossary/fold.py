"""Leaf primitive for the glossary package: term folding.

This module deliberately has **no intra-package imports** so it can sit at
the bottom of the glossary dependency graph. ``schema`` and ``normalizer``
both need ``fold_term`` but ``normalizer`` already imports ``schema`` for
its error type; keeping the fold primitive here breaks what would otherwise
be a ``schema`` ↔ ``normalizer`` import cycle (flagged by CodeQL).
"""

from __future__ import annotations

import unicodedata


def fold_term(term: str) -> str:
    """Fold a term into its match key (NFKC + casefold + strip).

    NFKC normalizes half/full-width forms; ``casefold`` removes case
    distinctions. The result is used only for lookup, never returned.
    """
    return unicodedata.normalize("NFKC", term).casefold().strip()


__all__ = ["fold_term"]
