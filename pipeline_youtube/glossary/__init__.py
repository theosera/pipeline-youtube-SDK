"""Proper-noun normalization glossary (Phase A foundation).

A deterministic, cross-playlist dictionary that maps misrecognized /
mis-transcribed proper-noun variants to their canonical form
(e.g. ``"ビブコーディング" → "Vibe Coding"``). This is the *source of
truth* the downstream fidelity layers verify against:

- Stage 02 cleansing applies it to normalize proper nouns without
  hallucinating (the canonical form must already exist in the glossary).
- The Stage 06 fidelity evaluator uses it as an oracle to flag residual
  mis-transcriptions as ``high`` findings.

This package is intentionally LLM-free: the schema is our own owned
artifact (not model output), so parsing is **strict** (malformed input
raises) and lookups are **pure** (``Normalizer`` builds an immutable
index once, then resolves terms in O(1)). That makes it the easiest
component in the fidelity chain to verify in isolation.
"""

from __future__ import annotations

from .normalizer import Normalizer, fold_term
from .schema import (
    Glossary,
    GlossaryConflictError,
    GlossaryEntry,
    GlossaryParseError,
    load_glossary,
    parse_glossary,
)
from .text import normalize_text, variant_surfaces

__all__ = [
    "Glossary",
    "GlossaryConflictError",
    "GlossaryEntry",
    "GlossaryParseError",
    "Normalizer",
    "fold_term",
    "load_glossary",
    "normalize_text",
    "parse_glossary",
    "variant_surfaces",
]
