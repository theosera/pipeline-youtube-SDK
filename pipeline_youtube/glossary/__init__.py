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
from .proper_nouns import (
    SHEET_FILENAME,
    ProperNounRow,
    ProperNounSheet,
    VideoSection,
    correction_entries,
    correction_glossary,
    known_pairs,
    load_sheet,
    parse_sheet,
    render_sheet,
    upsert_video_terms,
    write_sheet,
)
from .schema import (
    Glossary,
    GlossaryConflictError,
    GlossaryEntry,
    GlossaryParseError,
    dump_glossary,
    load_glossary,
    merge_glossary,
    parse_glossary,
    write_glossary,
)
from .text import normalize_text, variant_surfaces

__all__ = [
    "SHEET_FILENAME",
    "Glossary",
    "GlossaryConflictError",
    "GlossaryEntry",
    "GlossaryParseError",
    "Normalizer",
    "ProperNounRow",
    "ProperNounSheet",
    "VideoSection",
    "correction_entries",
    "correction_glossary",
    "dump_glossary",
    "fold_term",
    "known_pairs",
    "load_glossary",
    "load_sheet",
    "merge_glossary",
    "normalize_text",
    "parse_glossary",
    "parse_sheet",
    "render_sheet",
    "upsert_video_terms",
    "variant_surfaces",
    "write_glossary",
    "write_sheet",
]
