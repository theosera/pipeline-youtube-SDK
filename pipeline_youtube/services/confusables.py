"""Filename concealment defense: invisible-char stripping + confusable detection.

YouTube video/playlist titles are attacker-controllable external text that
flows verbatim into on-disk note filenames, folder names, and YAML
frontmatter (see ``services/obsidian.py``). Two concealment classes matter
for a filename:

  1. **Invisible / bidi controls** -- zero-width joiners, the RIGHT-TO-LEFT
     OVERRIDE (U+202E), directional isolates, BOM, interlinear anchors.
     These have no legitimate use in a filename and can visually reverse or
     hide part of a name (a U+202E before ``gpj.exe`` renders as ``exe.jpg``).
     They are **stripped unconditionally** -- zero false positives on real
     titles because a genuine title never needs them.
  2. **Mixed-script confusables** -- a single word carrying both a Latin
     letter and a Cyrillic/Greek look-alike (e.g. a word whose leading ``A``
     is Cyrillic U+0410, not Latin U+0041). This is the classic homoglyph
     attack. Such tokens are **detected and reported, never silently
     rewritten**: a legitimately Cyrillic or Greek title is valid content and
     must not be corrupted.

Deliberately NOT done here: NFKC / width folding / punctuation rewriting.
This pipeline's titles are heavily Japanese and routinely contain legitimate
typography (em dash, curly apostrophe, full-width solidus U+FF0F); folding
those would corrupt visible content and break existing naming behavior.
Whether such typography should count as a "concealment" signal is a
scanner-side allowlist concern, not a filename-safety one.

Whitespace control chars (tab, newline, CR, VT, FF) are intentionally left in
place: the filename chokepoint collapses them to a single space downstream,
and the frontmatter path escapes them, so stripping them here would drop the
word boundary they represent.

Every character class below is written with ``\\uXXXX`` / ``\\xXX`` escapes
(never literal glyphs) on purpose -- a module that defends against invisible
and confusable characters must not itself embed any in its own source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Invisible / bidi / zero-width chars plus non-whitespace C0 controls and DEL.
# Excludes tab/LF/VT/FF/CR (\x09-\x0d) so the caller's whitespace handling
# still sees the word boundary. Members (written as escapes, never literal glyphs):
#   \x00-\x08, \x0e-\x1f, \x7f : C0 controls minus whitespace, plus DEL
#   \u200b-\u200f             : ZWSP, ZWNJ, ZWJ, LRM, RLM
#   \u2028, \u2029            : line / paragraph separators
#   \u202a-\u202e             : LRE, RLE, PDF, LRO, RLO (bidi embed/override)
#   \u2060-\u206f             : word joiner, invisible operators, bidi
#                                isolates (LRI/RLI/FSI/PDI), deprecated formats
#   \ufeff                    : BOM / zero-width no-break space
#   \ufff9-\ufffb            : interlinear annotation anchor/terminator/sep
_INVISIBLE_RE = re.compile(
    "[\x00-\x08\x0e-\x1f\x7f\u200b-\u200f\u2028\u2029\u202a-\u202e\u2060-\u206f\ufeff\ufff9-\ufffb]"
)

# Confusable-prone alphabets. The homoglyph signal is intra-word mixing of
# Latin with Cyrillic or Greek; CJK / Japanese are their own scripts and are
# never flagged when combined with Latin (a Latin word plus Japanese kana is
# not a signal). Ranges (written as escapes):
#   Latin:    A-Z a-z + \u00c0-\u024f (Latin-1 Supplement .. Latin Extended-B)
#   Cyrillic: \u0400-\u052f (Cyrillic + Cyrillic Supplement)
#   Greek:    \u0370-\u03ff (Greek) + \u1f00-\u1fff (Greek Extended)
_LATIN_RE = re.compile("[A-Za-z\u00c0-\u024f]")
_CYRILLIC_RE = re.compile("[\u0400-\u052f]")
_GREEK_RE = re.compile("[\u0370-\u03ff\u1f00-\u1fff]")
_TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class ConcealmentReport:
    """Outcome of analyzing one piece of filename-bound external text.

    ``text`` is the invisible-stripped result (safe to use as a filename
    component). ``mixed_script_tokens`` lists the tokens that mix Latin with
    Cyrillic/Greek -- these are reported for human/scanner review but are left
    intact inside ``text`` (no destructive transliteration).
    """

    text: str
    invisible_removed: int
    mixed_script_tokens: tuple[str, ...]

    @property
    def has_signal(self) -> bool:
        """True when anything worth alerting on was found."""
        return self.invisible_removed > 0 or bool(self.mixed_script_tokens)


def strip_invisibles(raw: str) -> tuple[str, int]:
    r"""Remove zero-width / bidi-control / non-whitespace-control chars.

    Returns ``(cleaned, removed_count)``. Idempotent: re-running on the result
    removes nothing. Whitespace controls (``\t \n \r \v \f``) are preserved so
    the caller can collapse them to a word boundary.
    """
    cleaned = _INVISIBLE_RE.sub("", raw)
    return cleaned, len(raw) - len(cleaned)


def find_mixed_script_tokens(text: str) -> tuple[str, ...]:
    """Return whitespace-delimited tokens that mix Latin with Cyrillic/Greek.

    A token is flagged only when it contains at least one Latin letter *and*
    at least one Cyrillic or Greek letter -- the shape of a homoglyph
    substitution. Pure-Latin, pure-Cyrillic, pure-Greek and Latin+CJK tokens
    are never flagged, keeping false positives near zero for this pipeline's
    Japanese/English titles.
    """
    flagged: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0)
        has_latin = bool(_LATIN_RE.search(token))
        has_confusable = bool(_CYRILLIC_RE.search(token) or _GREEK_RE.search(token))
        if has_latin and has_confusable:
            flagged.append(token)
    return tuple(flagged)


def analyze_filename_text(raw: str | None) -> ConcealmentReport:
    """Strip invisibles and detect mixed-script confusables in one pass.

    The single entry point for the fetch boundary and any caller that wants
    both the cleaned text and the concealment signal. Mixed-script detection
    runs on the *stripped* text so an invisible char wedged inside a word
    cannot split it and hide the script mix.
    """
    if not raw:
        return ConcealmentReport(text="", invisible_removed=0, mixed_script_tokens=())
    cleaned, removed = strip_invisibles(raw)
    return ConcealmentReport(
        text=cleaned,
        invisible_removed=removed,
        mixed_script_tokens=find_mixed_script_tokens(cleaned),
    )
