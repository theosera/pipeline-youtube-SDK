"""Filename concealment defense: invisible-char stripping + confusable detection.

YouTube video/playlist titles are attacker-controllable external text that
flows verbatim into on-disk note filenames, folder names, and YAML
frontmatter (see ``services/obsidian.py``). Two concealment classes matter
for a filename:

  1. **Invisible / bidi / control chars** -- zero-width joiners, the
     RIGHT-TO-LEFT OVERRIDE (U+202E), directional isolates, BOM, C0/C1
     controls, line/paragraph separators. These have no legitimate use in a
     filename and can visually reverse or hide part of a name (a U+202E before
     ``gpj.exe`` renders as ``exe.jpg``). They are **stripped unconditionally**
     via Unicode general category (Cc / Cf / Zl / Zp), so the defense covers
     every such code point -- not just a hand-picked list -- with zero false
     positives on real titles.
  2. **Mixed-script confusables** -- a single word carrying both a Latin
     letter and a Cyrillic/Greek look-alike (e.g. a word whose leading ``A``
     is Cyrillic U+0410, not Latin U+0041). This is the classic homoglyph
     attack. NFC/NFKC and invisible-char stripping do NOT touch it, so it
     needs its own handling -- and there are two entry points, by design:
       - ``find_mixed_script_tokens`` -- **detect and report only**. Used at
         the filename / fetch boundary, where a legitimately Cyrillic or Greek
         title is valid content and must not be corrupted.
       - ``fold_mixed_script_confusables`` -- **rewrite** the Cyrillic/Greek
         look-alikes to Latin. Used on the LLM-output path (note body written
         to the vault, e.g. Stage 02), where a Latin word carrying a
         Cyrillic/Greek look-alike is a model slip, not user content, so
         folding it back to the intended pure-Latin word is safe.

Deliberately NOT done here: NFKC / width folding / punctuation rewriting.
This pipeline's titles are heavily Japanese and routinely contain legitimate
typography (em dash, curly apostrophe, full-width solidus U+FF0F); folding
those would corrupt visible content and break existing naming behavior.
Whether such typography should count as a "concealment" signal is a
scanner-side allowlist concern, not a filename-safety one.

TAB / newline / CR are intentionally kept (the filename chokepoint collapses
them to a space; frontmatter escapes them). VT / FF are NOT kept -- they are
forbidden in YAML and have no legitimate title use.

The confusable-script character classes use ``\\uXXXX`` escapes (never literal
glyphs) on purpose -- a module that defends against invisible and confusable
characters must not itself embed any in its own source.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Whitespace controls the caller still needs (tab / newline / CR). VT (U+000B)
# and FF (U+000C) are deliberately NOT kept: only TAB/LF/CR are legal C0 chars
# in a YAML stream, so keeping VT/FF would let build_frontmatter emit
# unparseable frontmatter. They are category Cc, so they get stripped below.
_KEEP_WHITESPACE = frozenset("\t\n\r")
# A char is stripped if its Unicode category is Cc / Cf / Zl / Zp ...
_STRIP_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})
# ... OR it is one of these visually-blank code points that live OUTSIDE those
# categories (Mn / Lo / So) yet are classic concealment fillers a category-only
# check would miss.
_EXTRA_INVISIBLE = frozenset(
    "\u034f"  # COMBINING GRAPHEME JOINER (Mn)
    "\u115f\u1160"  # HANGUL CHOSEONG / JUNGSEONG FILLER (Lo)
    "\u17b4\u17b5"  # KHMER VOWEL INHERENT AQ / AA (Mn)
    "\u2800"  # BRAILLE PATTERN BLANK (So)
    "\u3164"  # HANGUL FILLER (Lo)
    "\uffa0"  # HALFWIDTH HANGUL FILLER (Lo)
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

# Cyrillic / Greek code points that are visual look-alikes of an ASCII Latin
# letter, mapped to that letter. This is the Unicode TR39 "confusables" subset
# restricted to ASCII-Latin targets -- the only fold safe for this pipeline:
# kana / kanji / emoji and any glyph without a plain ASCII-Latin look-alike are
# deliberately absent, so a fold can never touch legitimate Japanese or other
# non-Latin content. Keys are ``\\uXXXX`` escapes and the comments name each
# code point (never the literal glyph) so this module still embeds no
# confusable character in its own source.
_CONFUSABLE_TO_LATIN: dict[str, str] = {
    # -- Cyrillic capitals --
    "\u0410": "A",  # CYRILLIC CAPITAL LETTER A
    "\u0412": "B",  # CYRILLIC CAPITAL LETTER VE
    "\u0415": "E",  # CYRILLIC CAPITAL LETTER IE
    "\u041a": "K",  # CYRILLIC CAPITAL LETTER KA
    "\u041c": "M",  # CYRILLIC CAPITAL LETTER EM
    "\u041d": "H",  # CYRILLIC CAPITAL LETTER EN
    "\u041e": "O",  # CYRILLIC CAPITAL LETTER O
    "\u0420": "P",  # CYRILLIC CAPITAL LETTER ER
    "\u0421": "C",  # CYRILLIC CAPITAL LETTER ES
    "\u0422": "T",  # CYRILLIC CAPITAL LETTER TE
    "\u0423": "Y",  # CYRILLIC CAPITAL LETTER U
    "\u0425": "X",  # CYRILLIC CAPITAL LETTER HA
    "\u0405": "S",  # CYRILLIC CAPITAL LETTER DZE
    "\u0406": "I",  # CYRILLIC CAPITAL LETTER BYELORUSSIAN-UKRAINIAN I
    "\u0408": "J",  # CYRILLIC CAPITAL LETTER JE
    "\u051a": "Q",  # CYRILLIC CAPITAL LETTER QA
    "\u051c": "W",  # CYRILLIC CAPITAL LETTER WE
    # -- Cyrillic small --
    "\u0430": "a",  # CYRILLIC SMALL LETTER A
    "\u0435": "e",  # CYRILLIC SMALL LETTER IE
    "\u043a": "k",  # CYRILLIC SMALL LETTER KA
    "\u043e": "o",  # CYRILLIC SMALL LETTER O
    "\u0440": "p",  # CYRILLIC SMALL LETTER ER
    "\u0441": "c",  # CYRILLIC SMALL LETTER ES
    "\u0443": "y",  # CYRILLIC SMALL LETTER U
    "\u0445": "x",  # CYRILLIC SMALL LETTER HA
    "\u0455": "s",  # CYRILLIC SMALL LETTER DZE
    "\u0456": "i",  # CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I
    "\u0458": "j",  # CYRILLIC SMALL LETTER JE
    "\u0501": "d",  # CYRILLIC SMALL LETTER KOMI DE
    "\u051b": "q",  # CYRILLIC SMALL LETTER QA
    "\u051d": "w",  # CYRILLIC SMALL LETTER WE
    "\u04bb": "h",  # CYRILLIC SMALL LETTER SHHA
    "\u04cf": "l",  # CYRILLIC SMALL LETTER PALOCHKA
    # -- Greek capitals --
    "\u0391": "A",  # GREEK CAPITAL LETTER ALPHA
    "\u0392": "B",  # GREEK CAPITAL LETTER BETA
    "\u0395": "E",  # GREEK CAPITAL LETTER EPSILON
    "\u0396": "Z",  # GREEK CAPITAL LETTER ZETA
    "\u0397": "H",  # GREEK CAPITAL LETTER ETA
    "\u0399": "I",  # GREEK CAPITAL LETTER IOTA
    "\u039a": "K",  # GREEK CAPITAL LETTER KAPPA
    "\u039c": "M",  # GREEK CAPITAL LETTER MU
    "\u039d": "N",  # GREEK CAPITAL LETTER NU
    "\u039f": "O",  # GREEK CAPITAL LETTER OMICRON
    "\u03a1": "P",  # GREEK CAPITAL LETTER RHO
    "\u03a4": "T",  # GREEK CAPITAL LETTER TAU
    "\u03a5": "Y",  # GREEK CAPITAL LETTER UPSILON
    "\u03a7": "X",  # GREEK CAPITAL LETTER CHI
    # -- Greek small --
    "\u03bf": "o",  # GREEK SMALL LETTER OMICRON
    "\u03c1": "p",  # GREEK SMALL LETTER RHO
    "\u03bd": "v",  # GREEK SMALL LETTER NU
    "\u03f2": "c",  # GREEK LUNATE SIGMA SYMBOL
    "\u03b1": "a",  # GREEK SMALL LETTER ALPHA
}


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


def _is_concealment_char(ch: str) -> bool:
    """True if ``ch`` is an invisible / control / separator char to strip.

    Category-driven (Cc/Cf/Zl/Zp) plus an explicit set of visually-blank code
    points outside those categories (``_EXTRA_INVISIBLE``). TAB/LF/CR are kept.
    ZWJ/ZWNJ (U+200D/U+200C) ARE stripped: the observed vault attack uses a
    zero-width joiner between words, so preserving them for the rare emoji-ZWJ /
    Persian case would reopen the exact hole (a broken emoji in a filename is
    cosmetic; a concealed filename is not).
    """
    if ch in _KEEP_WHITESPACE:
        return False
    if ch in _EXTRA_INVISIBLE:
        return True
    return unicodedata.category(ch) in _STRIP_CATEGORIES


def strip_invisibles(raw: str) -> tuple[str, int]:
    r"""Remove zero-width / bidi / control / separator chars.

    Returns ``(cleaned, removed_count)``. Idempotent: re-running on the result
    removes nothing. Whitespace controls (``\t \n \r``) are preserved so the
    caller can collapse them to a word boundary.
    """
    cleaned = "".join(ch for ch in raw if not _is_concealment_char(ch))
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


def fold_mixed_script_confusables(text: str) -> str:
    """Fold Cyrillic/Greek homoglyphs to Latin inside mixed-script tokens.

    Complements ``find_mixed_script_tokens``: that detector *reports* the
    homoglyph shape (a token mixing Latin with Cyrillic/Greek) and leaves the
    text intact -- correct at the filename / fetch boundary, where a genuinely
    Cyrillic or Greek title must survive. This function *rewrites* the same
    shape and is meant for the LLM-output path (note body written to the vault,
    e.g. Stage 02), where a Latin word carrying a Cyrillic/Greek look-alike is a
    model slip, not user content, so restoring the intended pure-Latin word is
    safe. NFC/NFKC and invisible-char stripping never remove this class, so the
    fold is a distinct, additional layer -- it does not replace the existing
    invisible-char defense.

    Per whitespace-delimited token: only when the token mixes at least one
    Latin letter with at least one Cyrillic/Greek letter, each character that
    has an ASCII-Latin look-alike in ``_CONFUSABLE_TO_LATIN`` is swapped for it.
    Every other token -- pure-Latin, pure-Cyrillic (legitimate Russian),
    pure-Greek, Latin+CJK/kana (Japanese) -- is returned unchanged, so
    legitimate non-Latin content is never corrupted.

    Deterministic (fixed table, no model) and idempotent: a folded token
    becomes pure-Latin and no longer qualifies as mixed-script, so a second
    pass is a no-op.
    """
    if not text:
        return text

    def _fold_token(match: re.Match[str]) -> str:
        token = match.group(0)
        has_latin = bool(_LATIN_RE.search(token))
        has_confusable = bool(_CYRILLIC_RE.search(token) or _GREEK_RE.search(token))
        if not (has_latin and has_confusable):
            return token
        return "".join(_CONFUSABLE_TO_LATIN.get(ch, ch) for ch in token)

    return _TOKEN_RE.sub(_fold_token, text)


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
