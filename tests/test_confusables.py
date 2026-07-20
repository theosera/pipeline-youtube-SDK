"""Tests for services.confusables: filename concealment defense.

Invisible / confusable characters are built with ``chr()`` rather than pasted
as literals, so the test source itself stays free of the very characters it
exercises (mirrors the module's own no-literal-glyphs rule).
"""

from __future__ import annotations

from pipeline_youtube.services.confusables import (
    ConcealmentReport,
    analyze_filename_text,
    find_mixed_script_tokens,
    strip_invisibles,
)

# --- concealment code points, by name ---
RLO = chr(0x202E)  # RIGHT-TO-LEFT OVERRIDE
LRO = chr(0x202D)  # LEFT-TO-RIGHT OVERRIDE
ZWSP = chr(0x200B)  # ZERO WIDTH SPACE
ZWNJ = chr(0x200C)  # ZERO WIDTH NON-JOINER
ZWJ = chr(0x200D)  # ZERO WIDTH JOINER
WJ = chr(0x2060)  # WORD JOINER
BOM = chr(0xFEFF)  # ZERO WIDTH NO-BREAK SPACE / BOM
RLI = chr(0x2067)  # RIGHT-TO-LEFT ISOLATE
LS = chr(0x2028)  # LINE SEPARATOR

# --- confusable letters ---
CYR_A = chr(0x430)  # CYRILLIC SMALL LETTER A (looks like Latin 'a')
CYR_CAP_A = chr(0x410)  # CYRILLIC CAPITAL LETTER A
GRK_O = chr(0x3BF)  # GREEK SMALL LETTER OMICRON (looks like Latin 'o')


class TestStripInvisibles:
    def test_rlo_override_removed(self):
        assert strip_invisibles(f"report{RLO}gpj.exe") == ("reportgpj.exe", 1)

    def test_zero_width_family_removed(self):
        assert strip_invisibles(f"a{ZWSP}b{ZWNJ}c{ZWJ}d{WJ}e") == ("abcde", 4)

    def test_bom_and_isolate_and_line_sep_removed(self):
        assert strip_invisibles(f"{BOM}x{RLI}y{LS}z") == ("xyz", 3)

    def test_control_and_del_removed(self):
        assert strip_invisibles("a\x00b\x01c\x7fd") == ("abcd", 3)

    def test_whitespace_controls_preserved(self):
        # tab / newline / CR / VT / FF stay — the caller collapses them to a
        # single space, so removing them here would drop the word boundary.
        raw = "a\tb\nc\rd\x0be\x0cf"
        assert strip_invisibles(raw) == (raw, 0)

    def test_plain_text_untouched(self):
        assert strip_invisibles("The AI System") == ("The AI System", 0)

    def test_japanese_and_typography_untouched(self):
        # em dash, curly apostrophe, full-width solidus are legitimate content.
        raw = "—wasn’t／ハーネス設計"
        assert strip_invisibles(raw) == (raw, 0)

    def test_idempotent(self):
        once, _ = strip_invisibles(f"a{ZWSP}{RLO}b")
        twice, removed = strip_invisibles(once)
        assert twice == once and removed == 0


class TestFindMixedScriptTokens:
    def test_cyrillic_in_latin_word(self):
        assert find_mixed_script_tokens(f"buy {CYR_A}pple") == (f"{CYR_A}pple",)

    def test_cyrillic_capital_in_latin_word(self):
        assert find_mixed_script_tokens(f"{CYR_CAP_A}pple") == (f"{CYR_CAP_A}pple",)

    def test_greek_in_latin_word(self):
        assert find_mixed_script_tokens(f"c{GRK_O}de") == (f"c{GRK_O}de",)

    def test_pure_latin_not_flagged(self):
        assert find_mixed_script_tokens("Anthropic public harness API") == ()

    def test_pure_cyrillic_not_flagged(self):
        # a legitimately Russian token is valid content, not a homoglyph.
        assert find_mixed_script_tokens("привет") == ()

    def test_latin_plus_japanese_not_flagged(self):
        assert find_mixed_script_tokens("Anthropicが公開したハーネス設計") == ()

    def test_multiple_tokens_flagged(self):
        got = find_mixed_script_tokens(f"{CYR_A}pple and c{GRK_O}de")
        assert got == (f"{CYR_A}pple", f"c{GRK_O}de")


class TestAnalyzeFilenameText:
    def test_none_and_empty(self):
        assert analyze_filename_text(None) == ConcealmentReport("", 0, ())
        assert analyze_filename_text("").has_signal is False

    def test_clean_title_no_signal(self):
        r = analyze_filename_text("The AI System Most People Aren't Building")
        assert r.text == "The AI System Most People Aren't Building"
        assert r.has_signal is False

    def test_invisible_only(self):
        r = analyze_filename_text(f"clean{ZWSP}{ZWSP}name")
        assert r.text == "cleanname"
        assert r.invisible_removed == 2
        assert r.mixed_script_tokens == ()
        assert r.has_signal is True

    def test_invisible_wedge_does_not_hide_script_mix(self):
        # a zero-width char inside a word must not split it and hide the mix.
        r = analyze_filename_text(f"x{ZWNJ}{CYR_A}pple")
        assert r.text == f"x{CYR_A}pple"
        assert r.mixed_script_tokens == (f"x{CYR_A}pple",)
        assert r.has_signal is True

    def test_mixed_script_only(self):
        r = analyze_filename_text(f"{CYR_A}pple")
        assert r.text == f"{CYR_A}pple"  # not rewritten
        assert r.invisible_removed == 0
        assert r.mixed_script_tokens == (f"{CYR_A}pple",)
        assert r.has_signal is True
