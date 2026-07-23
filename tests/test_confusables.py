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
    fold_mixed_script_confusables,
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
CYR_E = chr(0x435)  # CYRILLIC SMALL LETTER IE (looks like Latin 'e')
GRK_O = chr(0x3BF)  # GREEK SMALL LETTER OMICRON (looks like Latin 'o')
GRK_CHI = chr(0x3C7)  # GREEK SMALL LETTER CHI (looks like Latin 'x')
GRK_IOTA = chr(0x3B9)  # GREEK SMALL LETTER IOTA (looks like Latin 'i')
# Legitimately Russian words, built via chr() so this test source embeds no
# literal confusable glyph (mirrors the module's own rule).
CYR_PRIVET = "".join(chr(c) for c in (0x43F, 0x440, 0x438, 0x432, 0x435, 0x442))  # привет
CYR_RUSSKIY = "".join(chr(c) for c in (0x420, 0x443, 0x441, 0x441, 0x43A, 0x438, 0x439))  # Русский
CYR_ROSSIYA = "".join(chr(c) for c in (0x420, 0x43E, 0x441, 0x441, 0x438, 0x44F))  # Россия


class TestStripInvisibles:
    def test_rlo_override_removed(self):
        assert strip_invisibles(f"report{RLO}gpj.exe") == ("reportgpj.exe", 1)

    def test_zero_width_family_removed(self):
        assert strip_invisibles(f"a{ZWSP}b{ZWNJ}c{ZWJ}d{WJ}e") == ("abcde", 4)

    def test_bom_and_isolate_and_line_sep_removed(self):
        assert strip_invisibles(f"{BOM}x{RLI}y{LS}z") == ("xyz", 3)

    def test_control_and_del_removed(self):
        assert strip_invisibles("a\x00b\x01c\x7fd") == ("abcd", 3)

    def test_c1_control_and_soft_hyphen_removed(self):
        # Category-based stripping (Cc/Cf) also covers C1 controls and the soft
        # hyphen that a hand-enumerated BMP range could miss.
        nel, shy = chr(0x85), chr(0x00AD)
        assert strip_invisibles(f"a{nel}b{shy}c") == ("abc", 2)

    def test_tab_lf_cr_kept_vt_ff_stripped(self):
        # tab / newline / CR stay (word boundary the caller collapses); VT / FF
        # are stripped — YAML-invalid and no legitimate title use.
        assert strip_invisibles("a\tb\nc\rd") == ("a\tb\nc\rd", 0)
        assert strip_invisibles("a\x0bb\x0cc") == ("abc", 2)

    def test_supplemental_blank_code_points_removed(self):
        # visually-blank chars outside Cc/Cf/Zl/Zp (Mn/Lo/So) are still stripped.
        cgj, hangul_filler, braille_blank = chr(0x034F), chr(0x3164), chr(0x2800)
        assert strip_invisibles(f"a{cgj}b{hangul_filler}c{braille_blank}d") == ("abcd", 3)

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


class TestFoldMixedScriptConfusables:
    def test_cyrillic_in_latin_word_becomes_pure_latin(self):
        # "Vibе" with a Cyrillic IE hidden in a Latin word -> pure ASCII "Vibe".
        folded = fold_mixed_script_confusables(f"Vib{CYR_E}")
        assert folded == "Vibe"
        assert folded.isascii()

    def test_cyrillic_a_folded(self):
        assert fold_mixed_script_confusables(f"buy {CYR_A}pple") == "buy apple"

    def test_greek_omicron_folded(self):
        assert fold_mixed_script_confusables(f"c{GRK_O}de") == "code"

    def test_multiple_mixed_tokens_folded(self):
        got = fold_mixed_script_confusables(f"{CYR_A}pple and c{GRK_O}de")
        assert got == "apple and code"

    def test_pure_cyrillic_untouched(self):
        # THE most important regression: a legitimately Russian word has no
        # Latin letter, so it is not a homoglyph and must never be folded.
        assert fold_mixed_script_confusables(CYR_PRIVET) == CYR_PRIVET

    def test_japanese_and_latin_untouched(self):
        # Latin + kana/kanji is not a homoglyph signal; content stays intact.
        raw = "Anthropicが公開したハーネス設計"
        assert fold_mixed_script_confusables(raw) == raw

    def test_pure_latin_untouched(self):
        assert fold_mixed_script_confusables("plain ascii summary body") == (
            "plain ascii summary body"
        )

    def test_greek_chi_folded(self):
        # eχit with Greek small chi -> exit (chi is an ASCII 'x' look-alike).
        assert fold_mixed_script_confusables(f"e{GRK_CHI}it") == "exit"

    def test_greek_iota_folded(self):
        assert fold_mixed_script_confusables(f"f{GRK_IOTA}le") == "file"

    def test_markdown_link_cyrillic_label_untouched(self):
        # Regression: whitespace tokenization used to glue the Latin URL to the
        # Cyrillic label into one token and corrupt the label. Letter-run
        # folding keeps the pure-Cyrillic label intact.
        raw = f"[{CYR_PRIVET}](https://example.com)"
        assert fold_mixed_script_confusables(raw) == raw

    def test_table_row_pure_scripts_untouched(self):
        # `|English|Русский|` has no whitespace between the Latin and Cyrillic
        # cells; each is its own pure-script word and must survive unchanged.
        raw = f"|English|{CYR_RUSSKIY}|"
        assert fold_mixed_script_confusables(raw) == raw

    def test_unicode_url_path_untouched(self):
        raw = f"https://ru.wikipedia.org/wiki/{CYR_ROSSIYA}"
        assert fold_mixed_script_confusables(raw) == raw

    def test_idempotent(self):
        once = fold_mixed_script_confusables(f"Vib{CYR_E} c{GRK_O}de e{GRK_CHI}it")
        assert fold_mixed_script_confusables(once) == once

    def test_empty(self):
        assert fold_mixed_script_confusables("") == ""
