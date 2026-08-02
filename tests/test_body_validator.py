"""Tests for H3: Leader body_markdown post-validation."""

from __future__ import annotations

import time

import pytest

from pipeline_youtube.synthesis.body_validator import (
    BodyValidationError,
    extract_allowed_embeds,
    validate_chapter_body,
)


class TestExtractAllowedEmbeds:
    def test_single_body(self):
        body = "intro\n![[foo.webp]]\noutro"
        assert extract_allowed_embeds([body]) == frozenset({"foo.webp"})

    def test_multiple_bodies(self):
        bodies = ["![[a.webp]]", "![[b.gif]] and ![[c.webp]]"]
        assert extract_allowed_embeds(bodies) == frozenset({"a.webp", "b.gif", "c.webp"})

    def test_no_embeds(self):
        assert extract_allowed_embeds(["plain text", "no embeds"]) == frozenset()

    def test_strips_whitespace(self):
        assert extract_allowed_embeds(["![[  spaced.webp  ]]"]) == frozenset({"spaced.webp"})

    def test_path_qualified_embed_under_bracketed_playlist_folder(self):
        # Stage 03 writes `![[{playlist_folder}/name.webp]]` and the folder keeps
        # brackets, so the target legitimately contains a lone `]`.
        target = "2026-06-14-1100 [LLM] Agent Teams/pyt_x_00.webp"
        assert extract_allowed_embeds([f"intro\n![[{target}]]\noutro"]) == frozenset({target})


class TestValidateChapterBody:
    def test_allowed_embed_preserved(self):
        out = validate_chapter_body("![[ok.webp]]", {"ok.webp"})
        assert "![[ok.webp]]" in out

    def test_disallowed_embed_dropped(self):
        out = validate_chapter_body("![[evil.webp]]", {"ok.webp"})
        assert "![[evil.webp]]" not in out
        assert "dropped embed" in out
        assert "evil.webp" in out

    def test_script_tag_stripped(self):
        out = validate_chapter_body("<script>alert(1)</script>hello", frozenset())
        assert "<script>" not in out
        assert "hello" in out

    def test_iframe_stripped(self):
        out = validate_chapter_body("<iframe src='x'></iframe>", frozenset())
        assert "<iframe" not in out

    def test_templater_stripped(self):
        out = validate_chapter_body("before <% tp.date.now() %> after", frozenset())
        assert "<%" not in out
        assert "%>" not in out
        assert "before" in out
        assert "after" in out

    def test_flashcards_preserved(self):
        body = "## フラッシュカード\n#flashcards\nQ: what?\nA: this.\n"
        out = validate_chapter_body(body, frozenset())
        assert "#flashcards" in out
        assert "Q: what?" in out
        assert "A: this." in out

    def test_wiki_link_preserved(self):
        out = validate_chapter_body("see [[other_note]]", frozenset())
        assert "[[other_note]]" in out

    def test_mixed_threats(self):
        body = "intro\n![[good.webp]]\n![[evil.webp]]\n<script>x</script>\n<% bad %>\nplain text\n"
        out = validate_chapter_body(body, {"good.webp"})
        assert "![[good.webp]]" in out
        assert "![[evil.webp]]" not in out
        assert "<script>" not in out
        assert "<%" not in out
        assert "plain text" in out

    def test_bracketed_playlist_embed_preserved_when_allowed(self):
        target = "2026-06-14-1100 [LLM] Agent Teams/pyt_x_00.webp"
        assert f"![[{target}]]" in validate_chapter_body(f"![[{target}]]", {target})

    def test_bracketed_playlist_embed_dropped_when_not_allowed(self):
        # Security regression: stopping the target at the first `]` left this
        # embed unmatched, so the filter failed open and wrote it verbatim.
        target = "2026-06-14-1100 [LLM] Agent Teams/evil.webp"
        out = validate_chapter_body(f"![[{target}]]", frozenset())
        assert f"![[{target}]]" not in out
        assert "dropped embed" in out


class TestNestedReconstruction:
    """A single strip pass let a removed tag splice its neighbours into a new one."""

    def test_nested_templater_token_is_not_reconstructed(self):
        # One pass turned this into the live token
        # `<%* await require('child_process').exec('...') %>`.
        body = "<%<% %>* await require('child_process').exec('...') %>"
        out = validate_chapter_body(body, frozenset())
        assert "<%" not in out  # no opener left, so Templater cannot evaluate it

    def test_nested_iframe_is_not_reconstructed(self):
        # One pass turned this into `<iframe src="https://attacker.example/" ...>`.
        body = '<ifr<iframe>ame src="https://attacker.example/" width=800>'
        assert validate_chapter_body(body, frozenset()) == ""

    def test_templater_token_containing_percent_is_stripped(self):
        # `<%[^%]*%>` could not cross the `%` in `100 % 7`, so this matched
        # nothing and survived however many passes the loop ran.
        body = "<%* const p = 100 % 7; app.vault.adapter.write(...) %>"
        assert validate_chapter_body(body, frozenset()) == ""

    @pytest.mark.parametrize(
        "token",
        [
            "<% tp.date.now() %>",
            "<%* const p = 100 % 7; %>",
            "<%_ trimmed _%>",
            "<%+ 'x' %>",
            "<%~ y %>",
        ],
    )
    def test_templater_variants_are_stripped(self, token: str):
        assert validate_chapter_body(f"before {token} after", frozenset()) == "before  after"

    def test_two_tokens_are_stripped_independently(self):
        # A greedy match would swallow ` mid ` between them.
        assert validate_chapter_body("<% a %> mid <% b %>", frozenset()) == " mid "

    def test_unterminated_opener_is_left_alone(self):
        body = "<%% not a token"
        assert validate_chapter_body(body, frozenset()) == body


class TestSanitizePassCap:
    """The fixed-point loop is capped, and the cap fails closed."""

    def test_three_level_nesting_still_settles(self):
        # 5 passes — the documented ceiling for legitimate settling.
        assert validate_chapter_body("<scr<scr<scr<script>ipt>ipt>ipt>", frozenset()) == ""

    def test_four_level_nesting_is_rejected(self):
        # Needs 6 passes, so the cap refuses it.
        with pytest.raises(BodyValidationError):
            validate_chapter_body("<scr<scr<scr<scr<script>ipt>ipt>ipt>ipt>", frozenset())

    def test_rejection_does_not_return_a_partially_stripped_body(self):
        # Truncating instead of raising would hand callers text that looks
        # sanitized while `<script>` is still reconstructible — worse than the
        # bug the cap exists to catch.
        body = "<scr<scr<scr<scr<script>ipt>ipt>ipt>ipt>"
        with pytest.raises(BodyValidationError) as excinfo:
            validate_chapter_body(body, frozenset())
        assert "<script" not in str(excinfo.value)

    def test_rejection_message_carries_no_body_text(self):
        # The body is attacker-influenced and this message reaches logs.
        marker = "MARKER-MUST-NOT-BE-LOGGED"
        body = f"<scr<scr<scr<scr<script>ipt>ipt>ipt>ipt>{marker}"
        with pytest.raises(BodyValidationError) as excinfo:
            validate_chapter_body(body, frozenset())
        assert marker not in str(excinfo.value)

    def test_deep_nesting_at_100kb_is_refused_quickly(self):
        # Uncapped, this input needs 12,802 passes / ~6.5 s. Drop
        # _MAX_SANITIZE_PASSES and this test stops passing.
        body = "<scr" * 12_800 + "<script>" + "ipt>" * 12_800
        started = time.perf_counter()
        with pytest.raises(BodyValidationError):
            validate_chapter_body(body, frozenset())
        assert time.perf_counter() - started < 1.0

    def test_unclosed_templater_run_at_400kb_stays_linear(self):
        # Guards the scanner against a regression to a regex. Every regex that
        # crosses `%` to find `%>` re-scans the tail from each `<%`:
        # `<%[\s\S]*?%>` measured ~17.9 s on 96 KB. Stage 01 will hand this
        # validator transcript-sized bodies, so the shape has to stay linear.
        body = "<% " * 133_000
        started = time.perf_counter()
        assert validate_chapter_body(body, frozenset()) == body
        assert time.perf_counter() - started < 1.0


class TestCurrentCodeFenceBehaviour:
    """Pins today's behaviour rather than asserting it is desirable.

    The strip is fence-blind, so a legitimate ```html block loses its tags.
    Stage 01 writes real source into fences (`01_Scripts.md`), so the next
    session needs this visible. Fence handling is deliberately not built here.
    """

    def test_html_inside_a_code_fence_is_stripped(self):
        body = "```html\n<script>alert(1)</script>\n<iframe src='x'></iframe>\n```\n"
        out = validate_chapter_body(body, frozenset())
        assert "<script>" not in out
        assert "<iframe" not in out
        assert "```html" in out

    def test_templater_token_inside_a_code_fence_is_stripped(self):
        out = validate_chapter_body("```js\n<% tp.date.now() %>\n```", frozenset())
        assert "<%" not in out
        assert "```js" in out


class TestNormalBodyUnchanged:
    def test_rich_markdown_passes_through_byte_identical(self):
        body = (
            "## 見出し\n\n"
            "- 箇条書き 1\n"
            "- 箇条書き 2\n\n"
            "| 列 A | 列 B |\n|---|---|\n| 1 | 2 |\n\n"
            "本文中の [[wikilink]] と ![[ok.webp]]\n\n"
            "#flashcards\nQ: 問\nA: 答\n"
        )
        assert validate_chapter_body(body, {"ok.webp"}) == body

    def test_event_handler_attribute_survives(self):
        # Not a regression — `on*=` is outside the five-tag pattern and this
        # PR does not address it. Pinned so nobody reads the loop as XSS cover.
        body = '<img src=x onerror="alert(1)">'
        assert validate_chapter_body(body, frozenset()) == body
