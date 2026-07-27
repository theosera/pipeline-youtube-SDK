"""Tests for H3: Leader body_markdown post-validation."""

from __future__ import annotations

from pipeline_youtube.synthesis.body_validator import (
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
