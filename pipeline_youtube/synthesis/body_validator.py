"""Post-validation for Leader-generated body markdown.

Leader outputs `body_markdown` that is written verbatim to the Vault.
Without a validation layer, prompt-injected inputs could cause the
model to emit:

- `![[...]]` embeds pointing at files that don't exist (or outside the
  allowed asset set) — pollutes the vault graph.
- Active HTML (`<script>`, `<iframe>`, `<object>`, `<embed>`, `<style>`)
  that Obsidian may render in preview mode.
- Templater `<% ... %>` tokens that Obsidian evaluates actively.

This module provides the last-line defense before disk write.
"""

from __future__ import annotations

import re

_EMBED_RE = re.compile(r"!\[\[([^\]]+?)\]\]")
_HTML_TAG_RE = re.compile(r"<(script|iframe|object|embed|style)[^>]*>", re.IGNORECASE)
_TEMPLATER_RE = re.compile(r"<%[^%]*%>")


def extract_allowed_embeds(md_bodies: list[str]) -> frozenset[str]:
    """Collect every `![[filename]]` embed target present in source md bodies.

    These are the only filenames that should survive validation in
    downstream synthesis output.
    """
    allowed: set[str] = set()
    for body in md_bodies:
        for match in _EMBED_RE.finditer(body):
            allowed.add(match.group(1).strip())
    return frozenset(allowed)


def validate_chapter_body(body: str, allowed_assets: frozenset[str] | set[str]) -> str:
    """Strip disallowed embeds, HTML tags, and Templater tokens.

    - `![[name]]` not in `allowed_assets` is replaced with a dropped-embed comment.
    - Active HTML opening tags are stripped.
    - Templater `<% ... %>` tokens are stripped.

    Preserves plain markdown (headings, lists, tables, Q:/A: flashcards,
    `#flashcards` tag, wiki links `[[...]]`, etc.).
    """
    allowed = frozenset(allowed_assets)

    def _filter_embed(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        if target in allowed:
            return match.group(0)
        return f"<!-- dropped embed: {target!r} -->"

    body = _EMBED_RE.sub(_filter_embed, body)
    body = _HTML_TAG_RE.sub("", body)
    body = _TEMPLATER_RE.sub("", body)
    return body
