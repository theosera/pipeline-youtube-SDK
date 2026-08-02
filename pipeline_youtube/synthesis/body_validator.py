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

Scope, so callers do not over-trust it: only the five tag names above are
removed. Event-handler attributes (`onerror=`, `onload=`, …) and
`javascript:` / `data:` URLs are not in the pattern and survive validation,
on permitted tags as much as anywhere else — `<img src=x onerror=…>` passes
through untouched. This is a targeted strip, not HTML sanitisation.
"""

from __future__ import annotations

import re
import sys

# The target must admit a literal ``]`` that is not the ``]]`` terminator.
# Stage 03 writes path-qualified embeds (``![[{playlist_folder}/name.webp]]``)
# and ``sanitize_title_for_filename`` keeps ``[`` / ``]``, so a playlist folder
# like ``2026-06-14-1100 [LLM] Agent Teams`` is a real target. Stopping at the
# first ``]`` broke this filter in both directions: legitimate embeds never
# reached the allow-list, and — worse — a *disallowed* embed under a bracketed
# folder was not matched at all and passed through unfiltered (fail-open).
# Same rule as ``services.confusables._WIKILINK_RE``.
_EMBED_RE = re.compile(r"!\[\[((?:[^\]\n]|\](?!\]))+)\]\]")
_HTML_TAG_RE = re.compile(r"<(script|iframe|object|embed|style)[^>]*>", re.IGNORECASE)

_TEMPLATER_OPEN = "<%"
_TEMPLATER_CLOSE = "%>"

# Removing one tag can splice its neighbours into a fresh one
# (``<ifr<iframe>ame …>`` -> ``<iframe …>``), so stripping has to run to a fixed
# point. Each nesting level costs one extra pass and a pass scans the whole
# body, so an *unbounded* loop is quadratic: 102 KB of nested ``<scr…ipt>``
# measured 12,802 passes / 6.5 s. The input is LLM output steered by
# attacker-controlled captions, so the loop is capped.
#
# Measured passes to settle: plain body 1, single-layer attack 2, and 3/4/5 for
# 1/2/3 levels of nesting. Four levels needs 6 and is refused.
_MAX_SANITIZE_PASSES = 5


class BodyValidationError(ValueError):
    """Raised when active markup outlives ``_MAX_SANITIZE_PASSES`` strip passes.

    Refusing beats returning the partially stripped body: a truncated run
    yields text that *looks* sanitized to every caller downstream, which is a
    worse failure than the reconstruction bug the cap exists to catch.
    """


def _strip_templater_tokens(text: str) -> str:
    """Remove ``<% … %>`` tokens, including bodies that contain a bare ``%``.

    Scanned rather than matched by a regex, deliberately. The previous
    ``<%[^%]*%>`` could not cross a ``%``, so ``<%* const p = 100 % 7; … %>``
    never matched and survived every pass. Every regex that *can* cross it
    re-scans the tail from each ``<%``, which is quadratic on a run of unclosed
    ``<%``: at 96 KB, ``<%[\\s\\S]*?%>`` measured 17.9 s and the alternation
    form ``<%(?:[^%]|%(?!>))*%>`` 42.1 s. Capping the pass count does not help,
    because that cost is inside a single pass.

    ``str.find`` is linear here: a missing ``%>`` ends the search outright,
    since no later ``<%`` can be closed either. Behaviour otherwise matches the
    lazy regex — the first ``%>`` closes the token.
    """
    out: list[str] = []
    pos = 0
    while True:
        start = text.find(_TEMPLATER_OPEN, pos)
        if start < 0:
            break
        end = text.find(_TEMPLATER_CLOSE, start + len(_TEMPLATER_OPEN))
        if end < 0:
            break
        out.append(text[pos:start])
        pos = end + len(_TEMPLATER_CLOSE)
    out.append(text[pos:])
    return "".join(out)


def _strip_active_markup(body: str) -> str:
    """One removal pass: active HTML opening tags, then Templater tokens."""
    return _strip_templater_tokens(_HTML_TAG_RE.sub("", body))


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

    The two strip steps repeat until the body stops changing, because removing
    a tag can splice the surrounding text into a new one. The embed filter
    stays outside that loop: it substitutes rather than deletes, so it cannot
    feed the reconstruction it would otherwise be re-run against.

    Preserves plain markdown (headings, lists, tables, Q:/A: flashcards,
    `#flashcards` tag, wiki links `[[...]]`, etc.).

    Raises
    ------
    BodyValidationError
        If the body still changes after ``_MAX_SANITIZE_PASSES`` passes.
    """
    allowed = frozenset(allowed_assets)

    def _filter_embed(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        if target in allowed:
            return match.group(0)
        return f"<!-- dropped embed: {target!r} -->"

    body = _EMBED_RE.sub(_filter_embed, body)

    for _ in range(_MAX_SANITIZE_PASSES):
        stripped = _strip_active_markup(body)
        if stripped == body:
            return body
        body = stripped

    # Carries no body text on purpose: the body is attacker-influenced and this
    # string reaches logs.
    sys.stderr.write(
        f"[body_validator] active markup still present after {_MAX_SANITIZE_PASSES} "
        f"strip passes ({len(body)} chars); rejecting body\n"
    )
    raise BodyValidationError(
        f"active markup survived {_MAX_SANITIZE_PASSES} strip passes "
        f"({len(body)} chars); refusing to emit a partially sanitized body"
    )
