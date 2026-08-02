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

import logging
import re

_LOG = logging.getLogger(__name__)

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

# Removing one construct can splice its neighbours into a fresh one
# (``<ifr<iframe>ame …>`` -> ``<iframe …>``), so sanitising has to run to a
# fixed point. Each nesting level costs one extra pass and a pass scans the
# whole body, so an *unbounded* loop is quadratic: 102 KB of nested
# ``<scr…ipt>`` measured 12,802 passes / 6.5 s. The input is LLM output steered
# by attacker-controlled captions, so the loop is capped.
#
# Measured passes to settle: plain body 1, one disallowed embed 2, single-layer
# markup attack 2, and 3/4/5 for 1/2/3 levels of nesting. The two embed
# reconstructions covered in the tests settle at 3 and 4. Four levels of
# nesting needs 6 and is refused.
_MAX_SANITIZE_PASSES = 5


class BodyValidationError(ValueError):
    """Raised when markup outlives ``_MAX_SANITIZE_PASSES`` sanitize passes.

    Refusing beats returning the partially sanitized body: a truncated run
    yields text that *looks* clean to every caller downstream, which is a worse
    failure than the reconstruction bug the cap exists to catch.

    Propagating is deliberate. Callers (``synthesis.chapter`` /
    ``synthesis.moc`` / ``handson.writer`` / ``stages.summary``) do **not**
    catch it, so a body this validator cannot settle aborts the write rather
    than reaching the vault. Per-caller recovery is out of scope for this
    layer — the point of a last-line defense is that there is no path around it.
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


def _filter_embed(match: re.Match[str], allowed: frozenset[str]) -> str:
    """Keep an allow-listed embed, else replace it with a dropped-embed note."""
    target = match.group(1).strip()
    if target in allowed:
        return match.group(0)
    return f"<!-- dropped embed: {target!r} -->"


def _sanitize_once(body: str, allowed: frozenset[str]) -> str:
    """One pass: disallowed embeds, then active HTML, then Templater tokens.

    The embed filter belongs *inside* the fixed-point loop rather than ahead of
    it. Stripping markup can assemble an embed that did not exist when the
    filter last ran, and that embed would then be written without ever being
    checked against ``allowed``:

    - ``![[evil<scr<script\\n>ipt>.webp]]`` — the newline keeps ``_EMBED_RE``
      from matching, then the tag strip yields a live ``![[evil.webp]]``.
    - ``![[../../secret]<scr<script>ipt>]`` — the tag straddles the ``]]``
      terminator, and removing it yields ``![[../../secret]]``.

    Re-running the filter every pass closes that. It still converges: an
    allowed embed returns itself, and a disallowed one becomes a comment
    holding no ``![[``, so neither feeds a further substitution.
    """
    body = _EMBED_RE.sub(lambda match: _filter_embed(match, allowed), body)
    body = _HTML_TAG_RE.sub("", body)
    return _strip_templater_tokens(body)


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

    All three repeat until the body stops changing, because removing one
    construct can splice the surrounding text into another one — including an
    embed, which is why the allow-list check runs on every pass instead of once
    up front (see ``_sanitize_once``).

    Preserves plain markdown (headings, lists, tables, Q:/A: flashcards,
    `#flashcards` tag, wiki links `[[...]]`, etc.).

    Raises
    ------
    BodyValidationError
        If the body still changes after ``_MAX_SANITIZE_PASSES`` passes.
    """
    allowed = frozenset(allowed_assets)

    for _ in range(_MAX_SANITIZE_PASSES):
        cleaned = _sanitize_once(body, allowed)
        if cleaned == body:
            return body
        body = cleaned

    # Carries no body text on purpose: the body is attacker-influenced and this
    # record reaches logs.
    _LOG.warning(
        "active markup still present after %d sanitize passes (%d chars); rejecting body",
        _MAX_SANITIZE_PASSES,
        len(body),
    )
    raise BodyValidationError(
        f"active markup survived {_MAX_SANITIZE_PASSES} sanitize passes "
        f"({len(body)} chars); refusing to emit a partially sanitized body"
    )
