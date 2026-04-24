"""Prompt injection mitigation, ported from pipeline/classifier.ts:209-216.

Block-list regex patterns are easily bypassed, so the defense relies on:
  1. Structural sanitization (strip control chars, zero-width unicode)
  2. Strict length cap (reduce payload surface)
  3. XML delimiter wrapping (<untrusted_content>) + explicit prompt policy
  4. Downstream AI-output structural validation

This module handles (1), (2), (3). Downstream validation lives in the
AI provider modules.

Sanitization alerts
-------------------
When sanitization removes a non-trivial amount of content (potential
injection attempt), an entry is appended to the configured alert sink
(JSONL). The sink is opt-in: call `configure_alert_sink(path)` at
startup (e.g. from `main.py:cli()`). Tests that do not configure a
sink produce no I/O.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

# Keep \t (\x09) and \n (\x0a) — these are legitimate content.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Zero-width and invisible Unicode: ZWSP, ZWNJ, ZWJ, LRM, RLM, LS, PS,
# various directional marks, word joiner, BOM, interlinear annotations.
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u2028-\u202f\u2060\ufeff\ufff9-\ufffb]")

UNTRUSTED_OPEN = "<untrusted_content>"
UNTRUSTED_CLOSE = "</untrusted_content>"

# Alert when at least this many characters are stripped by the filter
# layers (not counting post-filter length capping).
_ALERT_REMOVED_THRESHOLD = 5

_alert_sink: Path | None = None


def configure_alert_sink(path: Path | None) -> None:
    """Set the JSONL file path for sanitization alerts. None disables."""
    global _alert_sink
    _alert_sink = path


def sanitize_untrusted_text(
    raw: str | None,
    max_length: int,
    *,
    context: str = "unknown",
) -> str:
    """Strip control / zero-width / null bytes and cap length.

    Preserves tabs and newlines (legitimate in transcripts/summaries).
    If the filtering stage removes ≥ 5 characters, an alert is recorded
    (no-op unless `configure_alert_sink()` was called).
    """
    if not raw or not isinstance(raw, str):
        return ""
    before_len = len(raw)
    cleaned = _CONTROL_CHARS_RE.sub("", raw)
    cleaned = _ZERO_WIDTH_RE.sub("", cleaned)
    cleaned = cleaned.replace("\x00", "")
    removed = before_len - len(cleaned)
    if removed >= _ALERT_REMOVED_THRESHOLD:
        _emit_alert(context, before_len, len(cleaned), raw[:64])
    return cleaned[:max_length]


def wrap_untrusted(content: str) -> str:
    """Wrap sanitized content in <untrusted_content> delimiter for AI prompts.

    The prompt-side system policy must explicitly instruct the model to
    treat anything inside these tags as data, not instructions.
    """
    return f"{UNTRUSTED_OPEN}\n{content}\n{UNTRUSTED_CLOSE}"


def _redact(sample: str, max_len: int = 24) -> str:
    """Return a short, non-leaky fingerprint of `sample` for alert logs.

    Keeps the first `max_len // 2` chars + a hash tail. The full string
    stays out of disk so transcript/title fragments don't leak via
    shared log files (see top-10 #10).
    """
    import hashlib

    if not sample:
        return ""
    half = max(max_len // 2, 4)
    head = sample[:half]
    digest = hashlib.sha1(sample.encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"{head}…[{digest}]"


def _emit_alert(context: str, before_len: int, after_len: int, sample: str) -> None:
    if _alert_sink is None:
        return
    try:
        _alert_sink.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "context": context,
            "before_len": before_len,
            "after_len": after_len,
            "removed": before_len - after_len,
            "sample": _redact(sample),
        }
        with _alert_sink.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass
