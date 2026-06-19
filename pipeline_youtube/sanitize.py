"""Backward-compat shim: moved to ``pipeline_youtube.services.sanitize``.

新規コードは ``services.sanitize`` を直接参照すること。
"""

from __future__ import annotations

from .services.sanitize import (
    UNTRUSTED_CLOSE,
    UNTRUSTED_OPEN,
    _redact,
    configure_alert_sink,
    sanitize_untrusted_text,
    wrap_untrusted,
)

__all__ = [
    "UNTRUSTED_CLOSE",
    "UNTRUSTED_OPEN",
    "_redact",
    "configure_alert_sink",
    "sanitize_untrusted_text",
    "wrap_untrusted",
]
