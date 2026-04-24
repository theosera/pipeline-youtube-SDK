"""Tests for #10: log output redacts external-data samples."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline_youtube.sanitize import (
    _redact,
    configure_alert_sink,
    sanitize_untrusted_text,
)


class TestRedact:
    def test_keeps_head_and_hash_tail(self):
        out = _redact("very sensitive transcript contents here", max_len=24)
        assert out.startswith("very sensi")  # head
        assert "[" in out and "]" in out  # hash tail

    def test_short_input_passes_through(self):
        out = _redact("tiny", max_len=24)
        assert "tiny" in out

    def test_empty(self):
        assert _redact("") == ""

    def test_same_input_same_hash(self):
        a = _redact("identical content")
        b = _redact("identical content")
        assert a == b

    def test_different_input_different_hash(self):
        a = _redact("abc def ghi")
        b = _redact("xyz uvw rst")
        assert a != b


class TestAlertSamplesAreRedacted:
    def test_sample_in_jsonl_is_not_verbatim(self, tmp_path: Path):
        sink = tmp_path / "alerts.jsonl"
        configure_alert_sink(sink)
        # Non-trivial removal to trigger the alert
        secret = "confidential-private-transcript-" + ("x" * 40)
        sanitize_untrusted_text(secret + "\u200b" * 10, 1000, context="test")
        configure_alert_sink(None)

        record = json.loads(sink.read_text(encoding="utf-8").splitlines()[0])
        # Full secret must NOT appear verbatim
        assert secret not in record["sample"]
        # Hash fingerprint must be present
        assert "[" in record["sample"] and "]" in record["sample"]
