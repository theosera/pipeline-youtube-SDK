"""Tests for L3: sanitize alert sink behavior."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline_youtube import sanitize as sanitize_mod
from pipeline_youtube.sanitize import configure_alert_sink, sanitize_untrusted_text


class TestAlertSink:
    def teardown_method(self):
        configure_alert_sink(None)

    def test_no_sink_no_alerts(self, tmp_path: Path):
        configure_alert_sink(None)
        sink = tmp_path / "alerts.jsonl"
        sanitize_untrusted_text("\u200b" * 10 + "clean", 100)
        assert not sink.exists()

    def test_sink_records_large_removals(self, tmp_path: Path):
        sink = tmp_path / "alerts.jsonl"
        configure_alert_sink(sink)
        sanitize_untrusted_text("\u200b" * 10 + "clean", 100, context="test.case")
        assert sink.exists()
        record = json.loads(sink.read_text(encoding="utf-8").splitlines()[0])
        assert record["context"] == "test.case"
        assert record["removed"] == 10
        assert record["after_len"] == len("clean")

    def test_sink_ignores_small_removals(self, tmp_path: Path):
        sink = tmp_path / "alerts.jsonl"
        configure_alert_sink(sink)
        sanitize_untrusted_text("ab\u200bc", 100, context="small")
        assert not sink.exists()

    def test_sink_ignores_length_truncation(self, tmp_path: Path):
        sink = tmp_path / "alerts.jsonl"
        configure_alert_sink(sink)
        sanitize_untrusted_text("a" * 1000, 10, context="truncate")
        assert not sink.exists()

    def test_sink_recoverable_on_bad_path(self, tmp_path: Path):
        bad = tmp_path / "readonly.jsonl"
        bad.write_text("")
        bad.chmod(0o400)
        configure_alert_sink(bad)
        try:
            sanitize_untrusted_text("\u200b" * 10 + "x", 100)
        finally:
            bad.chmod(0o600)
        assert sanitize_mod._alert_sink is bad
