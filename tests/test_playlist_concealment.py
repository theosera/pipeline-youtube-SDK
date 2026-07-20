"""Fetch-boundary concealment alerting in playlist.fetch_metadata.

yt_dlp is stubbed so no network is touched; concealment code points are built
with ``chr()`` to keep this source free of literal invisible glyphs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import pipeline_youtube.playlist as playlist_mod
from pipeline_youtube.services.sanitize import configure_alert_sink

ZWSP = chr(0x200B)
RLO = chr(0x202E)
CYR_A = chr(0x430)

_VALID_URL = "https://www.youtube.com/playlist?list=PLabcdefghijklmnop"


class _FakeYDL:
    def __init__(self, info: dict) -> None:
        self._info = info

    def __enter__(self) -> _FakeYDL:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def extract_info(self, url: str, download: bool = False) -> dict:
        return self._info


def _patch_ydl(monkeypatch: pytest.MonkeyPatch, info: dict) -> None:
    monkeypatch.setattr(playlist_mod.yt_dlp, "YoutubeDL", lambda opts: _FakeYDL(info))


def teardown_function() -> None:
    configure_alert_sink(None)


def test_invisible_title_alerts_but_metadata_kept_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = {
        "_type": "playlist",
        "title": "Clean Playlist",
        "entries": [{"id": "vid123", "title": f"hello{ZWSP}{ZWSP}world"}],
    }
    _patch_ydl(monkeypatch, info)
    sink = tmp_path / "alerts.jsonl"
    configure_alert_sink(sink)

    videos = playlist_mod.fetch_metadata(_VALID_URL)

    # VideoMeta keeps the raw title (stripping happens at the naming chokepoint).
    assert videos[0].title == f"hello{ZWSP}{ZWSP}world"
    text = sink.read_text(encoding="utf-8")
    records = [json.loads(line) for line in text.splitlines()]
    assert any(r["context"] == "playlist.fetch.title:vid123" for r in records)
    assert any("invisible" in r["message"] for r in records)
    # no plaintext title fragment leaks into the log.
    assert "hello" not in text


def test_mixed_script_title_alerts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    info = {"id": "solo1", "title": f"{CYR_A}pple review"}
    _patch_ydl(monkeypatch, info)
    sink = tmp_path / "alerts.jsonl"
    configure_alert_sink(sink)

    playlist_mod.fetch_metadata("https://www.youtube.com/watch?v=solo1")

    records = [json.loads(line) for line in sink.read_text(encoding="utf-8").splitlines()]
    hit = [r for r in records if r["context"] == "playlist.fetch.title:solo1"]
    assert hit and "mixed-script token(s)" in hit[0]["message"]


def test_clean_metadata_produces_no_alert(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    info = {
        "_type": "playlist",
        "title": "AI Engineering",
        "entries": [{"id": "ok1", "title": "The AI System Most People Aren't Building"}],
    }
    _patch_ydl(monkeypatch, info)
    sink = tmp_path / "alerts.jsonl"
    configure_alert_sink(sink)

    videos = playlist_mod.fetch_metadata(_VALID_URL)

    assert videos[0].video_id == "ok1"
    assert not sink.exists()  # no signal -> sink never written
