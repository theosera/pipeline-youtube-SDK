"""Tests for Phase 3 A: resource-class semaphores (LLM + download caps).

The caps bound concurrent usage of a resource independently of the
per-video --concurrency, and default to unbounded (prior behavior).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from pipeline_youtube.providers import registry as reg
from pipeline_youtube.providers.base import LLMResponse
from pipeline_youtube.stages import capture as cap


class _ConcurrencyProbe:
    """Tracks the peak number of threads inside the guarded region."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def enter(self) -> None:
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)

    def exit(self) -> None:
        with self._lock:
            self.active -= 1


def _run_threads(fn, n: int) -> None:
    threads = [threading.Thread(target=fn) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


class TestLLMConcurrencyCap:
    def _patch_provider(self, monkeypatch, probe: _ConcurrencyProbe) -> None:
        class _FakeProvider:
            provider_name = "fake"

            def invoke(self, prompt: str, **kw: Any) -> LLMResponse:
                probe.enter()
                try:
                    time.sleep(0.05)
                finally:
                    probe.exit()
                return LLMResponse(text="ok", model="m", provider="fake")

        monkeypatch.setattr(reg, "get_provider", lambda name: _FakeProvider())

    def test_cap_limits_concurrent_calls(self, monkeypatch):
        probe = _ConcurrencyProbe()
        self._patch_provider(monkeypatch, probe)
        reg.configure_llm_concurrency(2)
        try:
            _run_threads(lambda: reg.invoke_llm("hi", provider_name="fake"), 6)
        finally:
            reg.configure_llm_concurrency(None)
        assert probe.peak <= 2

    def test_unbounded_by_default(self, monkeypatch):
        probe = _ConcurrencyProbe()
        self._patch_provider(monkeypatch, probe)
        reg.configure_llm_concurrency(None)
        _run_threads(lambda: reg.invoke_llm("hi", provider_name="fake"), 5)
        # Without a cap, several calls overlap.
        assert probe.peak >= 2


class _FakeBackend:
    def __init__(self, probe: _ConcurrencyProbe) -> None:
        self._probe = probe

    def download_video(self, url: str, dest: Path, *, resolution: str = "480") -> None:
        self._probe.enter()
        try:
            time.sleep(0.05)
        finally:
            self._probe.exit()


class TestDownloadConcurrencyCap:
    def test_cap_limits_concurrent_downloads(self, tmp_path: Path):
        probe = _ConcurrencyProbe()
        backend = _FakeBackend(probe)
        cap.configure_download_concurrency(2)
        try:
            _run_threads(
                lambda: cap._download_video("http://x", tmp_path / "v.mp4", backend=backend),
                6,
            )
        finally:
            cap.configure_download_concurrency(None)
        assert probe.peak <= 2

    def test_unbounded_by_default(self, tmp_path: Path):
        probe = _ConcurrencyProbe()
        backend = _FakeBackend(probe)
        cap.configure_download_concurrency(None)
        _run_threads(
            lambda: cap._download_video("http://x", tmp_path / "v.mp4", backend=backend),
            5,
        )
        assert probe.peak >= 2


class TestConfigureAcceptsNonPositive:
    def test_zero_and_negative_clear_caps(self):
        reg.configure_llm_concurrency(0)
        assert reg._llm_semaphore is None
        reg.configure_llm_concurrency(-3)
        assert reg._llm_semaphore is None
        cap.configure_download_concurrency(0)
        assert cap._download_semaphore is None
