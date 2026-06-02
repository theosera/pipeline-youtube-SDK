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


class TestPrefetchSingleOwner:
    """Regression: a prefetch throttled by the download semaphore must not
    complete prematurely.

    main.py waits to completion (``wait(timeout=None)``) so the prefetch stays
    the single writer of ``tmp/<video_id>.mp4``. With a fixed timeout, a queued
    prefetch outlives the wait while still alive, and the capture fallback then
    re-downloads the same path — the two downloads race on unlink/overwrite.
    """

    def test_throttled_prefetch_is_not_abandoned(self):
        from pipeline_youtube.playlist import VideoMeta
        from pipeline_youtube.stages.capture import prefetch_video_download

        video = VideoMeta(
            video_id="zzz9999999",
            title="t",
            url="https://www.youtube.com/watch?v=zzz9999999",
            duration=60,
            channel="ch",
            upload_date=None,
            playlist_title=None,
        )

        calls = {"n": 0}

        class _CountingBackend:
            def download_video(self, url: str, dest: Path, *, resolution: str = "480") -> None:
                calls["n"] += 1
                dest.write_bytes(b"mp4")

        cap.configure_download_concurrency(1)
        sem = cap._download_semaphore
        assert sem is not None
        handle = None
        try:
            # Occupy the only download slot, as another in-flight download would.
            sem.acquire()
            try:
                handle = prefetch_video_download(video, backend=_CountingBackend())
                # Queued behind the held semaphore: a fixed timeout abandons it
                # while the thread is still alive (the bug). No download yet.
                assert isinstance(handle.wait(timeout=0.2), TimeoutError)
                assert calls["n"] == 0
            finally:
                sem.release()
            # Owner waits to completion: the download runs exactly once.
            assert handle.wait(timeout=None) is None
            assert handle.path.exists()
            assert calls["n"] == 1
        finally:
            if handle is not None:
                handle.path.unlink(missing_ok=True)
            cap.configure_download_concurrency(None)


class TestConfigureAcceptsNonPositive:
    def test_zero_and_negative_clear_caps(self):
        reg.configure_llm_concurrency(0)
        assert reg._llm_semaphore is None
        reg.configure_llm_concurrency(-3)
        assert reg._llm_semaphore is None
        cap.configure_download_concurrency(0)
        assert cap._download_semaphore is None
