"""Whisper bounded-concurrency + in-process model cache (Phase 2)."""

from __future__ import annotations

import sys
import threading
import types

from pipeline_youtube.transcript import whisper_fallback as wf


class TestConcurrencyConfig:
    def test_default_semaphore_is_bounded(self):
        assert isinstance(wf._whisper_semaphore, threading.BoundedSemaphore)

    def test_configure_sets_bound(self):
        try:
            wf.configure_whisper_concurrency(3)
            sem = wf._whisper_semaphore
            acquired = [sem.acquire(blocking=False) for _ in range(3)]
            assert all(acquired)
            assert sem.acquire(blocking=False) is False  # bound is 3
            for _ in range(3):
                sem.release()
        finally:
            wf.configure_whisper_concurrency(wf.DEFAULT_WHISPER_CONCURRENCY)

    def test_configure_floors_at_one(self):
        try:
            wf.configure_whisper_concurrency(0)
            sem = wf._whisper_semaphore
            assert sem.acquire(blocking=False) is True
            assert sem.acquire(blocking=False) is False
            sem.release()
        finally:
            wf.configure_whisper_concurrency(wf.DEFAULT_WHISPER_CONCURRENCY)


class TestModelCache:
    def test_model_loaded_once_per_name(self, monkeypatch):
        load_calls = {"n": 0}

        fake = types.ModuleType("whisper")
        fake._MODELS = {}  # makes integrity check skip silently

        def fake_load(name):
            load_calls["n"] += 1
            return f"model::{name}"

        fake.load_model = fake_load
        monkeypatch.setitem(sys.modules, "whisper", fake)
        monkeypatch.setattr(wf, "_model_cache", {})

        m1 = wf._load_model_cached("small")
        m2 = wf._load_model_cached("small")
        assert m1 == m2 == "model::small"
        assert load_calls["n"] == 1  # memoized

        wf._load_model_cached("tiny")
        assert load_calls["n"] == 2  # distinct model name reloads
