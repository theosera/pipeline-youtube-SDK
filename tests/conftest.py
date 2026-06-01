"""Shared test fixtures.

Cache isolation: the persistent cache (``pipeline_youtube.cache``) is a
process-wide singleton that defaults to ``~/.cache/pipeline-youtube``. Left
unmanaged it would (a) write into the developer's real home dir during tests
and (b) leak enabled-state and cached values between tests. The autouse
fixture below redirects the default root to a per-test temp dir and resets
the singleton around every test, so each test starts from a clean, disabled
cache exactly like a fresh process.
"""

from __future__ import annotations

import pytest

from pipeline_youtube.cache import reset_cache


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path_factory, monkeypatch):
    cache_root = tmp_path_factory.mktemp("pyt-cache")
    monkeypatch.setenv("PIPELINE_YOUTUBE_CACHE", str(cache_root))
    reset_cache()
    yield
    reset_cache()
