"""Shared test fixtures.

Cache isolation: ``configure_cache(root=None)`` resolves its root from
``PIPELINE_YOUTUBE_CACHE`` (else ``~/.cache/pipeline-youtube``). The autouse
fixture below redirects that default to a per-test temp dir so a test that
builds an *enabled* cache never writes into the developer's real home dir.
The cache is no longer a process-global singleton (it is injected via
``Runtime.cache``), so there is nothing to reset between tests.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path_factory, monkeypatch):
    cache_root = tmp_path_factory.mktemp("pyt-cache")
    monkeypatch.setenv("PIPELINE_YOUTUBE_CACHE", str(cache_root))
    yield
