"""Tests for the content-addressed persistent cache (pipeline_youtube.cache)."""

from __future__ import annotations

import os
from pathlib import Path

from pipeline_youtube.cache import (
    Cache,
    configure_cache,
    llm_key,
    resolve_cache_root,
    url_key,
)


class TestRootResolution:
    def test_explicit_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PIPELINE_YOUTUBE_CACHE", str(tmp_path / "env"))
        assert resolve_cache_root(tmp_path / "explicit") == tmp_path / "explicit"

    def test_env_over_xdg(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PIPELINE_YOUTUBE_CACHE", str(tmp_path / "env"))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        assert resolve_cache_root(None) == tmp_path / "env"

    def test_xdg_over_home(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PIPELINE_YOUTUBE_CACHE", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        assert resolve_cache_root(None) == tmp_path / "xdg" / "pipeline-youtube"

    def test_home_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PIPELINE_YOUTUBE_CACHE", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
        assert resolve_cache_root(None) == tmp_path / "home" / ".cache" / "pipeline-youtube"


class TestKeys:
    def test_llm_key_is_stable_and_64_hex(self):
        k = llm_key("anthropic", "sonnet", "sys", "prompt")
        assert k == llm_key("anthropic", "sonnet", "sys", "prompt")
        assert len(k) == 64 and all(c in "0123456789abcdef" for c in k)

    def test_llm_key_model_swap_differs(self):
        assert llm_key("anthropic", "sonnet", "s", "p") != llm_key("anthropic", "haiku", "s", "p")

    def test_llm_key_nul_separation_prevents_boundary_collision(self):
        # Without a separator, ("ab","c") and ("a","bc") would collide.
        assert llm_key("ab", "c", None, "x") != llm_key("a", "bc", None, "x")

    def test_llm_key_none_system_matches_empty(self):
        assert llm_key("p", "m", None, "x") == llm_key("p", "m", "", "x")

    def test_url_key(self):
        assert len(url_key("https://example.com")) == 64


class TestDisabledCache:
    def test_disabled_always_misses_and_drops(self, tmp_path):
        c = Cache(tmp_path, enabled=False)
        assert not c.enabled
        c.put_transcript("v", "official", "ja", {"x": 1})
        c.put_llm("k", {"text": "t"})
        assert c.get_transcript("v", "official", "ja") is None
        assert c.get_llm("k") is None
        # Nothing written to disk.
        assert not any(tmp_path.iterdir())


class TestRoundTrip:
    def _cache(self, tmp_path) -> Cache:
        return Cache(tmp_path, enabled=True)

    def test_transcript_round_trip(self, tmp_path):
        c = self._cache(tmp_path)
        obj = {"video_id": "v", "snippets": [{"text": "hi"}]}
        c.put_transcript("v", "whisper", "ja", obj)
        assert c.get_transcript("v", "whisper", "ja") == obj

    def test_namespaces_isolated(self, tmp_path):
        c = self._cache(tmp_path)
        c.put_transcript("v", "official", "ja", {"a": 1})
        assert c.get_transcript("v", "official", "en") is None  # different lang
        assert c.get_transcript("v", "auto", "ja") is None  # different tier

    def test_llm_and_code_fetch_sharded(self, tmp_path):
        c = self._cache(tmp_path)
        key = llm_key("p", "m", "s", "prompt")
        c.put_llm(key, {"text": "out"})
        assert c.get_llm(key) == {"text": "out"}
        # 2-char shard dir exists.
        assert (tmp_path / "llm" / key[:2] / key).exists()
        ck = url_key("https://x/y")
        c.put_code_fetch(ck, [{"filename": "a.py"}])
        assert c.get_code_fetch(ck) == [{"filename": "a.py"}]

    def test_atomic_write_leaves_no_temp(self, tmp_path):
        c = self._cache(tmp_path)
        c.put_transcript("v", "official", "ja", {"a": 1})
        tmps = [p for p in (tmp_path / "transcript").rglob("*") if p.name.startswith(".tmp-")]
        assert tmps == []

    def test_unserializable_value_skipped(self, tmp_path):
        c = self._cache(tmp_path)
        c.put_llm("k", {"bad": object()})  # not JSON-serializable → silently skipped
        assert c.get_llm("k") is None

    def test_traversal_segment_is_neutralized(self, tmp_path):
        c = self._cache(tmp_path)
        c.put_transcript("../../etc", "official", "ja", {"a": 1})
        # The escape attempt is sanitized; nothing is written outside root.
        assert c.get_transcript("../../etc", "official", "ja") == {"a": 1}
        for p in tmp_path.rglob("*"):
            assert tmp_path in p.resolve().parents or p.resolve() == tmp_path


class TestVideoCache:
    def test_put_get_round_trip(self, tmp_path):
        c = Cache(tmp_path, enabled=True)
        src = tmp_path / "src.mp4"
        src.write_bytes(b"video-bytes")
        c.put_video("vid", "480", src)
        cached = c.get_video("vid", "480")
        assert cached is not None and cached.read_bytes() == b"video-bytes"

    def test_get_miss_returns_none(self, tmp_path):
        assert Cache(tmp_path, enabled=True).get_video("nope", "480") is None

    def test_lru_eviction_keeps_recently_used(self, tmp_path):
        # Cap at ~25 bytes; each blob is 10 bytes → 3 won't fit.
        c = Cache(tmp_path, enabled=True, max_video_bytes=25)
        for i in range(3):
            src = tmp_path / f"s{i}.mp4"
            src.write_bytes(b"0123456789")
            c.put_video(f"v{i}", "480", src)
            # Touch v0 so it stays "recently used" via atime bump on read.
            os.utime(src, None)
            if c.get_video("v0", "480") is not None:
                pass
        video_root = tmp_path / "video"
        total = sum(p.stat().st_size for p in video_root.rglob("*") if p.is_file())
        assert total <= 25
        # v0 was kept warm, so it should survive while an older one is evicted.
        assert c.get_video("v0", "480") is not None


class TestConfigureCache:
    def test_configure_enabled_uses_env_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PIPELINE_YOUTUBE_CACHE", str(tmp_path / "root"))
        c = configure_cache(None, enabled=True)
        assert c.enabled and c.root == tmp_path / "root"

    def test_configure_disabled(self):
        c = configure_cache(None, enabled=False)
        assert not c.enabled
