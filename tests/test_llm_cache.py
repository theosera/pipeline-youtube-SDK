"""Per-role LLM-output cache policy and transcript-cache integration."""

from __future__ import annotations

import pytest

from pipeline_youtube import cache as cache_mod
from pipeline_youtube.providers import registry
from pipeline_youtube.providers.base import LLMResponse
from pipeline_youtube.transcript.base import (
    TranscriptNotAvailable,
    TranscriptSource,
    build_result,
    fetch_with_fallback,
)


class _CountingProvider:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    def invoke(self, prompt, **kw) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=f"out-{self.calls}", model="fake-model", provider="fake")


@pytest.fixture
def provider(monkeypatch, tmp_path):
    """Enabled cache + a counting provider wired into the registry.

    The configured ``Cache`` is attached as ``provider.cache`` so tests inject
    it explicitly into ``invoke_llm`` (the LLM-output cache under test).
    """
    p = _CountingProvider()
    p.cache = cache_mod.configure_cache(tmp_path / "c", enabled=True)
    monkeypatch.setattr(registry, "get_provider", lambda name: p)
    # Resolve any role to ("fake", "fake-model") so invoke_llm doesn't touch config.
    monkeypatch.setattr(registry, "resolve_role", lambda role: ("fake", "fake-model"))
    return p


class TestPerRolePolicy:
    def test_stage_02_cached_by_default(self, provider):
        registry.configure_llm_cache(stages=True, synthesis=False)
        r1 = registry.invoke_llm(
            "p", system_prompt="s", role="stage_02", provider_name="fake", cache=provider.cache
        )
        r2 = registry.invoke_llm(
            "p", system_prompt="s", role="stage_02", provider_name="fake", cache=provider.cache
        )
        assert provider.calls == 1  # second call served from cache
        assert r1.text == r2.text

    def test_stage_01_correct_cached_by_default(self, provider):
        # Stage 01b correction is a paid web-search call — re-runs must hit the
        # LLM cache instead of repeating it.
        registry.configure_llm_cache(stages=True, synthesis=False)
        registry.invoke_llm(
            "p",
            system_prompt="s",
            role="stage_01_correct",
            provider_name="fake",
            cache=provider.cache,
        )
        registry.invoke_llm(
            "p",
            system_prompt="s",
            role="stage_01_correct",
            provider_name="fake",
            cache=provider.cache,
        )
        assert provider.calls == 1  # second call served from cache

    def test_synthesis_not_cached_by_default(self, provider):
        registry.configure_llm_cache(stages=True, synthesis=False)
        registry.invoke_llm(
            "p", system_prompt="s", role="alpha", provider_name="fake", cache=provider.cache
        )
        registry.invoke_llm(
            "p", system_prompt="s", role="alpha", provider_name="fake", cache=provider.cache
        )
        assert provider.calls == 2  # synthesis fresh each time

    def test_synthesis_cached_when_opted_in(self, provider):
        registry.configure_llm_cache(stages=True, synthesis=True)
        registry.invoke_llm(
            "p", system_prompt="s", role="leader", provider_name="fake", cache=provider.cache
        )
        registry.invoke_llm(
            "p", system_prompt="s", role="leader", provider_name="fake", cache=provider.cache
        )
        assert provider.calls == 1

    def test_unknown_role_never_cached(self, provider):
        registry.configure_llm_cache(stages=True, synthesis=True)
        registry.invoke_llm("p", role=None, provider_name="fake", cache=provider.cache)
        registry.invoke_llm("p", role=None, provider_name="fake", cache=provider.cache)
        assert provider.calls == 2

    def test_different_prompt_misses(self, provider):
        registry.configure_llm_cache(stages=True, synthesis=False)
        registry.invoke_llm(
            "p1", system_prompt="s", role="stage_04", provider_name="fake", cache=provider.cache
        )
        registry.invoke_llm(
            "p2", system_prompt="s", role="stage_04", provider_name="fake", cache=provider.cache
        )
        assert provider.calls == 2

    def test_multi_turn_bypasses_cache(self, provider):
        registry.configure_llm_cache(stages=True, synthesis=False)
        msgs = [{"role": "user", "content": "hi"}]
        registry.invoke_llm(
            "p", role="stage_02", provider_name="fake", messages=msgs, cache=provider.cache
        )
        registry.invoke_llm(
            "p", role="stage_02", provider_name="fake", messages=msgs, cache=provider.cache
        )
        assert provider.calls == 2

    def test_disabled_cache_master_switch(self, monkeypatch, tmp_path):
        cache = cache_mod.configure_cache(None, enabled=False)  # --no-cache
        p = _CountingProvider()
        monkeypatch.setattr(registry, "get_provider", lambda name: p)
        registry.configure_llm_cache(stages=True, synthesis=True)
        registry.invoke_llm(
            "p", system_prompt="s", role="stage_02", provider_name="fake", cache=cache
        )
        registry.invoke_llm(
            "p", system_prompt="s", role="stage_02", provider_name="fake", cache=cache
        )
        assert p.calls == 2


def _ok_fetcher(source):
    def _f(video_id, languages):
        return build_result(video_id, source, languages[0], [])

    return _f


def _fail_fetcher(msg):
    def _f(video_id, languages):
        raise TranscriptNotAvailable(msg)

    return _f


class TestTranscriptCacheIntegration:
    def test_second_run_hits_cache_without_calling_fetcher(self, tmp_path):
        cache = cache_mod.configure_cache(tmp_path / "c", enabled=True)
        calls = {"n": 0}

        def counting(video_id, languages):
            calls["n"] += 1
            return build_result(video_id, TranscriptSource.WHISPER, languages[0], [])

        fetchers = [("whisper", counting)]
        r1 = fetch_with_fallback("vidX", ["ja"], fetchers, cache=cache)
        r2 = fetch_with_fallback("vidX", ["ja"], fetchers, cache=cache)
        assert calls["n"] == 1  # second run served from cache
        assert r1.source == r2.source == TranscriptSource.WHISPER

    def test_tier_ordering_preserved_with_cache(self, tmp_path):
        cache = cache_mod.configure_cache(tmp_path / "c", enabled=True)
        fetchers = [
            ("official", _fail_fetcher("no")),
            ("auto", _ok_fetcher(TranscriptSource.AUTO)),
        ]
        result = fetch_with_fallback("vidY", ["ja"], fetchers, cache=cache)
        assert result.source == TranscriptSource.AUTO
        # Re-run still yields AUTO from cache (official remains unavailable).
        assert fetch_with_fallback("vidY", ["ja"], fetchers, cache=cache).source == (
            TranscriptSource.AUTO
        )

    def test_disabled_cache_always_calls_fetcher(self, tmp_path):
        cache = cache_mod.configure_cache(None, enabled=False)
        calls = {"n": 0}

        def counting(video_id, languages):
            calls["n"] += 1
            return build_result(video_id, TranscriptSource.WHISPER, languages[0], [])

        fetch_with_fallback("v", ["ja"], [("whisper", counting)], cache=cache)
        fetch_with_fallback("v", ["ja"], [("whisper", counting)], cache=cache)
        assert calls["n"] == 2
