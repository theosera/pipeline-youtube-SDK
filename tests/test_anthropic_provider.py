"""Tests for the Anthropic provider, focused on prompt caching (Phase 3 B)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from pipeline_youtube.providers.anthropic_sdk import AnthropicProvider, _estimate_cost


class _FakeMessages:
    def __init__(self, usage: Any) -> None:
        self.captured: dict[str, Any] = {}
        self._usage = usage

    def create(self, **kwargs: Any) -> Any:
        self.captured = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(text="ok")],
            usage=self._usage,
            stop_reason="end_turn",
            model_dump=lambda: {},
        )


def _provider(monkeypatch, usage: Any, *, prompt_caching: bool = True) -> AnthropicProvider:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_PROMPT_CACHING", raising=False)
    p = AnthropicProvider(prompt_caching=prompt_caching)
    p._client = SimpleNamespace(messages=_FakeMessages(usage))  # type: ignore[attr-defined]
    return p


class TestPromptCaching:
    def test_system_prompt_sent_as_cache_breakpoint(self, monkeypatch):
        usage = SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=2000,
            cache_read_input_tokens=0,
        )
        p = _provider(monkeypatch, usage)
        resp = p.invoke("hi", system_prompt="big system prompt", model="sonnet")

        system = p._client.messages.captured["system"]  # type: ignore[attr-defined]
        assert isinstance(system, list)
        assert system[0]["cache_control"] == {"type": "ephemeral"}
        assert system[0]["text"] == "big system prompt"
        # cache token counts flow into the response
        assert resp.cache_creation_tokens == 2000
        assert resp.cache_read_tokens == 0

    def test_cache_read_tokens_flow_through(self, monkeypatch):
        usage = SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=2000,
        )
        p = _provider(monkeypatch, usage)
        resp = p.invoke("hi", system_prompt="big system prompt", model="sonnet")
        assert resp.cache_read_tokens == 2000

    def test_disabled_sends_plain_system_string(self, monkeypatch):
        usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        p = _provider(monkeypatch, usage, prompt_caching=False)
        p.invoke("hi", system_prompt="plain", model="sonnet")
        assert p._client.messages.captured["system"] == "plain"  # type: ignore[attr-defined]

    def test_env_var_disables_caching(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("ANTHROPIC_PROMPT_CACHING", "0")
        p = AnthropicProvider()
        usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        p._client = SimpleNamespace(messages=_FakeMessages(usage))  # type: ignore[attr-defined]
        p.invoke("hi", system_prompt="plain", model="sonnet")
        assert p._client.messages.captured["system"] == "plain"  # type: ignore[attr-defined]

    def test_no_system_prompt_omits_system(self, monkeypatch):
        usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        p = _provider(monkeypatch, usage)
        p.invoke("hi", model="sonnet")
        assert "system" not in p._client.messages.captured  # type: ignore[attr-defined]


class TestCacheCostEstimation:
    def test_cache_read_is_cheaper_than_fresh_input(self):
        model = "claude-sonnet-4-20250514"
        fresh = _estimate_cost(model, 1000, 0)
        cached = _estimate_cost(model, 0, 0, cache_read_tokens=1000)
        assert fresh is not None and cached is not None
        # cache read billed at 0.1x base input rate
        assert cached == pytest.approx(fresh * 0.10)

    def test_cache_write_premium(self):
        model = "claude-sonnet-4-20250514"
        fresh = _estimate_cost(model, 1000, 0)
        written = _estimate_cost(model, 0, 0, cache_creation_tokens=1000)
        assert fresh is not None and written is not None
        assert written == pytest.approx(fresh * 1.25)

    def test_unknown_model_returns_none(self):
        assert _estimate_cost("mystery-model", 100, 100) is None
