"""Runtime --provider / --hybrid backend selection (pure, no LLM)."""

from __future__ import annotations

from pipeline_youtube.providers.selection import HEAVY_STAGES, apply_selection

_STAGES = ("router", "stage_02", "stage_04", "alpha", "beta", "leader", "reviewer")
_PROVIDERS = {
    "ollama": {"base_url": "http://localhost:11434/v1", "default_model": "qwen3:8b"},
    "anthropic": {"api_key": "x"},
    "lmstudio": {"base_url": "http://localhost:1234/v1", "default_model": "qwen3-8b"},
}


def _all_ollama() -> dict[str, dict[str, str]]:
    return {s: {"provider": "ollama", "model": "qwen3:8b"} for s in _STAGES}


def test_no_flags_leaves_everything_untouched() -> None:
    models = _all_ollama()
    effective, overrides, warnings = apply_selection(models, _PROVIDERS, _STAGES)
    assert effective == models
    assert overrides == {}  # no flag -> no model-name overlay
    assert warnings == []  # no warning on the config (no-flag) path


def test_provider_anthropic_overrides_provider_and_model_name() -> None:
    effective, overrides, warnings = apply_selection(
        _all_ollama(), _PROVIDERS, _STAGES, provider="anthropic"
    )
    assert all(effective[s] == {"provider": "anthropic", "model": "sonnet"} for s in _STAGES)
    # model-NAME overlay so stages send "sonnet" to anthropic (not qwen3:8b)
    assert all(overrides[s] == "sonnet" for s in _STAGES)
    assert warnings == []


def test_provider_ollama_uses_config_default_model_and_warns() -> None:
    effective, overrides, warnings = apply_selection(
        _all_ollama(), _PROVIDERS, _STAGES, provider="ollama"
    )
    assert all(effective[s] == {"provider": "ollama", "model": "qwen3:8b"} for s in _STAGES)
    assert all(overrides[s] == "qwen3:8b" for s in _STAGES)
    assert warnings  # open provider selected for heavy stages, no --hybrid


def test_provider_ollama_plus_hybrid_keeps_heavy_on_anthropic_no_warning() -> None:
    effective, overrides, warnings = apply_selection(
        _all_ollama(), _PROVIDERS, _STAGES, provider="ollama", hybrid=True
    )
    for s in HEAVY_STAGES:
        assert effective[s] == {"provider": "anthropic", "model": "sonnet"}
        assert overrides[s] == "sonnet"
    assert effective["router"] == {"provider": "ollama", "model": "qwen3:8b"}
    assert overrides["router"] == "qwen3:8b"
    assert warnings == []  # --hybrid lifts the heavy stages


def test_hybrid_alone_lifts_heavy_no_warning() -> None:
    effective, overrides, warnings = apply_selection(
        _all_ollama(), _PROVIDERS, _STAGES, hybrid=True
    )
    for s in HEAVY_STAGES:
        assert effective[s]["provider"] == "anthropic"
        assert overrides[s] == "sonnet"
    # light stages untouched (no provider flag)
    assert "router" not in overrides
    assert warnings == []


def test_provider_default_model_falls_back_when_absent() -> None:
    providers = {"anthropic": {"api_key": "x"}}  # no default_model
    effective, overrides, _ = apply_selection(
        _all_ollama(), providers, _STAGES, provider="anthropic"
    )
    assert effective["leader"] == {"provider": "anthropic", "model": "sonnet"}
    assert overrides["leader"] == "sonnet"


def test_lmstudio_open_provider_also_warns() -> None:
    _e, _o, warnings = apply_selection(_all_ollama(), _PROVIDERS, _STAGES, provider="lmstudio")
    assert warnings


def test_result_is_a_copy() -> None:
    models = _all_ollama()
    effective, _o, _w = apply_selection(models, _PROVIDERS, _STAGES, provider="anthropic")
    effective["leader"]["model"] = "MUTATED"
    assert models["leader"]["model"] == "qwen3:8b"  # input untouched


def test_override_makes_provider_receive_selected_model(monkeypatch) -> None:
    # End-to-end through the registry: with --provider anthropic over an
    # all-Ollama config, the stage must send "sonnet" to the anthropic
    # provider — NOT the config's "qwen3:8b" (the P1 the overlay fixes).
    from pipeline_youtube.providers import registry as registry_mod
    from pipeline_youtube.providers.base import LLMResponse

    captured: dict[str, object] = {}

    class _FakeProvider:
        def invoke(
            self, prompt: str, *, system_prompt: object = None, model: str = "default", **_kw
        ):
            captured["model"] = model
            return LLMResponse(text="ok", model=str(model))

    effective, overrides, _ = apply_selection(
        _all_ollama(), _PROVIDERS, _STAGES, provider="anthropic"
    )
    registry_mod.configure_providers(_PROVIDERS, effective)
    monkeypatch.setitem(registry_mod._provider_cache, "anthropic", _FakeProvider())

    registry_mod.invoke_llm(prompt="hi", model=overrides["stage_02"], role="stage_02")
    assert captured["model"] == "sonnet"
