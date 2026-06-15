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


def test_no_flags_passes_config_through_unchanged() -> None:
    models = _all_ollama()
    effective, warnings = apply_selection(models, _PROVIDERS, _STAGES)
    assert effective == models
    # heavy stages on ollama -> advisory warning
    assert warnings and "重い工程" in warnings[0]


def test_no_flags_with_anthropic_heavy_has_no_warning() -> None:
    models = _all_ollama()
    models["stage_04"] = {"provider": "anthropic", "model": "sonnet"}
    models["leader"] = {"provider": "anthropic", "model": "sonnet"}
    _effective, warnings = apply_selection(models, _PROVIDERS, _STAGES)
    assert warnings == []


def test_provider_anthropic_overrides_all_stages() -> None:
    effective, warnings = apply_selection(_all_ollama(), _PROVIDERS, _STAGES, provider="anthropic")
    assert all(effective[s] == {"provider": "anthropic", "model": "sonnet"} for s in _STAGES)
    assert warnings == []


def test_provider_ollama_uses_config_default_model_and_warns() -> None:
    effective, warnings = apply_selection(_all_ollama(), _PROVIDERS, _STAGES, provider="ollama")
    assert all(effective[s] == {"provider": "ollama", "model": "qwen3:8b"} for s in _STAGES)
    assert warnings  # heavy stages are local


def test_provider_ollama_plus_hybrid_keeps_heavy_on_anthropic() -> None:
    effective, warnings = apply_selection(
        _all_ollama(), _PROVIDERS, _STAGES, provider="ollama", hybrid=True
    )
    for s in HEAVY_STAGES:
        assert effective[s] == {"provider": "anthropic", "model": "sonnet"}
    # a light stage stays local
    assert effective["router"] == {"provider": "ollama", "model": "qwen3:8b"}
    assert warnings == []  # heavy stages no longer open


def test_hybrid_alone_lifts_heavy_from_config_local() -> None:
    effective, warnings = apply_selection(_all_ollama(), _PROVIDERS, _STAGES, hybrid=True)
    for s in HEAVY_STAGES:
        assert effective[s]["provider"] == "anthropic"
    assert warnings == []


def test_provider_default_model_falls_back_when_absent() -> None:
    # providers cfg without default_model for anthropic -> built-in fallback "sonnet"
    providers = {"anthropic": {"api_key": "x"}}
    effective, _ = apply_selection(_all_ollama(), providers, _STAGES, provider="anthropic")
    assert effective["leader"] == {"provider": "anthropic", "model": "sonnet"}


def test_legacy_string_anthropic_alias_not_flagged() -> None:
    models = _all_ollama()
    models["leader"] = "haiku"  # legacy string form, anthropic alias
    models["stage_04"] = {"provider": "anthropic", "model": "sonnet"}
    _effective, warnings = apply_selection(models, _PROVIDERS, _STAGES)
    assert warnings == []


def test_result_is_a_copy() -> None:
    models = _all_ollama()
    effective, _ = apply_selection(models, _PROVIDERS, _STAGES, provider="anthropic")
    effective["leader"]["model"] = "MUTATED"
    assert models["leader"]["model"] == "qwen3:8b"  # input untouched
