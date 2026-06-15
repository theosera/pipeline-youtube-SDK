"""Runtime --provider / --hybrid backend selection (pure, no LLM).

The CLI derives each stage's explicit ``model=`` name from the SAME
``effective_models`` via ``registry.resolve_role``, so these tests pin both
``apply_selection`` (pure) and the end-to-end resolution through the registry
(which is what fixes the object-config "dict-as-model" bug).
"""

from __future__ import annotations

from pipeline_youtube.providers import registry as registry_mod
from pipeline_youtube.providers.base import LLMResponse
from pipeline_youtube.providers.selection import HEAVY_STAGES, apply_selection

_STAGES = ("router", "stage_02", "stage_04", "alpha", "beta", "leader", "reviewer")
_PROVIDERS = {
    "ollama": {"base_url": "http://localhost:11434/v1", "default_model": "qwen3:8b"},
    "anthropic": {"api_key": "x"},
    "lmstudio": {"base_url": "http://localhost:1234/v1", "default_model": "qwen3-8b"},
}


def _all_ollama() -> dict[str, dict[str, str]]:
    return {s: {"provider": "ollama", "model": "qwen3:8b"} for s in _STAGES}


def test_no_flags_leaves_effective_unchanged_no_warning() -> None:
    models = _all_ollama()
    effective, warnings = apply_selection(models, _PROVIDERS, _STAGES)
    # stage_01_correct is always pinned to Anthropic (web-search correction is
    # Anthropic-only); everything else is left untouched on the no-flag path.
    assert effective == {**models, "stage_01_correct": {"provider": "anthropic", "model": "opus"}}
    assert warnings == []  # no warning on the config (no-flag) path


def test_stage_01_correct_always_pinned_to_anthropic() -> None:
    # Even with --provider ollama (no --hybrid), correction must stay Anthropic,
    # and must NOT carry the local model name (qwen3:8b) — it defaults to opus.
    effective, _ = apply_selection(_all_ollama(), _PROVIDERS, _STAGES, provider="ollama")
    assert effective["stage_01_correct"] == {"provider": "anthropic", "model": "opus"}


def test_stage_01_correct_preserves_user_anthropic_model() -> None:
    # A user-configured Anthropic correction model is preserved.
    models = _all_ollama()
    models["stage_01_correct"] = {"provider": "anthropic", "model": "sonnet"}
    effective, _ = apply_selection(models, _PROVIDERS, _STAGES, provider="ollama")
    assert effective["stage_01_correct"] == {"provider": "anthropic", "model": "sonnet"}


def test_provider_anthropic_overrides_all_stages() -> None:
    effective, warnings = apply_selection(_all_ollama(), _PROVIDERS, _STAGES, provider="anthropic")
    assert all(effective[s] == {"provider": "anthropic", "model": "sonnet"} for s in _STAGES)
    assert warnings == []


def test_provider_ollama_uses_config_default_model_and_warns() -> None:
    effective, warnings = apply_selection(_all_ollama(), _PROVIDERS, _STAGES, provider="ollama")
    assert all(effective[s] == {"provider": "ollama", "model": "qwen3:8b"} for s in _STAGES)
    assert warnings  # open provider selected for heavy stages, no --hybrid


def test_provider_ollama_plus_hybrid_keeps_heavy_on_anthropic_no_warning() -> None:
    effective, warnings = apply_selection(
        _all_ollama(), _PROVIDERS, _STAGES, provider="ollama", hybrid=True
    )
    for s in HEAVY_STAGES:
        assert effective[s] == {"provider": "anthropic", "model": "sonnet"}
    assert effective["router"] == {"provider": "ollama", "model": "qwen3:8b"}
    assert warnings == []


def test_hybrid_alone_lifts_heavy_no_warning() -> None:
    effective, warnings = apply_selection(_all_ollama(), _PROVIDERS, _STAGES, hybrid=True)
    for s in HEAVY_STAGES:
        assert effective[s]["provider"] == "anthropic"
    assert warnings == []


def test_provider_default_model_falls_back_when_absent() -> None:
    providers = {"anthropic": {"api_key": "x"}}  # no default_model
    effective, _ = apply_selection(_all_ollama(), providers, _STAGES, provider="anthropic")
    assert effective["leader"] == {"provider": "anthropic", "model": "sonnet"}


def test_lmstudio_open_provider_also_warns() -> None:
    _e, warnings = apply_selection(_all_ollama(), _PROVIDERS, _STAGES, provider="lmstudio")
    assert warnings


def test_result_is_a_copy() -> None:
    models = _all_ollama()
    effective, _ = apply_selection(models, _PROVIDERS, _STAGES, provider="anthropic")
    effective["leader"]["model"] = "MUTATED"
    assert models["leader"]["model"] == "qwen3:8b"  # input untouched


def _resolved_model_sent_to_provider(monkeypatch, effective: dict, role: str) -> str:
    """Mirror the CLI: configure registry from effective, resolve the stage
    model NAME, and capture what the provider actually receives."""
    captured: dict[str, object] = {}

    class _FakeProvider:
        def invoke(
            self, prompt: str, *, system_prompt: object = None, model: str = "default", **_kw
        ):
            captured["model"] = model
            return LLMResponse(text="ok", model=str(model))

    registry_mod.configure_providers(_PROVIDERS, effective)
    provider_name = registry_mod.resolve_role(role)[0]
    monkeypatch.setitem(registry_mod._provider_cache, provider_name, _FakeProvider())
    # CLI builds the stage model map this way:
    stage_model = registry_mod.resolve_role(role)[1]
    registry_mod.invoke_llm(prompt="hi", model=stage_model, role=role)
    return str(captured["model"])


def test_object_config_sends_model_NAME_not_dict(monkeypatch) -> None:
    # The pre-existing bug: object-style config used to forward the whole
    # {provider, model} dict as the model. Resolving via resolve_role yields
    # the model NAME string instead.
    effective, _ = apply_selection(_all_ollama(), _PROVIDERS, _STAGES)
    assert _resolved_model_sent_to_provider(monkeypatch, effective, "stage_02") == "qwen3:8b"


def test_provider_override_sends_selected_model_name(monkeypatch) -> None:
    effective, _ = apply_selection(_all_ollama(), _PROVIDERS, _STAGES, provider="anthropic")
    assert _resolved_model_sent_to_provider(monkeypatch, effective, "stage_02") == "sonnet"


def test_string_fallback_is_preserved_not_dropped_to_default() -> None:
    # cfg.models supplies per-stage fallbacks as strings (router→"haiku",
    # unspecified stages→the --model value). apply_selection must keep them so
    # missing roles still honor --model instead of the registry default.
    models = _all_ollama()
    models["router"] = "haiku"  # _load_config's router fallback (string form)
    effective, _ = apply_selection(models, _PROVIDERS, _STAGES)
    assert effective["router"] == "haiku"
    registry_mod.configure_providers(_PROVIDERS, effective)
    # resolve_role treats a known Anthropic alias string as anthropic.
    assert registry_mod.resolve_role("router") == ("anthropic", "haiku")
