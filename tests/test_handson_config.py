"""Verifies the hands-on config surface: resolve_config_path precedence,
handson model keys in _load_config, LLM cache role policy, and --hybrid
pinning of the hands-on heavy roles.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pytest

from pipeline_youtube.cli_config import (
    _MODEL_KEYS,
    DEFAULT_CONFIG_PATH,
    DEFAULT_HANDSON_CONFIG_PATH,
    _load_config,
    resolve_config_path,
)
from pipeline_youtube.providers import registry as registry_mod
from pipeline_youtube.providers.registry import configure_llm_cache
from pipeline_youtube.providers.selection import HEAVY_STAGES, apply_selection


class TestResolveConfigPath:
    def test_explicit_config_wins_in_both_modes(self, tmp_path: Path):
        explicit = tmp_path / "my.json"
        assert resolve_config_path(explicit, handson=False) == explicit
        assert resolve_config_path(explicit, handson=True) == explicit

    def test_handson_defaults_to_handson_config(self):
        assert resolve_config_path(None, handson=True) == DEFAULT_HANDSON_CONFIG_PATH
        assert DEFAULT_HANDSON_CONFIG_PATH.name == "config.handson.json"

    def test_normal_mode_defaults_to_config_json(self):
        assert resolve_config_path(None, handson=False) == DEFAULT_CONFIG_PATH


class TestHandsonModelKeys:
    def test_handson_keys_are_registered(self):
        assert {"handson_segment", "handson_plan", "handson_step", "handson_moc"} <= _MODEL_KEYS

    def test_config_with_handson_keys_loads(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        cfg = tmp_path / "config.handson.json"
        cfg.write_text(
            json.dumps(
                {
                    "vault_root": str(vault),
                    "models": {"handson_step": "opus", "handson_moc": "opus"},
                }
            ),
            encoding="utf-8",
        )
        result = _load_config(cfg, fallback_model="sonnet")
        assert result.models["handson_step"] == "opus"
        assert result.models["handson_segment"] == "sonnet"  # fallback

    def test_unknown_key_still_rejected(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        cfg = tmp_path / "config.json"
        cfg.write_text(
            json.dumps({"vault_root": str(vault), "models": {"handson_typo": "opus"}}),
            encoding="utf-8",
        )
        with pytest.raises(click.UsageError, match="unknown model keys"):
            _load_config(cfg, fallback_model="sonnet")


@pytest.fixture
def _restore_llm_cache_policy():
    yield
    configure_llm_cache(stages=True, synthesis=False)  # module default


@pytest.mark.usefixtures("_restore_llm_cache_policy")
class TestLlmCachePolicy:
    def test_segment_and_plan_follow_stage_policy(self):
        configure_llm_cache(stages=True, synthesis=False)
        assert registry_mod._llm_cache_enabled_for_role("handson_segment") is True
        assert registry_mod._llm_cache_enabled_for_role("handson_plan") is True
        configure_llm_cache(stages=False, synthesis=False)
        assert registry_mod._llm_cache_enabled_for_role("handson_segment") is False

    def test_step_and_moc_follow_synthesis_policy(self):
        configure_llm_cache(stages=True, synthesis=False)
        assert registry_mod._llm_cache_enabled_for_role("handson_step") is False
        assert registry_mod._llm_cache_enabled_for_role("handson_moc") is False
        configure_llm_cache(stages=True, synthesis=True)
        assert registry_mod._llm_cache_enabled_for_role("handson_step") is True


class TestHybridSelection:
    def test_heavy_stages_include_handson_generation_roles(self):
        assert "handson_step" in HEAVY_STAGES
        assert "handson_moc" in HEAVY_STAGES

    def test_hybrid_pins_handson_heavy_roles_to_anthropic(self):
        models_cfg = {key: {"provider": "ollama", "model": "qwen3:8b"} for key in _MODEL_KEYS}
        providers_cfg = {
            "ollama": {"base_url": "http://localhost:11434/v1"},
            "anthropic": {"default_model": "opus"},
        }
        effective, _ = apply_selection(
            models_cfg, providers_cfg, _MODEL_KEYS, provider="ollama", hybrid=True
        )
        assert effective["handson_step"] == {"provider": "anthropic", "model": "opus"}
        assert effective["handson_moc"] == {"provider": "anthropic", "model": "opus"}
        # The classifier role stays on the selected open provider.
        assert effective["handson_segment"]["provider"] == "ollama"
