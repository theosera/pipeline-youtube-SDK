"""Tests for WS2: per-stage / per-agent model cascade loaded from config.json."""

from __future__ import annotations

import json
from pathlib import Path

import click
import pytest

from pipeline_youtube.main import _load_config


def _write_config(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestLoadConfig:
    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(click.UsageError, match="config.json not found"):
            _load_config(tmp_path / "does-not-exist.json", fallback_model="sonnet")

    def test_placeholder_vault_rejected(self, tmp_path: Path):
        cfg = _write_config(
            tmp_path / "config.json",
            {"vault_root": "/path/to/your/Obsidian Vault"},
        )
        with pytest.raises(click.UsageError, match="vault_root"):
            _load_config(cfg, fallback_model="sonnet")

    def test_models_omitted_uses_fallback(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        cfg = _write_config(tmp_path / "config.json", {"vault_root": str(vault)})
        result = _load_config(cfg, fallback_model="sonnet")
        assert result.vault_root == vault
        assert result.models == {
            "router": "haiku",  # router defaults to haiku regardless of fallback
            "stage_02": "sonnet",
            "stage_04": "sonnet",
            "alpha": "sonnet",
            "beta": "sonnet",
            "leader": "sonnet",
            "reviewer": "sonnet",
        }

    def test_partial_models_filled_with_fallback(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        cfg = _write_config(
            tmp_path / "config.json",
            {
                "vault_root": str(vault),
                "models": {"alpha": "haiku", "leader": "opus"},
            },
        )
        result = _load_config(cfg, fallback_model="sonnet")
        assert result.models["alpha"] == "haiku"
        assert result.models["leader"] == "opus"
        assert result.models["beta"] == "sonnet"
        assert result.models["stage_02"] == "sonnet"
        assert result.models["stage_04"] == "sonnet"

    def test_all_models_explicit(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        cfg = _write_config(
            tmp_path / "config.json",
            {
                "vault_root": str(vault),
                "models": {
                    "stage_02": "haiku",
                    "stage_04": "sonnet",
                    "alpha": "haiku",
                    "beta": "sonnet",
                    "leader": "opus",
                },
            },
        )
        result = _load_config(cfg, fallback_model="sonnet")
        assert result.models["stage_02"] == "haiku"
        assert result.models["leader"] == "opus"
        assert result.models["alpha"] == "haiku"

    def test_deprecated_gamma_key_accepted_silently(self, tmp_path: Path):
        """Existing config.json with `gamma` key must not break after γ removal."""
        vault = tmp_path / "vault"
        vault.mkdir()
        cfg = _write_config(
            tmp_path / "config.json",
            {
                "vault_root": str(vault),
                "models": {"gamma": "haiku", "alpha": "sonnet"},
            },
        )
        result = _load_config(cfg, fallback_model="sonnet")
        assert "gamma" not in result.models
        assert result.models["alpha"] == "sonnet"

    def test_unknown_model_key_rejected(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        cfg = _write_config(
            tmp_path / "config.json",
            {
                "vault_root": str(vault),
                "models": {"delta": "haiku"},
            },
        )
        with pytest.raises(click.UsageError, match="unknown model keys"):
            _load_config(cfg, fallback_model="sonnet")

    def test_capture_backend_default_host(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        cfg = _write_config(tmp_path / "config.json", {"vault_root": str(vault)})
        result = _load_config(cfg, fallback_model="sonnet")
        assert result.capture_backend == "host"
        assert result.capture_docker_image == "pipeline-youtube-capture:latest"

    def test_capture_backend_docker_accepted(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        cfg = _write_config(
            tmp_path / "config.json",
            {
                "vault_root": str(vault),
                "capture_backend": "docker",
                "capture_docker_image": "custom-image:v2",
            },
        )
        result = _load_config(cfg, fallback_model="sonnet")
        assert result.capture_backend == "docker"
        assert result.capture_docker_image == "custom-image:v2"

    def test_invalid_capture_backend_rejected(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        cfg = _write_config(
            tmp_path / "config.json",
            {"vault_root": str(vault), "capture_backend": "kubernetes"},
        )
        with pytest.raises(click.UsageError, match="capture_backend must be one of"):
            _load_config(cfg, fallback_model="sonnet")

    def test_models_must_be_object(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        cfg = _write_config(
            tmp_path / "config.json",
            {"vault_root": str(vault), "models": "sonnet"},
        )
        with pytest.raises(click.UsageError, match="'models' must be an object"):
            _load_config(cfg, fallback_model="sonnet")
