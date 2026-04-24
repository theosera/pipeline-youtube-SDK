"""Tests for dynamic synthesis timeout computation and preflight estimation."""

from __future__ import annotations

import json
from pathlib import Path

import click
import pytest

from pipeline_youtube.main import _load_config
from pipeline_youtube.stages.synthesis import log_synthesis_preflight
from pipeline_youtube.synthesis.agents import (
    _BETA_TIMEOUT_CAP,
    SYNTHESIS_TIMEOUT_BASE,
    SYNTHESIS_TIMEOUT_PER_VIDEO,
    compute_synthesis_timeouts,
)

# =====================================================
# compute_synthesis_timeouts
# =====================================================


class TestComputeSynthesisTimeouts:
    def test_auto_scales_with_video_count(self):
        t = compute_synthesis_timeouts(10)
        expected_heavy = SYNTHESIS_TIMEOUT_BASE + SYNTHESIS_TIMEOUT_PER_VIDEO * 10
        assert t["alpha"] == expected_heavy
        assert t["leader"] == expected_heavy
        assert t["beta"] == min(expected_heavy, _BETA_TIMEOUT_CAP)

    def test_auto_small_playlist(self):
        t = compute_synthesis_timeouts(3)
        expected = SYNTHESIS_TIMEOUT_BASE + SYNTHESIS_TIMEOUT_PER_VIDEO * 3
        assert t["alpha"] == expected
        assert t["leader"] == expected
        assert t["beta"] == min(expected, _BETA_TIMEOUT_CAP)

    def test_auto_large_playlist(self):
        t = compute_synthesis_timeouts(50)
        expected = SYNTHESIS_TIMEOUT_BASE + SYNTHESIS_TIMEOUT_PER_VIDEO * 50
        assert t["alpha"] == expected
        assert t["leader"] == expected
        assert t["beta"] == _BETA_TIMEOUT_CAP

    def test_override_applied(self):
        t = compute_synthesis_timeouts(10, override=3600)
        assert t["alpha"] == 3600
        assert t["leader"] == 3600
        assert t["beta"] == _BETA_TIMEOUT_CAP

    def test_override_small_caps_beta(self):
        t = compute_synthesis_timeouts(3, override=300)
        assert t["alpha"] == 300
        assert t["leader"] == 300
        assert t["beta"] == 300

    def test_zero_videos(self):
        t = compute_synthesis_timeouts(0)
        assert t["alpha"] == SYNTHESIS_TIMEOUT_BASE
        assert t["leader"] == SYNTHESIS_TIMEOUT_BASE

    def test_default_formula_matches_legacy_24_videos(self):
        """24 videos was the playlist that originally timed out at 1800s.
        The auto formula should produce a comparable timeout."""
        t = compute_synthesis_timeouts(24)
        assert t["alpha"] >= 1740
        assert t["leader"] >= 1740


# =====================================================
# log_synthesis_preflight
# =====================================================


class TestLogSynthesisPreflight:
    def test_no_truncation(self):
        bodies = ["x" * 1000] * 5
        timeouts = {"alpha": 600, "beta": 600, "leader": 600}
        msg = log_synthesis_preflight(5, bodies, timeouts)
        assert "videos: 5" in msg
        assert "α=600s" in msg
        assert "truncation: none" in msg

    def test_truncation_detected(self):
        # 400_000 / 2 = 200_000 per video; make one body exceed that
        bodies = ["x" * 300_000, "y" * 100]
        timeouts = {"alpha": 600, "beta": 600, "leader": 600}
        msg = log_synthesis_preflight(2, bodies, timeouts)
        assert "truncated: 1/2" in msg

    def test_fill_percentage(self):
        bodies = ["x" * 200_000, "y" * 200_000]
        timeouts = {"alpha": 1200, "beta": 600, "leader": 1200}
        msg = log_synthesis_preflight(2, bodies, timeouts)
        assert "100%" in msg


# =====================================================
# config.json synthesis_timeout loading
# =====================================================


def _write_config(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestConfigSynthesisTimeout:
    def test_auto_string(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        cfg = _write_config(
            tmp_path / "config.json",
            {"vault_root": str(vault), "synthesis_timeout": "auto"},
        )
        result = _load_config(cfg, fallback_model="sonnet")
        assert result.synthesis_timeout is None

    def test_omitted_defaults_to_auto(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        cfg = _write_config(
            tmp_path / "config.json",
            {"vault_root": str(vault)},
        )
        result = _load_config(cfg, fallback_model="sonnet")
        assert result.synthesis_timeout is None

    def test_integer_accepted(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        cfg = _write_config(
            tmp_path / "config.json",
            {"vault_root": str(vault), "synthesis_timeout": 3600},
        )
        result = _load_config(cfg, fallback_model="sonnet")
        assert result.synthesis_timeout == 3600

    def test_invalid_string_rejected(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        cfg = _write_config(
            tmp_path / "config.json",
            {"vault_root": str(vault), "synthesis_timeout": "fast"},
        )
        with pytest.raises(click.UsageError, match="synthesis_timeout"):
            _load_config(cfg, fallback_model="sonnet")

    def test_zero_rejected(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        cfg = _write_config(
            tmp_path / "config.json",
            {"vault_root": str(vault), "synthesis_timeout": 0},
        )
        with pytest.raises(click.UsageError, match="synthesis_timeout"):
            _load_config(cfg, fallback_model="sonnet")

    def test_negative_rejected(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        cfg = _write_config(
            tmp_path / "config.json",
            {"vault_root": str(vault), "synthesis_timeout": -100},
        )
        with pytest.raises(click.UsageError, match="synthesis_timeout"):
            _load_config(cfg, fallback_model="sonnet")
