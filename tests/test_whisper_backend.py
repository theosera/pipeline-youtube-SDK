"""Tests for the Whisper backend/model selection (MLX vs openai-whisper)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import pipeline_youtube.transcript.whisper_fallback as wf


@pytest.fixture(autouse=True)
def _reset_whisper_config() -> Iterator[None]:
    saved = (wf._BACKEND, wf._MODEL)
    yield
    wf._BACKEND, wf._MODEL = saved


class TestConfigureWhisper:
    def test_sets_backend_and_model(self) -> None:
        wf.configure_whisper(backend="openai", model="medium")
        assert wf._BACKEND == "openai"
        assert wf._MODEL == "medium"

    def test_empty_model_becomes_none(self) -> None:
        wf.configure_whisper(backend="auto", model="")
        assert wf._MODEL is None

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="whisper_backend"):
            wf.configure_whisper(backend="bogus")


class TestResolveBackend:
    def test_explicit_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(wf, "_mlx_available", lambda: True)  # ignored when explicit
        wf.configure_whisper(backend="openai")
        assert wf._resolve_backend() == "openai"

    def test_explicit_mlx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(wf, "_mlx_available", lambda: False)  # ignored when explicit
        wf.configure_whisper(backend="mlx")
        assert wf._resolve_backend() == "mlx"

    def test_auto_picks_mlx_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(wf, "_mlx_available", lambda: True)
        wf.configure_whisper(backend="auto")
        assert wf._resolve_backend() == "mlx"

    def test_auto_falls_back_to_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(wf, "_mlx_available", lambda: False)
        wf.configure_whisper(backend="auto")
        assert wf._resolve_backend() == "openai"


class TestResolveModel:
    def test_openai_default(self) -> None:
        wf.configure_whisper(backend="openai", model=None)
        assert wf._resolve_openai_model() == wf.DEFAULT_WHISPER_MODEL

    def test_openai_explicit(self) -> None:
        wf.configure_whisper(backend="openai", model="medium")
        assert wf._resolve_openai_model() == "medium"

    def test_mlx_default_is_turbo(self) -> None:
        wf.configure_whisper(backend="mlx", model=None)
        assert wf._resolve_mlx_repo() == "mlx-community/whisper-large-v3-turbo"

    def test_mlx_logical_name_maps_to_repo(self) -> None:
        wf.configure_whisper(backend="mlx", model="medium")
        assert wf._resolve_mlx_repo() == "mlx-community/whisper-medium"

    def test_mlx_unknown_name_passes_through(self) -> None:
        # A full HF repo id should be usable verbatim.
        wf.configure_whisper(backend="mlx", model="mlx-community/whisper-custom")
        assert wf._resolve_mlx_repo() == "mlx-community/whisper-custom"


class TestMlxAvailability:
    def test_false_off_apple_silicon(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(wf.platform, "system", lambda: "Linux")
        monkeypatch.setattr(wf.platform, "machine", lambda: "x86_64")
        assert wf._mlx_available() is False
