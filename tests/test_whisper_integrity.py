"""L2 — Whisper cache model integrity tests.

`verify_whisper_model_integrity` is the last line of defense when a
cached `.pt` file gets tampered with between the initial whisper
download (which does verify SHA256) and a subsequent load (which does
not). Tests cover:

  - Matching hash → silent pass
  - Mismatching hash → TranscriptNotAvailable with redacted prefix
  - Missing cache file → silent skip (whisper's own download path)
  - Unknown model name → silent skip
  - Whisper not installed → silent skip
  - Malformed `_MODELS` URL → silent skip
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("whisper", reason="whisper optional extra not installed")

from pipeline_youtube.transcript.base import TranscriptNotAvailable  # noqa: E402
from pipeline_youtube.transcript.whisper_fallback import (  # noqa: E402
    _expected_sha256_for_model,
    _sha256_of_file,
    verify_whisper_model_integrity,
)


def _write_cached_model(cache_root: Path, name: str, contents: bytes) -> Path:
    """Create a fake whisper cache directory with a model file."""
    whisper_dir = cache_root / "whisper"
    whisper_dir.mkdir(parents=True, exist_ok=True)
    path = whisper_dir / f"{name}.pt"
    path.write_bytes(contents)
    return path


def _fake_models_dict(name: str, sha: str) -> dict[str, str]:
    return {name: (f"https://openaipublic.azureedge.net/main/whisper/models/{sha}/{name}.pt")}


class TestExpectedSha256:
    def test_extracts_hash_from_real_whisper_models_dict(self):
        """Smoke test against the actual installed whisper._MODELS."""
        import whisper  # type: ignore[import-untyped]

        name = next(iter(whisper._MODELS))  # any key
        sha = _expected_sha256_for_model(name)
        assert sha is not None
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)

    def test_returns_none_for_unknown_model(self):
        assert _expected_sha256_for_model("nonexistent-model-xyz") is None

    def test_returns_none_for_malformed_url(self, monkeypatch):
        import whisper

        monkeypatch.setattr(whisper, "_MODELS", {"weird": "not-a-valid-url"}, raising=False)
        assert _expected_sha256_for_model("weird") is None


class TestSha256OfFile:
    def test_computes_known_hash(self, tmp_path: Path):
        data = b"hello whisper"
        expected = hashlib.sha256(data).hexdigest()
        target = tmp_path / "x.pt"
        target.write_bytes(data)
        assert _sha256_of_file(target) == expected

    def test_handles_large_file_streaming(self, tmp_path: Path):
        # 3 MiB so the 1 MiB chunk loop runs more than once
        data = b"A" * (3 * 1024 * 1024)
        expected = hashlib.sha256(data).hexdigest()
        target = tmp_path / "big.pt"
        target.write_bytes(data)
        assert _sha256_of_file(target) == expected


class TestVerifyIntegrity:
    def test_matching_hash_passes_silently(self, tmp_path: Path, monkeypatch):
        data = b"genuine weights"
        sha = hashlib.sha256(data).hexdigest()
        _write_cached_model(tmp_path, "tiny", data)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

        import whisper

        monkeypatch.setattr(whisper, "_MODELS", _fake_models_dict("tiny", sha), raising=False)

        # Does not raise.
        verify_whisper_model_integrity("tiny")

    def test_mismatching_hash_raises(self, tmp_path: Path, monkeypatch):
        data = b"tampered weights"
        wrong_sha = "0" * 64
        _write_cached_model(tmp_path, "tiny", data)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

        import whisper

        monkeypatch.setattr(whisper, "_MODELS", _fake_models_dict("tiny", wrong_sha), raising=False)

        with pytest.raises(TranscriptNotAvailable) as exc_info:
            verify_whisper_model_integrity("tiny")
        msg = str(exc_info.value)
        assert "whisper_model_integrity_mismatch" in msg
        # Hash prefixes should be redacted (only 12 chars shown).
        assert "expected=000000000000..." in msg
        actual_prefix = hashlib.sha256(data).hexdigest()[:12]
        assert f"actual={actual_prefix}..." in msg

    def test_missing_cache_file_skips(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        import whisper

        monkeypatch.setattr(whisper, "_MODELS", _fake_models_dict("tiny", "a" * 64), raising=False)
        # No file at tmp_path/whisper/tiny.pt → skip, don't raise.
        verify_whisper_model_integrity("tiny")

    def test_unknown_model_skips(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        # whisper._MODELS has no "xyz" entry — expected hash is None → skip.
        verify_whisper_model_integrity("xyz-not-a-whisper-model")

    def test_whisper_not_installed_skips(self, monkeypatch):
        """When whisper import fails, integrity check no-ops (defense-in-depth,
        not a hard requirement — whisper's absence is handled upstream)."""
        # Make `import whisper` inside `_expected_sha256_for_model` raise.
        with patch.dict("sys.modules", {"whisper": None}):
            verify_whisper_model_integrity("tiny")

    def test_respects_home_cache_fallback(self, tmp_path: Path, monkeypatch):
        """Without XDG_CACHE_HOME, falls back to ~/.cache/whisper."""
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        data = b"home-cache weights"
        sha = hashlib.sha256(data).hexdigest()
        cache_dir = tmp_path / ".cache" / "whisper"
        cache_dir.mkdir(parents=True)
        (cache_dir / "tiny.pt").write_bytes(data)

        import whisper

        monkeypatch.setattr(whisper, "_MODELS", _fake_models_dict("tiny", sha), raising=False)
        verify_whisper_model_integrity("tiny")  # no raise
