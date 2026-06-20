"""Tests for path_safety, mirroring pipeline/test/security.ts cases.

Every TS test case has a 1:1 Python equivalent below, plus a few
additional edge cases that are Python-specific.
"""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path

import pytest

from pipeline_youtube import config
from pipeline_youtube.path_safety import FALLBACK_PATH, ensure_safe_path, safe_rename


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """Fresh tmp vault_root per test, injected explicitly into ensure_safe_path."""
    config.set_dry_run(False)
    return tmp_path


# =====================================================
# 1. Path traversal defense
# =====================================================


class TestPathTraversal:
    def test_single_dotdot_rejected(self, vault: Path):
        assert ensure_safe_path("../etc/passwd", vault_root=vault) == FALLBACK_PATH

    def test_multiple_dotdot_rejected(self, vault: Path):
        assert ensure_safe_path("../../etc/passwd", vault_root=vault) == FALLBACK_PATH

    def test_middle_dotdot_rejected(self, vault: Path):
        assert ensure_safe_path("foo/../../../etc/passwd", vault_root=vault) == FALLBACK_PATH

    def test_absolute_slash_rejected(self, vault: Path):
        assert ensure_safe_path("/etc/passwd", vault_root=vault) == FALLBACK_PATH

    def test_tilde_rejected(self, vault: Path):
        assert ensure_safe_path("~/secret", vault_root=vault) == FALLBACK_PATH

    def test_windows_drive_letter_rejected(self, vault: Path):
        assert ensure_safe_path("C:\\Windows\\System32", vault_root=vault) == FALLBACK_PATH

    def test_url_encoded_dotdot_rejected(self, vault: Path):
        assert ensure_safe_path("%2e%2e/etc/passwd", vault_root=vault) == FALLBACK_PATH

    def test_url_encoded_slash_and_dotdot_rejected(self, vault: Path):
        assert ensure_safe_path("foo%2f..%2f..%2fetc%2fpasswd", vault_root=vault) == FALLBACK_PATH

    def test_backslash_dotdot_rejected(self, vault: Path):
        assert ensure_safe_path("Engineer\\..\\etc", vault_root=vault) == FALLBACK_PATH


# =====================================================
# 2. Normal paths are preserved
# =====================================================


class TestNormalPaths:
    def test_simple_path_preserved(self, vault: Path):
        assert ensure_safe_path("Engineer/LLM", vault_root=vault) == f"Engineer{os.sep}LLM"

    def test_japanese_path_preserved(self, vault: Path):
        assert (
            ensure_safe_path("Engineer/AGENT経済圏", vault_root=vault)
            == f"Engineer{os.sep}AGENT経済圏"
        )

    def test_excluded_special_value_passes_through(self, vault: Path):
        # classifier uses '__EXCLUDED__' as a sentinel — must not be rewritten
        assert ensure_safe_path("__EXCLUDED__", vault_root=vault) == "__EXCLUDED__"

    def test_dot_segment_filtered(self, vault: Path):
        assert ensure_safe_path("Engineer/./LLM", vault_root=vault) == f"Engineer{os.sep}LLM"

    def test_deep_nested_path_preserved(self, vault: Path):
        assert (
            ensure_safe_path(
                "Permanent Note/08_YouTube学習/01_Scripts_Processing_Unit",
                vault_root=vault,
            )
            == f"Permanent Note{os.sep}08_YouTube学習{os.sep}01_Scripts_Processing_Unit"
        )


# =====================================================
# 3. Sanitization (control chars, length, unicode)
# =====================================================


class TestSanitization:
    def test_empty_fallback(self, vault: Path):
        assert ensure_safe_path("", vault_root=vault) == FALLBACK_PATH

    def test_none_fallback(self, vault: Path):
        assert ensure_safe_path(None, vault_root=vault) == FALLBACK_PATH

    def test_non_string_fallback(self, vault: Path):
        assert ensure_safe_path(123, vault_root=vault) == FALLBACK_PATH  # type: ignore[arg-type]

    def test_null_byte_removed(self, vault: Path):
        result = ensure_safe_path("Engineer/\x00LLM", vault_root=vault)
        assert "\x00" not in result
        assert result == f"Engineer{os.sep}LLM"

    def test_control_char_removed(self, vault: Path):
        result = ensure_safe_path("Engineer/\rLLM", vault_root=vault)
        assert "\r" not in result

    def test_extremely_long_path_rejected(self, vault: Path):
        assert ensure_safe_path("a" * 600, vault_root=vault) == FALLBACK_PATH

    def test_nfc_nfd_unified(self, vault: Path):
        nfc = unicodedata.normalize("NFC", "テスト")
        nfd = unicodedata.normalize("NFD", "テスト")
        assert ensure_safe_path(nfc, vault_root=vault) == ensure_safe_path(nfd, vault_root=vault)


# =====================================================
# 4. Dry-run (safe_rename)
# =====================================================


class TestDryRun:
    def test_dry_run_does_not_move_file(self, tmp_path: Path):
        src = tmp_path / "src.md"
        dest = tmp_path / "dest.md"
        src.write_text("content")

        safe_rename(src, dest, dry_run=True)

        assert src.exists()
        assert not dest.exists()

    def test_real_mode_moves_file(self, tmp_path: Path):
        src = tmp_path / "src.md"
        dest = tmp_path / "dest.md"
        src.write_text("content")

        safe_rename(src, dest, dry_run=False)

        assert not src.exists()
        assert dest.exists()

    def test_module_level_dry_run_flag(self, tmp_path: Path):
        config.set_dry_run(True)
        src = tmp_path / "a.md"
        dest = tmp_path / "b.md"
        src.write_text("x")

        safe_rename(src, dest)  # no explicit arg → uses module flag

        assert src.exists()
        assert not dest.exists()
        config.set_dry_run(False)


# =====================================================
# 5. Vault root binding (resolve / symlink defense)
# =====================================================


class TestVaultBinding:
    def test_symlink_escape_blocked(self, tmp_path: Path):
        # Create a symlink inside the vault pointing outside
        outside = tmp_path.parent / "outside_vault"
        outside.mkdir(exist_ok=True)
        symlink = tmp_path / "escape"
        try:
            symlink.symlink_to(outside)
        except OSError:
            pytest.skip("symlinks not supported on this filesystem")

        # Accessing 'escape/foo' should be caught at Phase 6 if the
        # resolved realpath leaves the vault
        result = ensure_safe_path("escape", vault_root=tmp_path)
        # The symlink itself exists and resolves outside → Phase 6 rejects
        assert result == FALLBACK_PATH


class TestInjectedVaultRoot:
    """DI: an explicitly injected vault_root must be normalized like the global."""

    def test_symlinked_injected_root_is_resolved(self, tmp_path: Path):
        # `link` points at the real vault but is itself unresolved (symlink).
        # Injecting it must still accept a valid relative path — the prefix check
        # would otherwise compare the child's realpath against the symlink name
        # and collapse to FALLBACK_PATH.
        link = tmp_path.parent / (tmp_path.name + "_vault_link")
        try:
            link.symlink_to(tmp_path, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks not supported on this filesystem")
        try:
            out = ensure_safe_path("Foo/bar", vault_root=link)
        finally:
            link.unlink()
        assert out == "Foo/bar"
