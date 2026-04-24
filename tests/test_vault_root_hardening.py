"""Tests for #3: vault_root hardening (reject home/root, require .obsidian)."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import pytest

from pipeline_youtube.config import VaultRootError, reset_vault_root, set_vault_root


class TestVaultRootStrict:
    def teardown_method(self):
        reset_vault_root()

    def test_rejects_home_directory(self):
        with pytest.raises(VaultRootError, match="home"):
            set_vault_root(os.path.expanduser("~"), strict=True)

    def test_rejects_filesystem_root(self):
        with pytest.raises(VaultRootError, match="root"):
            set_vault_root("/", strict=True)

    def test_warns_without_obsidian_dir(self, tmp_path: Path):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            set_vault_root(tmp_path, strict=True)
            assert any(".obsidian" in str(w.message) for w in caught)

    def test_accepts_valid_vault(self, tmp_path: Path):
        (tmp_path / ".obsidian").mkdir()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            set_vault_root(tmp_path, strict=True)
            assert not any(".obsidian" in str(w.message) for w in caught)


class TestVaultRootPermissive:
    def teardown_method(self):
        reset_vault_root()

    def test_default_permissive(self, tmp_path: Path):
        # Legacy tests rely on this path not raising
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # promote warnings to errors
            set_vault_root(tmp_path)

    def test_resolves_symlinks(self, tmp_path: Path):
        target = tmp_path / "real"
        target.mkdir()
        link = tmp_path / "linked"
        link.symlink_to(target)
        set_vault_root(link)
        from pipeline_youtube.config import get_vault_root

        assert get_vault_root() == target.resolve()
