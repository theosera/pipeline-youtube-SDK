"""Module-level configuration state.

Mirrors the `pipeline/config.ts` pattern: simple setters/getters so
tests can swap vault_root without a full config file. JSON config
loading is layered on top in later steps.

SDK version adds provider-level configuration for multi-LLM support.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

from .domain.errors import VaultRootError as VaultRootError

_vault_root: Path | None = None
_dry_run: bool = False


def set_vault_root(path: str | Path, *, strict: bool = False) -> None:
    """Set the vault root after symlink resolution + safety checks.

    `strict=True` (production path — `main.cli()`):
      - Rejects the user's home directory itself (too broad).
      - Rejects the filesystem root.
      - Warns if `.obsidian/` is missing (likely a misconfiguration).

    `strict=False` (tests, library callers): only `expanduser` +
    `resolve` so symlinks/realpath are still normalized. This preserves
    the legacy behavior for callers that assign a fresh `tmp_path`.
    """
    global _vault_root
    resolved = Path(path).expanduser().resolve()

    if strict:
        if resolved == Path(os.path.expanduser("~")).resolve():
            raise VaultRootError(f"vault_root may not be the user's home directory: {resolved}")
        if str(resolved) == resolved.root or str(resolved) in ("/", "C:\\"):
            raise VaultRootError(f"vault_root may not be the filesystem root: {resolved}")
        if not (resolved / ".obsidian").is_dir():
            warnings.warn(
                f"vault_root {resolved!s} does not contain `.obsidian/` "
                "(not recognized as an Obsidian vault)",
                stacklevel=2,
            )

    _vault_root = resolved


def get_vault_root() -> Path:
    if _vault_root is None:
        raise RuntimeError("vault_root is not set. Call set_vault_root() before using path_safety.")
    return _vault_root


def reset_vault_root() -> None:
    global _vault_root
    _vault_root = None


def set_dry_run(flag: bool) -> None:
    global _dry_run
    _dry_run = flag


def is_dry_run() -> bool:
    return _dry_run
