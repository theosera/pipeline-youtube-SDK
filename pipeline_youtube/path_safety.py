"""Path traversal defense, ported from pipeline/storage.ts:24-97.

7-phase defense:
  0. URL decode (catches %2e%2e encoded traversal)
  1. Absolute-path rejection (/, \\, ~, drive letters)
  2. Control-char / null-byte removal
  3. Unicode NFC normalization (macOS HFS+ NFD vs NFC divergence)
  4. Path-segment validation (reject if any segment is '..')
  5. resolve() prefix check against vault_root
  6. realpath() check for symlink-based escape
  7. Path length cap (500 chars)

Any violation returns FALLBACK_PATH. This mirrors the TypeScript
implementation's security guarantees byte-for-byte.
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from urllib.parse import unquote

from .config import get_vault_root, is_dry_run

FALLBACK_PATH = "Clippings/Inbox"
MAX_PATH_LENGTH = 500

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")
_ABSOLUTE_PATH_RE = re.compile(r"^[/\\~]|^[A-Za-z]:")
_PATH_SPLIT_RE = re.compile(r"[/\\]")


def ensure_safe_path(proposed: str | None) -> str:
    """Validate and sanitize a vault-relative path; return FALLBACK_PATH on violation."""
    if not proposed or not isinstance(proposed, str):
        return FALLBACK_PATH

    vault_root = get_vault_root()

    # Phase 0: URL decode (catch %2e%2e encoded traversal)
    try:
        decoded = unquote(proposed)
    except Exception:
        decoded = proposed

    # Phase 1: reject absolute paths (/, \, ~, drive letter)
    if _ABSOLUTE_PATH_RE.match(decoded):
        return FALLBACK_PATH

    # Phase 2: remove control chars and null bytes
    no_control = _CONTROL_CHARS_RE.sub("", decoded)

    # Phase 3: NFC normalization (unify NFC/NFD on macOS HFS+)
    normalized = unicodedata.normalize("NFC", no_control)

    # Phase 4: split segments, reject any '..'
    segments = _PATH_SPLIT_RE.split(normalized)
    if any(seg == ".." for seg in segments):
        return FALLBACK_PATH

    # Filter '.' and empty segments, rejoin with os.sep
    filtered = [seg for seg in segments if seg not in (".", "")]
    if not filtered:
        return FALLBACK_PATH
    sanitized = os.sep.join(filtered)

    # Phase 7 (early length check, before resolve to avoid OS-level issues)
    if len(sanitized) > MAX_PATH_LENGTH:
        return FALLBACK_PATH

    # Phase 5: resolve() + prefix check against vault_root
    try:
        resolved = (vault_root / sanitized).resolve(strict=False)
        resolved.relative_to(vault_root)
    except (ValueError, OSError):
        return FALLBACK_PATH

    # Phase 6: realpath check if the path already exists (symlink defense)
    if resolved.exists():
        try:
            real = Path(os.path.realpath(resolved))
            real_vault = Path(os.path.realpath(vault_root))
            real.relative_to(real_vault)
        except ValueError:
            return FALLBACK_PATH
        except OSError:
            # realpath failed; Phase 5 prefix check is sufficient
            pass

    return sanitized


def safe_rename(src: str | Path, dest: str | Path, dry_run: bool | None = None) -> None:
    """Move src -> dest. In dry-run mode, logs only without touching the filesystem.

    `dry_run` explicit arg overrides the module-level `is_dry_run()` flag.
    """
    effective_dry = is_dry_run() if dry_run is None else dry_run
    if effective_dry:
        print(f"  [DRY-RUN] {src} -> {dest}")
        return
    os.rename(src, dest)
