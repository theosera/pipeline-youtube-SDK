"""Tests for checkpoint.py — video completion detection."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from pipeline_youtube.checkpoint import (
    get_completed_video_ids,
    is_video_complete,
)
from pipeline_youtube.config import reset_vault_root, set_vault_root


@pytest.fixture()
def vault(tmp_path: Path):
    """Set up a vault with a 04_Learning_Material playlist folder."""
    set_vault_root(tmp_path)
    yield tmp_path
    reset_vault_root()


def _create_04_md(
    vault: Path,
    folder_name: str,
    video_id: str,
    title: str = "test",
    *,
    include_url: bool = True,
) -> Path:
    """Create a minimal 04 md with the real pipeline frontmatter shape.

    `include_url=True` mirrors the production frontmatter which always
    contains `URL: "https://www.youtube.com/watch?v=<id>"` (used by the
    M3 integrity cross-check).
    """
    folder = vault / "Permanent Note" / "08_YouTube学習" / "04_Learning_Material" / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    md = folder / f"2026-04-16-0914 {title}.md"
    lines = [
        "---",
        "date: 2026-04-16 09:14",
        f'title: "{title}"',
    ]
    if include_url:
        lines.append(f'URL: "https://www.youtube.com/watch?v={video_id}"')
    lines.extend(
        [
            f'video_id: "{video_id}"',
            "tags: [memo, youtube]",
            "---",
            "",
            "Body.",
            "",
        ]
    )
    md.write_text("\n".join(lines), encoding="utf-8")
    return md


# 11-char YouTube-shaped video IDs for fixture consistency.
_VID_A = "abc123DEFGH"
_VID_B = "xyz789IJKLM"
_VID_C = "vid000XYZ_1"


class TestIsVideoComplete:
    def test_no_folder_returns_false(self, vault):
        dt = datetime(2026, 4, 16, 9, 14)
        assert is_video_complete(_VID_A, "AI駆動経営", dt) is False

    def test_existing_video_returns_true(self, vault):
        dt = datetime(2026, 4, 16, 9, 14)
        _create_04_md(vault, "2026-04-16-0914 AI駆動経営", _VID_A, "テスト動画")
        assert is_video_complete(_VID_A, "AI駆動経営", dt) is True

    def test_different_video_id_returns_false(self, vault):
        dt = datetime(2026, 4, 16, 9, 14)
        _create_04_md(vault, "2026-04-16-0914 AI駆動経営", _VID_A)
        assert is_video_complete(_VID_B, "AI駆動経営", dt) is False

    def test_legacy_folder_name_fallback(self, vault):
        """Should find videos in legacy folder (no HHmm, old title format)."""
        dt = datetime(2026, 4, 16, 9, 14)
        _create_04_md(vault, "2026-04-16 AI駆動経営", _VID_A)
        assert is_video_complete(_VID_A, "AI駆動経営", dt) is True

    def test_slash_playlist_title(self, vault):
        """Playlist title with `/` — display title is last segment."""
        dt = datetime(2026, 4, 16, 9, 14)
        _create_04_md(vault, "2026-04-16-0914 AI駆動経営", _VID_A)
        assert is_video_complete(_VID_A, "2026Agent Teams/AI駆動経営", dt) is True

    def test_multiple_videos(self, vault):
        dt = datetime(2026, 4, 16, 9, 14)
        folder_name = "2026-04-16-0914 AI駆動経営"
        _create_04_md(vault, folder_name, _VID_A, "動画1")
        _create_04_md(vault, folder_name, _VID_B, "動画2")
        assert is_video_complete(_VID_A, "AI駆動経営", dt) is True
        assert is_video_complete(_VID_B, "AI駆動経営", dt) is True
        assert is_video_complete(_VID_C, "AI駆動経営", dt) is False


class TestGetCompletedVideoIds:
    def test_empty_folder(self, vault):
        dt = datetime(2026, 4, 16, 9, 14)
        assert get_completed_video_ids("AI駆動経営", dt) == set()

    def test_collects_all_ids(self, vault):
        dt = datetime(2026, 4, 16, 9, 14)
        folder_name = "2026-04-16-0914 AI駆動経営"
        _create_04_md(vault, folder_name, _VID_A, "動画1")
        _create_04_md(vault, folder_name, _VID_B, "動画2")
        _create_04_md(vault, folder_name, _VID_C, "動画3")
        ids = get_completed_video_ids("AI駆動経営", dt)
        assert ids == {_VID_A, _VID_B, _VID_C}

    def test_no_folder_returns_empty_set(self, vault):
        dt = datetime(2026, 4, 16, 9, 14)
        assert get_completed_video_ids("nonexistent", dt) == set()
