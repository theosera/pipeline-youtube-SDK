"""Resume-from-explicit-folder flow (--folder-name).

The URL-free reconstruction and explicit-folder body collection are scaffold
stubs; this locks in that they are wired but not yet implemented, plus the
detailed cases that will be filled in.
"""

from __future__ import annotations

import pytest

from pipeline_youtube.main import _videos_from_learning_folder


def test_videos_from_learning_folder_is_scaffolded() -> None:
    # Wired into the CLI (URL-free resume) but logic is still TODO.
    with pytest.raises(NotImplementedError):
        _videos_from_learning_folder("2026-06-03-1200 Example Playlist")


@pytest.mark.skip(reason="scaffold: explicit --folder-name resume logic TODO")
def test_explicit_folder_loads_past_date() -> None:
    """--folder-name resumes a folder whose date != today (no date constraint)."""


@pytest.mark.skip(reason="scaffold: URL-free reconstruction logic TODO")
def test_url_free_reconstruction_from_frontmatter() -> None:
    """Videos are rebuilt from each 04 md's trusted frontmatter (no URL)."""


@pytest.mark.skip(reason="scaffold: explicit --folder-name resume logic TODO")
def test_missing_folder_raises_usage_error() -> None:
    """A non-existent explicit folder name fails with a clear UsageError."""


@pytest.mark.skip(reason="scaffold: resume + eval wiring TODO")
def test_folder_name_with_eval_loop_runs_05_then_eval() -> None:
    """--synthesis-only --folder-name <N> --eval-loop 2 resumes 05 then evaluates."""
