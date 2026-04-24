"""Tests for #9: allowlist of `extra` keys in build_frontmatter."""

from __future__ import annotations

from datetime import datetime

import pytest

from pipeline_youtube.obsidian import build_frontmatter


class TestBuildFrontmatterAllowlist:
    def test_all_allowed_keys_accepted(self):
        out = build_frontmatter(
            dt=datetime(2026, 4, 18, 20, 30),
            title="t",
            extra={
                "playlist": "p",
                "video_id": "vid",
                "reviewed": "false",
                "one_liner": "1 行",
                "chapter": "1",
                "category": "core",
                "sources": "a, b",
            },
        )
        assert 'video_id: "vid"' in out

    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError, match="disallowed"):
            build_frontmatter(
                dt=datetime(2026, 4, 18, 20, 30),
                title="t",
                extra={"injected_by_user": "evil"},
            )

    def test_partial_allowed_keys(self):
        out = build_frontmatter(
            dt=datetime(2026, 4, 18, 20, 30),
            title="t",
            extra={"video_id": "abc"},
        )
        assert 'video_id: "abc"' in out

    def test_none_extra_ok(self):
        out = build_frontmatter(
            dt=datetime(2026, 4, 18, 20, 30),
            title="t",
        )
        assert "---" in out
