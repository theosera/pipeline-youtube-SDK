"""Tests for the --local-media offline input builder (local_media.py)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pipeline_youtube.local_media import (
    build_local_videos,
    extract_video_id,
    synthesize_video_id,
    title_from_filename,
)

_CANONICAL_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


class TestExtractVideoId:
    def test_bare_id_stem(self) -> None:
        assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_bracketed_id(self) -> None:
        assert extract_video_id("My Great Talk [dQw4w9WgXcQ]") == "dQw4w9WgXcQ"

    def test_uses_trailing_bracketed_id_when_title_contains_id_shaped_token(self) -> None:
        # yt-dlp always appends the real id last; an id-shaped token earlier in
        # the title must not win.
        stem = "01 [abcdefghijk] Talk [dQw4w9WgXcQ]"
        assert extract_video_id(stem) == "dQw4w9WgXcQ"

    def test_non_trailing_bracketed_token_is_not_trusted_as_id(self) -> None:
        assert extract_video_id("01 [abcdefghijk] Talk") is None

    def test_no_id_returns_none(self) -> None:
        assert extract_video_id("just a plain title") is None

    def test_short_token_not_matched(self) -> None:
        # 10 chars is not a YouTube id.
        assert extract_video_id("abc1234567") is None


class TestSynthesizeVideoId:
    def test_canonical_format(self) -> None:
        vid = synthesize_video_id("some video.mp4")
        assert _CANONICAL_ID.match(vid)

    def test_deterministic(self) -> None:
        assert synthesize_video_id("a.mp4") == synthesize_video_id("a.mp4")

    def test_distinct_inputs_distinct_ids(self) -> None:
        assert synthesize_video_id("a.mp4") != synthesize_video_id("b.mp4")


class TestTitleFromFilename:
    def test_strips_bracketed_id(self) -> None:
        assert title_from_filename("My Talk [dQw4w9WgXcQ]", "dQw4w9WgXcQ") == "My Talk"

    def test_bare_id_falls_back(self) -> None:
        assert title_from_filename("dQw4w9WgXcQ", "dQw4w9WgXcQ") == "dQw4w9WgXcQ"


class TestBuildLocalVideos:
    def _touch(self, d: Path, name: str) -> None:
        (d / name).write_bytes(b"\x00")

    def test_builds_videos_and_map(self, tmp_path: Path) -> None:
        media = tmp_path / "My Playlist"
        media.mkdir()
        self._touch(media, "01 Intro [dQw4w9WgXcQ].mp4")
        self._touch(media, "02 Deep Dive [9bZkp7q19f0].mkv")
        self._touch(media, "notes.txt")  # ignored (not media)

        videos, media_map = build_local_videos(media)

        assert [v.video_id for v in videos] == ["dQw4w9WgXcQ", "9bZkp7q19f0"]
        assert videos[0].title == "01 Intro"
        assert all(v.playlist_title == "My Playlist" for v in videos)
        assert set(media_map) == {"dQw4w9WgXcQ", "9bZkp7q19f0"}
        assert media_map["dQw4w9WgXcQ"].suffix == ".mp4"

    def test_sorted_by_filename(self, tmp_path: Path) -> None:
        media = tmp_path / "pl"
        media.mkdir()
        self._touch(media, "b.mp4")
        self._touch(media, "a.mp4")
        videos, _ = build_local_videos(media)
        # 'a.mp4' sorts before 'b.mp4'; ids are synthesized but order is by name.
        assert media_titles(videos) == ["a", "b"]

    def test_synthesized_id_for_plain_name(self, tmp_path: Path) -> None:
        media = tmp_path / "pl"
        media.mkdir()
        self._touch(media, "plain title.mp4")
        videos, _ = build_local_videos(media)
        assert len(videos) == 1
        assert _CANONICAL_ID.match(videos[0].video_id)

    def test_synthesized_ids_disambiguated_by_folder(self, tmp_path: Path) -> None:
        # Same filename under two differently-named folders must NOT collide,
        # or their transcript cache / checkpoints would be cross-wired.
        dir_a = tmp_path / "course-a"
        dir_b = tmp_path / "course-b"
        dir_a.mkdir()
        dir_b.mkdir()
        self._touch(dir_a, "lesson.mp4")
        self._touch(dir_b, "lesson.mp4")

        (videos_a, _), (videos_b, _) = build_local_videos(dir_a), build_local_videos(dir_b)
        assert videos_a[0].video_id != videos_b[0].video_id

    def test_synthesized_id_stable_across_runs(self, tmp_path: Path) -> None:
        media = tmp_path / "pl"
        media.mkdir()
        self._touch(media, "plain title.mp4")
        first, _ = build_local_videos(media)
        second, _ = build_local_videos(media)
        assert first[0].video_id == second[0].video_id

    def test_empty_dir(self, tmp_path: Path) -> None:
        media = tmp_path / "empty"
        media.mkdir()
        videos, media_map = build_local_videos(media)
        assert videos == []
        assert media_map == {}

    def test_not_a_directory_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "file.mp4"
        f.write_bytes(b"x")
        with pytest.raises(ValueError, match="not a directory"):
            build_local_videos(f)


def media_titles(videos: list) -> list[str]:  # type: ignore[type-arg]
    return [v.title for v in videos]
