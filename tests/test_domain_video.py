"""Tests for pipeline_youtube.domain.video.VideoMeta."""

from __future__ import annotations

import pytest

from pipeline_youtube.domain.video import VideoMeta


def _make_video(**kwargs) -> VideoMeta:
    """Create a VideoMeta with sensible defaults, overriding with kwargs."""
    defaults = dict(
        video_id="dQw4w9WgXcQ",
        title="Never Gonna Give You Up",
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        duration=213,
        channel="Rick Astley",
        upload_date="19871027",
        playlist_title="80s Classics",
    )
    defaults.update(kwargs)
    return VideoMeta(**defaults)


class TestVideoMetaConstruction:
    def test_all_fields_stored(self):
        v = _make_video()
        assert v.video_id == "dQw4w9WgXcQ"
        assert v.title == "Never Gonna Give You Up"
        assert v.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert v.duration == 213
        assert v.channel == "Rick Astley"
        assert v.upload_date == "19871027"
        assert v.playlist_title == "80s Classics"

    def test_optional_fields_accept_none(self):
        v = _make_video(duration=None, channel=None, upload_date=None, playlist_title=None)
        assert v.duration is None
        assert v.channel is None
        assert v.upload_date is None
        assert v.playlist_title is None

    def test_frozen_rejects_mutation(self):
        v = _make_video()
        with pytest.raises(AttributeError):
            v.title = "changed"  # type: ignore[misc]

    def test_frozen_rejects_new_attribute(self):
        v = _make_video()
        with pytest.raises((AttributeError, TypeError)):
            v.new_field = "x"  # type: ignore[attr-defined]


class TestVideoMetaEquality:
    def test_equal_instances(self):
        v1 = _make_video()
        v2 = _make_video()
        assert v1 == v2

    def test_different_video_id_not_equal(self):
        v1 = _make_video(video_id="aaa")
        v2 = _make_video(video_id="bbb")
        assert v1 != v2

    def test_hashable(self):
        v = _make_video()
        # frozen dataclasses are hashable and can be used in sets/dicts
        s = {v}
        assert v in s

    def test_usable_as_dict_key(self):
        v = _make_video()
        d = {v: "value"}
        assert d[v] == "value"


class TestVideoMetaWatchUrl:
    def test_watch_url_format(self):
        v = _make_video(video_id="dQw4w9WgXcQ")
        assert v.watch_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_watch_url_uses_video_id_not_url_field(self):
        # watch_url is derived from video_id, independent of the url field
        v = _make_video(video_id="abc123XYZ00", url="https://youtu.be/abc123XYZ00")
        assert v.watch_url == "https://www.youtube.com/watch?v=abc123XYZ00"

    def test_watch_url_with_short_video_id(self):
        v = _make_video(video_id="abcdef")
        assert v.watch_url == "https://www.youtube.com/watch?v=abcdef"

    def test_watch_url_with_special_chars_in_id(self):
        # YouTube IDs use [A-Za-z0-9_-]
        v = _make_video(video_id="A_B-C1D2E3F")
        assert v.watch_url == "https://www.youtube.com/watch?v=A_B-C1D2E3F"


class TestVideoMetaTimestampUrl:
    def test_timestamp_url_equals_watch_url(self):
        v = _make_video(video_id="dQw4w9WgXcQ")
        assert v.timestamp_url == v.watch_url

    def test_timestamp_url_suitable_for_appending(self):
        v = _make_video(video_id="dQw4w9WgXcQ")
        stamped = v.timestamp_url + "&t=42"
        assert stamped == "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42"

    def test_timestamp_url_different_video_ids(self):
        v1 = _make_video(video_id="vid1111111")
        v2 = _make_video(video_id="vid2222222")
        assert v1.timestamp_url != v2.timestamp_url


class TestVideoMetaEdgeCases:
    def test_empty_string_title(self):
        v = _make_video(title="")
        assert v.title == ""

    def test_unicode_title(self):
        v = _make_video(title="日本語タイトル — テスト動画 🎬")
        assert v.title == "日本語タイトル — テスト動画 🎬"

    def test_zero_duration(self):
        v = _make_video(duration=0)
        assert v.duration == 0

    def test_very_long_duration(self):
        v = _make_video(duration=86400)  # 24 hours
        assert v.duration == 86400
