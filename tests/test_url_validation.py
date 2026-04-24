"""Tests for YouTube URL whitelist validation (H1)."""

from __future__ import annotations

import pytest

from pipeline_youtube.playlist import validate_youtube_url


class TestValidateYouTubeUrlAccepted:
    def test_standard_watch_url(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert validate_youtube_url(url) == url

    def test_short_youtu_be(self):
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert validate_youtube_url(url) == url

    def test_mobile(self):
        url = "https://m.youtube.com/watch?v=dQw4w9WgXcQ"
        assert validate_youtube_url(url) == url

    def test_playlist(self):
        url = "https://www.youtube.com/playlist?list=PLabc"
        assert validate_youtube_url(url) == url

    def test_http_scheme_allowed(self):
        url = "http://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert validate_youtube_url(url) == url


class TestValidateYouTubeUrlRejected:
    def test_file_scheme(self):
        with pytest.raises(ValueError, match="scheme"):
            validate_youtube_url("file:///etc/passwd")

    def test_internal_host(self):
        with pytest.raises(ValueError, match="host"):
            validate_youtube_url("http://localhost/watch?v=abc")

    def test_third_party_host(self):
        with pytest.raises(ValueError, match="host"):
            validate_youtube_url("https://evil.example.com/watch?v=abc")

    def test_ftp_scheme(self):
        with pytest.raises(ValueError, match="scheme"):
            validate_youtube_url("ftp://www.youtube.com/watch?v=abc")

    def test_empty(self):
        with pytest.raises(ValueError):
            validate_youtube_url("")

    def test_spoofed_subdomain(self):
        with pytest.raises(ValueError, match="host"):
            validate_youtube_url("https://www.youtube.com.evil.com/watch?v=abc")

    def test_too_long(self):
        long_url = "https://www.youtube.com/watch?v=abc&" + ("x=y&" * 200)
        with pytest.raises(ValueError, match="exceeds"):
            validate_youtube_url(long_url)

    def test_unknown_path_on_canonical_host(self):
        with pytest.raises(ValueError, match="path"):
            validate_youtube_url("https://www.youtube.com/api/redirect?url=http://evil")

    def test_account_path_rejected(self):
        with pytest.raises(ValueError, match="path"):
            validate_youtube_url("https://www.youtube.com/account")

    def test_youtu_be_with_unexpected_path(self):
        with pytest.raises(ValueError, match="youtu.be"):
            validate_youtube_url("https://youtu.be/admin/settings")


class TestValidateYouTubeUrlPaths:
    def test_shorts(self):
        url = "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        assert validate_youtube_url(url) == url

    def test_live(self):
        url = "https://www.youtube.com/live/dQw4w9WgXcQ"
        assert validate_youtube_url(url) == url

    def test_embed(self):
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        assert validate_youtube_url(url) == url

    def test_bare_root(self):
        url = "https://www.youtube.com/"
        assert validate_youtube_url(url) == url
