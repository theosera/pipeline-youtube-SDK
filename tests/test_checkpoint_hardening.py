"""M3 — defensive tests for `extract_trusted_video_id`.

The checkpoint helper refuses to honor any file that does not look like
pipeline-produced output. These tests encode the three trust layers:

  1. Well-formed `---...---` frontmatter block required.
  2. `video_id` must match YouTube's 11-char `[A-Za-z0-9_-]` format.
  3. If `URL:` is present, it must reference the same video_id.
"""

from __future__ import annotations

from pipeline_youtube.checkpoint import extract_trusted_video_id

_VALID = "abc123DEFGH"  # 11 chars, matches [A-Za-z0-9_-]


def _fm(*lines: str) -> bytes:
    body = "\n".join(("---", *lines, "---", "", "Body.", ""))
    return body.encode("utf-8")


class TestHappyPath:
    def test_accepts_minimal_valid_frontmatter(self):
        data = _fm(
            "date: 2026-04-18 12:34",
            'title: "x"',
            f'video_id: "{_VALID}"',
        )
        assert extract_trusted_video_id(data) == _VALID

    def test_accepts_frontmatter_with_matching_url(self):
        data = _fm(
            f'URL: "https://www.youtube.com/watch?v={_VALID}"',
            f'video_id: "{_VALID}"',
        )
        assert extract_trusted_video_id(data) == _VALID

    def test_accepts_youtu_be_short_url(self):
        data = _fm(
            f'URL: "https://youtu.be/{_VALID}"',
            f'video_id: "{_VALID}"',
        )
        assert extract_trusted_video_id(data) == _VALID

    def test_accepts_id_with_underscore_and_hyphen(self):
        vid = "_abc-123xyz"  # 11 chars, includes _ and -
        data = _fm(
            f'URL: "https://www.youtube.com/watch?v={vid}"',
            f'video_id: "{vid}"',
        )
        assert extract_trusted_video_id(data) == vid


class TestLayer1FrontmatterBlock:
    def test_rejects_file_without_leading_delimiter(self):
        data = f'video_id: "{_VALID}"\n'.encode()
        assert extract_trusted_video_id(data) is None

    def test_rejects_video_id_only_in_body(self):
        """`video_id:` found outside the frontmatter block must be ignored."""
        data = (
            b'---\ntitle: "no id here"\n---\n\n'
            b'Spoof payload: video_id: "' + _VALID.encode("utf-8") + b'"\n'
        )
        assert extract_trusted_video_id(data) is None

    def test_rejects_unclosed_frontmatter(self):
        data = f'---\ntitle: "x"\nvideo_id: "{_VALID}"\n'.encode()
        assert extract_trusted_video_id(data) is None


class TestLayer2VideoIdFormat:
    def test_rejects_too_short_id(self):
        data = _fm('video_id: "abc123"')  # 6 chars
        assert extract_trusted_video_id(data) is None

    def test_rejects_too_long_id(self):
        data = _fm('video_id: "abc123DEFGHJK"')  # 13 chars
        assert extract_trusted_video_id(data) is None

    def test_rejects_invalid_character(self):
        data = _fm('video_id: "abc123DEFG!"')  # 11 chars but contains !
        assert extract_trusted_video_id(data) is None

    def test_rejects_path_traversal_style(self):
        data = _fm('video_id: "../../etc"')  # path-like, not 11 chars
        assert extract_trusted_video_id(data) is None

    def test_rejects_empty_id(self):
        data = _fm('video_id: ""')
        assert extract_trusted_video_id(data) is None


class TestLayer3UrlIntegrity:
    def test_rejects_url_referencing_different_id(self):
        data = _fm(
            'URL: "https://www.youtube.com/watch?v=xxxxxxxxxxx"',
            f'video_id: "{_VALID}"',
        )
        assert extract_trusted_video_id(data) is None

    def test_rejects_url_with_no_id_reference(self):
        data = _fm(
            'URL: "https://example.com/other"',
            f'video_id: "{_VALID}"',
        )
        assert extract_trusted_video_id(data) is None

    def test_accepts_when_url_field_absent(self):
        """URL cross-check is conditional — missing URL still accepted."""
        data = _fm(
            "date: 2026-04-18 12:34",
            f'video_id: "{_VALID}"',
        )
        assert extract_trusted_video_id(data) == _VALID


class TestRobustness:
    def test_handles_invalid_utf8_gracefully(self):
        data = b'---\n\xff\xfe\nvideo_id: "' + _VALID.encode("utf-8") + b'"\n---\n'
        # Non-UTF-8 bytes decode with 'replace' and the regex still works
        # only when the frontmatter structure survives — either way, no raise.
        result = extract_trusted_video_id(data)
        assert result is None or result == _VALID

    def test_handles_empty_bytes(self):
        assert extract_trusted_video_id(b"") is None

    def test_caps_scan_at_first_2kb(self):
        """A malicious large file with video_id far beyond the scan window
        must not be matched (self-DoS prevention + bounded read cost)."""
        padding = b"x" * 4096
        data = b"---\n" + padding + b'\nvideo_id: "' + _VALID.encode("utf-8") + b'"\n---\n'
        assert extract_trusted_video_id(data) is None
