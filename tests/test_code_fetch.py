"""Tests for the code_fetch module (GitHub URL extraction + raw fetch).

Network calls are mocked. Real-world behavior of yt-dlp / GitHub is out
of scope for unit tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pipeline_youtube import code_fetch

# =====================================================
# extract_github_urls
# =====================================================


class TestExtractGithubUrls:
    def test_blob_url_extracted(self):
        desc = (
            "ソースコード: https://github.com/anthropics/anthropic-sdk-python/blob/main/README.md"
        )
        urls = code_fetch.extract_github_urls(desc)
        assert urls == ["https://github.com/anthropics/anthropic-sdk-python/blob/main/README.md"]

    def test_gist_url_extracted(self):
        desc = "Gist: https://gist.github.com/anthropics/abc123def456"
        urls = code_fetch.extract_github_urls(desc)
        assert urls == ["https://gist.github.com/abc123def456"]

    def test_repo_url_extracted_when_no_blob(self):
        desc = "Repository: https://github.com/anthropics/claude-code"
        urls = code_fetch.extract_github_urls(desc)
        assert urls == ["https://github.com/anthropics/claude-code"]

    def test_blob_takes_priority_over_repo_for_same_repo(self):
        desc = """
        Repo: https://github.com/anthropics/claude-code
        File: https://github.com/anthropics/claude-code/blob/main/foo.py
        """
        urls = code_fetch.extract_github_urls(desc)
        # Both still appear (the bare repo URL could refer to a different
        # part), but the blob URL wins by appearing first in iteration
        # order. The repo URL is filtered out only when it duplicates
        # the same owner/repo as an existing blob URL.
        assert "https://github.com/anthropics/claude-code/blob/main/foo.py" in urls
        assert "https://github.com/anthropics/claude-code" not in urls

    def test_dedupes_identical_urls(self):
        desc = """
        First mention: https://github.com/foo/bar/blob/main/a.py
        Second mention: https://github.com/foo/bar/blob/main/a.py
        """
        urls = code_fetch.extract_github_urls(desc)
        assert urls == ["https://github.com/foo/bar/blob/main/a.py"]

    def test_empty_description(self):
        assert code_fetch.extract_github_urls("") == []
        assert code_fetch.extract_github_urls(None) == []  # type: ignore[arg-type]

    def test_no_github_urls_in_description(self):
        desc = "プログラミング動画です。よろしく！ https://twitter.com/foo"
        assert code_fetch.extract_github_urls(desc) == []

    def test_multiple_distinct_urls(self):
        desc = """
        - https://github.com/foo/bar/blob/main/a.py
        - https://gist.github.com/abc123def456
        - https://github.com/baz/qux
        """
        urls = code_fetch.extract_github_urls(desc)
        assert "https://github.com/foo/bar/blob/main/a.py" in urls
        assert "https://gist.github.com/abc123def456" in urls
        assert "https://github.com/baz/qux" in urls

    def test_caps_at_2x_max_for_filtering_room(self):
        urls_in_desc = "\n".join(f"https://github.com/foo/bar{i}/blob/main/a.py" for i in range(20))
        urls = code_fetch.extract_github_urls(urls_in_desc)
        assert len(urls) <= code_fetch.MAX_URLS_PER_VIDEO * 2


# =====================================================
# language detection
# =====================================================


class TestLanguageDetection:
    def test_python_extension(self):
        assert code_fetch._language_for_path("foo/bar.py") == "python"

    def test_typescript_extension(self):
        assert code_fetch._language_for_path("src/index.ts") == "typescript"

    def test_dockerfile_no_extension(self):
        assert code_fetch._language_for_path("docker/Dockerfile") == "dockerfile"
        assert code_fetch._language_for_path("Dockerfile") == "dockerfile"

    def test_unknown_extension(self):
        assert code_fetch._language_for_path("foo.xyz") == ""

    def test_no_extension(self):
        assert code_fetch._language_for_path("LICENSE") == ""


# =====================================================
# fetch_snippets_for_urls
# =====================================================


class TestFetchSnippetsForUrls:
    def test_blob_url_fetches_raw(self):
        with patch.object(
            code_fetch,
            "_fetch_raw",
            return_value=("print('hi')\n", False),
        ) as fetch:
            snippets = code_fetch.fetch_snippets_for_urls(
                ["https://github.com/anthropics/foo/blob/main/x.py"]
            )

        assert len(snippets) == 1
        s = snippets[0]
        assert s.filename == "x.py"
        assert s.language == "python"
        assert s.content == "print('hi')\n"
        assert s.truncated is False
        # Verify the fetch URL is the raw form
        assert (
            fetch.call_args.args[0] == "https://raw.githubusercontent.com/anthropics/foo/main/x.py"
        )

    def test_repo_url_skipped(self):
        """Bare repo URLs should be skipped (no /blob/, no fetch)."""
        with patch.object(code_fetch, "_fetch_raw") as fetch:
            snippets = code_fetch.fetch_snippets_for_urls(
                ["https://github.com/anthropics/claude-code"]
            )
        assert snippets == []
        fetch.assert_not_called()

    def test_capped_at_max(self):
        """Only MAX_URLS_PER_VIDEO snippets are fetched even when more are given."""
        urls = [f"https://github.com/o/r{i}/blob/main/x.py" for i in range(10)]
        with patch.object(
            code_fetch,
            "_fetch_raw",
            return_value=("ok", False),
        ) as fetch:
            snippets = code_fetch.fetch_snippets_for_urls(urls)
        assert len(snippets) == code_fetch.MAX_URLS_PER_VIDEO
        assert fetch.call_count == code_fetch.MAX_URLS_PER_VIDEO

    def test_failed_fetch_skipped(self):
        with patch.object(code_fetch, "_fetch_raw", return_value=None):
            snippets = code_fetch.fetch_snippets_for_urls(["https://github.com/o/r/blob/main/x.py"])
        assert snippets == []

    def test_truncation_flag_propagated(self):
        with patch.object(
            code_fetch,
            "_fetch_raw",
            return_value=("x" * 50_000, True),
        ):
            snippets = code_fetch.fetch_snippets_for_urls(
                ["https://github.com/o/r/blob/main/big.py"]
            )
        assert len(snippets) == 1
        assert snippets[0].truncated is True

    def test_gist_via_api(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"files": {"main.py": {"content": "print(\\"hi\\")"}, "README.md": {"content": "# example"}}}'
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = lambda *a: None

        with patch("urllib.request.urlopen", return_value=mock_resp):
            snippets = code_fetch.fetch_snippets_for_urls(["https://gist.github.com/abc123def456"])

        assert len(snippets) == 1
        s = snippets[0]
        assert s.filename == "gist_abc123def456"
        assert "main.py" in s.content
        assert "README.md" in s.content


# =====================================================
# render_code_section
# =====================================================


class TestRenderCodeSection:
    def test_empty_returns_empty_string(self):
        assert code_fetch.render_code_section([]) == ""

    def test_single_snippet_format(self):
        snippet = code_fetch.CodeSnippet(
            source_url="https://github.com/o/r/blob/main/x.py",
            raw_url="https://raw.githubusercontent.com/o/r/main/x.py",
            filename="x.py",
            language="python",
            content="print('hi')",
            truncated=False,
        )
        rendered = code_fetch.render_code_section([snippet])
        assert "## 関連コード" in rendered
        assert "### [x.py](https://github.com/o/r/blob/main/x.py)" in rendered
        assert "```python" in rendered
        assert "print('hi')" in rendered
        assert "```" in rendered
        assert "truncated" not in rendered

    def test_truncated_snippet_shows_warning(self):
        snippet = code_fetch.CodeSnippet(
            source_url="https://github.com/o/r/blob/main/big.py",
            raw_url="https://raw.githubusercontent.com/o/r/main/big.py",
            filename="big.py",
            language="python",
            content="x" * 100,
            truncated=True,
        )
        rendered = code_fetch.render_code_section([snippet])
        assert "truncated" in rendered

    def test_unknown_language_no_fence_lang(self):
        snippet = code_fetch.CodeSnippet(
            source_url="https://github.com/o/r/blob/main/foo.xyz",
            raw_url="https://raw.githubusercontent.com/o/r/main/foo.xyz",
            filename="foo.xyz",
            language="",
            content="?",
            truncated=False,
        )
        rendered = code_fetch.render_code_section([snippet])
        assert "```\n" in rendered or rendered.split("```")[1].startswith("\n")


# =====================================================
# fetch_video_description (yt-dlp mocked)
# =====================================================


class TestFetchVideoDescription:
    def test_returns_description_string(self):
        fake_ydl = MagicMock()
        fake_ydl.__enter__ = lambda self: self
        fake_ydl.__exit__ = lambda *a: None
        fake_ydl.extract_info.return_value = {"description": "some description"}

        with patch("yt_dlp.YoutubeDL", return_value=fake_ydl):
            result = code_fetch.fetch_video_description("vid001")
        assert result == "some description"

    def test_returns_none_on_extract_failure(self):
        fake_ydl = MagicMock()
        fake_ydl.__enter__ = lambda self: self
        fake_ydl.__exit__ = lambda *a: None
        fake_ydl.extract_info.side_effect = RuntimeError("boom")

        with patch("yt_dlp.YoutubeDL", return_value=fake_ydl):
            result = code_fetch.fetch_video_description("vid001")
        assert result is None

    def test_returns_none_when_description_missing(self):
        fake_ydl = MagicMock()
        fake_ydl.__enter__ = lambda self: self
        fake_ydl.__exit__ = lambda *a: None
        fake_ydl.extract_info.return_value = {"title": "no desc field"}

        with patch("yt_dlp.YoutubeDL", return_value=fake_ydl):
            result = code_fetch.fetch_video_description("vid001")
        assert result is None
