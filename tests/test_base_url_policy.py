"""SSRF allowlist for OpenAI-compatible provider base URLs (#10)."""

from __future__ import annotations

import pytest

from pipeline_youtube.providers.base import LLMError
from pipeline_youtube.providers.base_url_policy import validate_base_url


class TestValidateBaseUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "https://api.openai.com/v1",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "http://localhost:11434/v1",
            "http://127.0.0.1:1234/v1",
            "http://[::1]:11434/v1",
        ],
    )
    def test_default_allowed_hosts_pass(self, url: str) -> None:
        assert validate_base_url(url) == url

    @pytest.mark.parametrize(
        "url",
        [
            "ftp://api.openai.com/v1",
            "file:///etc/passwd",
            "gopher://api.openai.com/",
        ],
    )
    def test_non_http_scheme_rejected(self, url: str) -> None:
        with pytest.raises(LLMError, match="scheme"):
            validate_base_url(url)

    def test_unknown_host_rejected(self) -> None:
        with pytest.raises(LLMError, match="not allowlisted"):
            validate_base_url("https://evil.example.com/v1")

    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/latest/meta-data/",
            "http://169.254.169.254:80/v1",
        ],
    )
    def test_metadata_ip_rejected(self, url: str) -> None:
        with pytest.raises(LLMError, match="metadata"):
            validate_base_url(url)

    def test_metadata_ip_rejected_even_if_allowlisted(self) -> None:
        # Defense in depth: link-local block wins over an explicit allowlist.
        with pytest.raises(LLMError, match="metadata"):
            validate_base_url(
                "http://169.254.169.254/v1",
                extra_allowed_hosts=["169.254.169.254"],
            )

    def test_extra_allowed_host_permits_self_host(self) -> None:
        url = "http://my-gpu.lan:11434/v1"
        assert validate_base_url(url, extra_allowed_hosts=["my-gpu.lan"]) == url

    def test_extra_allowed_private_ip_permitted(self) -> None:
        url = "http://192.168.1.50:11434/v1"
        assert validate_base_url(url, extra_allowed_hosts=["192.168.1.50"]) == url

    def test_no_host_rejected(self) -> None:
        with pytest.raises(LLMError, match="no host"):
            validate_base_url("http:///v1")
