"""SSRF allowlist for OpenAI-compatible provider base URLs (#10)."""

from __future__ import annotations

import pytest

from pipeline_youtube.providers.base import LLMError
from pipeline_youtube.providers.base_url_policy import validate_base_url


class TestValidateBaseUrl:
    @pytest.mark.parametrize(
        ("url", "provider"),
        [
            ("https://api.openai.com/v1", "openai"),
            ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini"),
            ("http://localhost:11434/v1", "ollama"),
            ("http://127.0.0.1:1234/v1", "lmstudio"),
            ("http://[::1]:11434/v1", "ollama"),
        ],
    )
    def test_provider_defaults_pass(self, url: str, provider: str) -> None:
        assert validate_base_url(url, provider_name=provider) == url

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
            validate_base_url(url, provider_name="openai")

    def test_unknown_host_rejected(self) -> None:
        with pytest.raises(LLMError, match="not allowlisted"):
            validate_base_url("https://evil.example.com/v1", provider_name="openai")

    # P1: loopback is not a global default — a credentialed provider must not
    # accept a config-controlled loopback base_url without an explicit opt-in.
    def test_loopback_rejected_for_managed_provider(self) -> None:
        with pytest.raises(LLMError, match="not allowlisted"):
            validate_base_url("http://localhost:8080/v1", provider_name="openai")

    def test_loopback_opt_in_for_managed_provider(self) -> None:
        url = "http://localhost:8080/v1"
        assert (
            validate_base_url(url, provider_name="openai", extra_allowed_hosts=["localhost"]) == url
        )

    # P2: managed/public hosts must use https (no cleartext API keys).
    @pytest.mark.parametrize(
        ("url", "provider"),
        [
            ("http://api.openai.com/v1", "openai"),
            ("http://generativelanguage.googleapis.com/v1beta/openai", "gemini"),
        ],
    )
    def test_http_rejected_for_managed_host(self, url: str, provider: str) -> None:
        with pytest.raises(LLMError, match="https is required"):
            validate_base_url(url, provider_name=provider)

    def test_https_for_self_host_opt_in_allowed(self) -> None:
        url = "https://my-gpu.lan:11434/v1"
        assert (
            validate_base_url(url, provider_name="ollama", extra_allowed_hosts=["my-gpu.lan"])
            == url
        )

    def test_http_allowed_for_self_host_opt_in(self) -> None:
        url = "http://192.168.1.50:11434/v1"
        assert (
            validate_base_url(url, provider_name="lmstudio", extra_allowed_hosts=["192.168.1.50"])
            == url
        )

    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/latest/meta-data/",
            "http://169.254.169.254:80/v1",
        ],
    )
    def test_metadata_ip_rejected(self, url: str) -> None:
        with pytest.raises(LLMError, match="metadata"):
            validate_base_url(url, provider_name="ollama")

    def test_metadata_ip_rejected_even_if_allowlisted(self) -> None:
        # Defense in depth: link-local block wins over an explicit allowlist.
        with pytest.raises(LLMError, match="metadata"):
            validate_base_url(
                "http://169.254.169.254/v1",
                provider_name="ollama",
                extra_allowed_hosts=["169.254.169.254"],
            )

    def test_unknown_provider_requires_opt_in(self) -> None:
        # A custom provider has no default hosts; its host must be opted in.
        with pytest.raises(LLMError, match="not allowlisted"):
            validate_base_url("https://my-llm.example/v1", provider_name="myorg")
        url = "https://my-llm.example/v1"
        assert (
            validate_base_url(url, provider_name="myorg", extra_allowed_hosts=["my-llm.example"])
            == url
        )

    def test_no_host_rejected(self) -> None:
        with pytest.raises(LLMError, match="no host"):
            validate_base_url("http:///v1", provider_name="ollama")
