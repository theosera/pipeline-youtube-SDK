"""SSRF guard for OpenAI-compatible provider ``base_url`` values.

The OpenAI-compatible backend (Ollama / LM Studio / OpenAI / Gemini / custom)
takes its endpoint from ``config.json`` (``providers.<name>.base_url``). Without
validation, a config that an attacker can influence could point an
API-key-bearing request at an internal service or a cloud metadata endpoint
(``169.254.169.254``) — a classic SSRF / credential-exfiltration vector.

Policy (allowlist, not blocklist):

  - scheme must be ``http`` or ``https``
  - host must be in :data:`DEFAULT_ALLOWED_HOSTS` (the managed endpoints +
    loopback for local self-host) **or** explicitly allowlisted per provider
    via ``providers.<name>.allowed_base_url_hosts`` (keeps the self-host /
    LAN-box use case working)
  - link-local IP literals (``169.254.0.0/16``, ``fe80::/10`` — i.e. cloud
    metadata) are rejected unconditionally, even if allowlisted, as defense in
    depth.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable
from urllib.parse import urlparse

from .base import LLMError

# Hosts always permitted: the managed OpenAI-compatible endpoints plus loopback
# (Ollama / LM Studio default to localhost). Anything else must be opted in via
# the per-provider ``allowed_base_url_hosts`` config key.
DEFAULT_ALLOWED_HOSTS: frozenset[str] = frozenset(
    {
        "api.openai.com",
        "generativelanguage.googleapis.com",
        "localhost",
        "127.0.0.1",
        "::1",
    }
)

_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})


def validate_base_url(base_url: str, *, extra_allowed_hosts: Iterable[str] = ()) -> str:
    """Return ``base_url`` unchanged if it passes the SSRF allowlist, else raise.

    ``extra_allowed_hosts`` lets a deployment opt a self-hosted endpoint
    (``my-gpu.lan``, ``192.168.1.50`` …) into the allowlist without weakening
    the default policy for everyone else.
    """
    parsed = urlparse(base_url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise LLMError(
            f"Disallowed base_url scheme {parsed.scheme!r} in {base_url!r}: "
            f"only http/https are permitted."
        )
    host = (parsed.hostname or "").lower()
    if not host:
        raise LLMError(f"base_url {base_url!r} has no host component.")

    _reject_metadata_ip(host, base_url)

    allowed = DEFAULT_ALLOWED_HOSTS | {h.lower() for h in extra_allowed_hosts}
    if host not in allowed:
        raise LLMError(
            f"base_url host {host!r} is not allowlisted (from {base_url!r}). "
            f"Allowed: {sorted(allowed)}. Add it to "
            f"providers.<name>.allowed_base_url_hosts in config.json if intended."
        )
    return base_url


def _reject_metadata_ip(host: str, base_url: str) -> None:
    """Hard-block link-local IP literals (cloud metadata) regardless of allowlist."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return  # Not an IP literal; the hostname allowlist governs it.
    if ip.is_link_local:
        raise LLMError(
            f"base_url host {host!r} is a link-local/metadata address "
            f"(from {base_url!r}); blocked to prevent SSRF to cloud metadata."
        )
