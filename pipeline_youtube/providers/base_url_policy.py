"""SSRF guard for OpenAI-compatible provider ``base_url`` values.

The OpenAI-compatible backend (Ollama / LM Studio / OpenAI / Gemini / custom)
takes its endpoint from ``config.json`` (``providers.<name>.base_url``). Without
validation, a config that an attacker can influence could point an
API-key-bearing request at an internal service or a cloud metadata endpoint
(``169.254.169.254``) — a classic SSRF / credential-exfiltration vector.

Policy (per-provider allowlist, not a blocklist):

  - scheme must be ``http`` or ``https``
  - host must be allowed **for that provider**: each provider has a default host
    set (:data:`_PROVIDER_DEFAULT_HOSTS`) — loopback for the local-by-design
    backends (Ollama / LM Studio), the real managed host for OpenAI / Gemini —
    plus any per-provider opt-in via ``providers.<name>.allowed_base_url_hosts``
    (keeps the self-host / LAN-box use case working).
  - plain ``http`` is only allowed for loopback or an explicitly opted-in host;
    managed/public hosts must use ``https`` so a real API key is never sent in
    cleartext.
  - link-local IP literals (``169.254.0.0/16``, ``fe80::/10`` — i.e. cloud
    metadata) are rejected unconditionally, even if allowlisted, as defense in
    depth.

Loopback is deliberately **not** a global default: a credentialed provider
(``openai`` / ``gemini`` with a real key) only reaches its managed host unless
the operator explicitly opts a loopback/self-host endpoint in, so a
config-controlled ``base_url`` cannot redirect a key-bearing request to an
arbitrary local service.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable
from urllib.parse import urlparse

from .base import LLMError

# Loopback hosts — the only hosts that may be reached over plain http by default
# (local self-host backends commonly run without TLS).
_LOOPBACK_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})

# Per-provider default host allowlist. Local-by-design backends default to
# loopback; managed backends default to their real public host (https-only,
# enforced below). Unknown/custom providers default to nothing and must declare
# their host via ``allowed_base_url_hosts``.
_PROVIDER_DEFAULT_HOSTS: dict[str, frozenset[str]] = {
    "ollama": _LOOPBACK_HOSTS,
    "lmstudio": _LOOPBACK_HOSTS,
    "openai": frozenset({"api.openai.com"}),
    "gemini": frozenset({"generativelanguage.googleapis.com"}),
}

_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})


def validate_base_url(
    base_url: str,
    *,
    provider_name: str,
    extra_allowed_hosts: Iterable[str] = (),
) -> str:
    """Return ``base_url`` unchanged if it passes the SSRF allowlist, else raise.

    ``provider_name`` selects the default host set (loopback for local backends,
    the managed host for OpenAI / Gemini). ``extra_allowed_hosts`` opts a
    self-hosted endpoint (``my-gpu.lan``, ``192.168.1.50`` …) into the allowlist
    without weakening the default policy for everyone else.
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

    extra = {h.lower() for h in extra_allowed_hosts}
    allowed = _PROVIDER_DEFAULT_HOSTS.get(provider_name, frozenset()) | extra
    if host not in allowed:
        raise LLMError(
            f"base_url host {host!r} is not allowlisted for provider "
            f"{provider_name!r} (from {base_url!r}). Allowed: {sorted(allowed)}. "
            f"Add it to providers.{provider_name}.allowed_base_url_hosts in "
            f"config.json if intended."
        )

    # Plain http only for loopback or an explicitly opted-in self-host endpoint;
    # managed/public hosts must use https so the API key is never sent in clear.
    if parsed.scheme == "http" and host not in (_LOOPBACK_HOSTS | extra):
        raise LLMError(
            f"base_url {base_url!r} uses plain http for managed host {host!r}; "
            f"https is required (plain http is only allowed for loopback or a "
            f"host listed in providers.{provider_name}.allowed_base_url_hosts)."
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
