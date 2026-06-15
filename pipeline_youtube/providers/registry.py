"""Provider registry — creates, caches, and exposes LLM providers.

Central entry point for all LLM calls in the pipeline. Replaces the old
``providers.claude_cli.invoke_claude()`` with a provider-agnostic
``invoke_llm()``.

Provider resolution
-------------------
Each pipeline role (``router``, ``stage_02``, ``stage_04``, ``alpha``,
``beta``, ``leader``, ``reviewer``) maps to a ``(provider, model)``
pair in ``config.json``. The registry lazily instantiates each provider
backend on first use and caches it for the process lifetime.

Backward compatibility
----------------------
``invoke_claude`` is kept as an alias of ``invoke_llm`` so existing
call sites compile without changes during incremental migration.
"""

from __future__ import annotations

import os
import threading
from dataclasses import asdict, fields
from typing import Any

from .base import ClaudeCliError, LLMError, LLMProvider, LLMResponse

# Re-export for backward compatibility.
__all__ = [
    "invoke_llm",
    "invoke_claude",
    "get_provider",
    "configure_providers",
    "configure_llm_concurrency",
    "LLMResponse",
    "LLMError",
    "ClaudeCliError",
]

# Global registry state.
_providers_config: dict[str, Any] = {}
_models_config: dict[str, dict[str, str]] = {}
_provider_cache: dict[str, LLMProvider] = {}
# Guards get-or-create of _provider_cache under raised --concurrency, where
# multiple worker threads may resolve the same provider simultaneously.
_provider_lock = threading.Lock()

# LLM-output cache policy (per-role). Stage 02/04 + the router transform a
# fixed input deterministically, so caching them makes re-runs / --synthesis-only
# near-instant. Stage 05 synthesis is creative/cross-video — users iterate on it,
# so fresh output is the sane default. ``--no-cache`` disables everything.
_LLM_CACHE_STAGE_ROLES = frozenset({"router", "stage_02", "stage_04"})
_LLM_CACHE_SYNTHESIS_ROLES = frozenset({"alpha", "beta", "leader", "reviewer"})
_llm_cache_stages_enabled = True
_llm_cache_synthesis_enabled = False


def configure_llm_cache(*, stages: bool = True, synthesis: bool = False) -> None:
    """Set the per-role LLM-output cache policy (called from ``main.cli()``)."""
    global _llm_cache_stages_enabled, _llm_cache_synthesis_enabled
    _llm_cache_stages_enabled = stages
    _llm_cache_synthesis_enabled = synthesis


# Resource-class semaphore (Phase 3 A): bounds concurrent LLM API calls
# independently of the per-video --concurrency. ``None`` = unbounded (default,
# preserves prior behavior). When set, lets --concurrency rise without
# over-subscribing the provider's rate/connection budget — downloads and other
# stages fill the slack instead.
_llm_semaphore: threading.BoundedSemaphore | None = None


def configure_llm_concurrency(limit: int | None) -> None:
    """Cap concurrent LLM provider calls. ``None`` / <=0 removes the cap."""
    global _llm_semaphore
    _llm_semaphore = threading.BoundedSemaphore(limit) if limit and limit > 0 else None


def _llm_cache_enabled_for_role(role: str | None) -> bool:
    if role in _LLM_CACHE_STAGE_ROLES:
        return _llm_cache_stages_enabled
    if role in _LLM_CACHE_SYNTHESIS_ROLES:
        return _llm_cache_synthesis_enabled
    return False  # unknown/None role: never cache (safe default)


# Default provider/model when config is not set (useful for tests).
_DEFAULT_PROVIDER = "ollama"
_DEFAULT_MODEL = "qwen3:8b"

# Default base URLs per provider.
_DEFAULT_BASE_URLS: dict[str, str] = {
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}


def configure_providers(
    providers_config: dict[str, Any],
    models_config: dict[str, dict[str, str]] | None = None,
) -> None:
    """Initialize the registry from config.json data.

    Called once from ``main.cli()`` at startup.
    """
    global _providers_config, _models_config
    _providers_config = dict(providers_config)
    _models_config = dict(models_config or {})
    _provider_cache.clear()


def _resolve_env_vars(value: str) -> str:
    """Replace ``${ENV_VAR}`` patterns with environment variable values."""
    if not value.startswith("${") or not value.endswith("}"):
        return value
    env_name = value[2:-1]
    return os.environ.get(env_name, "")


def get_provider(provider_name: str) -> LLMProvider:
    """Get or create a cached provider instance (thread-safe)."""
    cached = _provider_cache.get(provider_name)
    if cached is not None:
        return cached

    with _provider_lock:
        # Re-check inside the lock: another thread may have just built it.
        cached = _provider_cache.get(provider_name)
        if cached is not None:
            return cached
        return _build_provider(provider_name)


def _build_provider(provider_name: str) -> LLMProvider:
    cfg = _providers_config.get(provider_name, {})
    provider: LLMProvider

    if provider_name == "anthropic":
        from .anthropic_sdk import AnthropicProvider

        api_key = _resolve_env_vars(cfg.get("api_key", "${ANTHROPIC_API_KEY}"))
        provider = AnthropicProvider(api_key=api_key or None)
    else:
        # OpenAI-compatible: ollama, lmstudio, openai, gemini, or custom.
        from .openai_compat import OpenAICompatibleProvider

        base_url = cfg.get("base_url", _DEFAULT_BASE_URLS.get(provider_name, ""))
        if not base_url:
            raise LLMError(
                f"No base_url configured for provider {provider_name!r}. "
                f"Set it in config.json providers.{provider_name}.base_url"
            )

        raw_key = cfg.get("api_key", "")
        api_key = _resolve_env_vars(raw_key) if raw_key else provider_name
        default_model = cfg.get("default_model", _DEFAULT_MODEL)

        provider = OpenAICompatibleProvider(
            base_url=base_url,
            api_key=api_key,
            provider_name=provider_name,
            default_model=default_model,
        )

    _provider_cache[provider_name] = provider
    return provider


def resolve_role(role: str) -> tuple[str, str]:
    """Return ``(provider_name, model)`` for a pipeline role.

    Falls back to the default provider/model if the role isn't
    explicitly configured.
    """
    role_cfg = _models_config.get(role, {})
    if isinstance(role_cfg, dict):
        provider_name = role_cfg.get("provider", _DEFAULT_PROVIDER)
        model = role_cfg.get("model", _DEFAULT_MODEL)
    elif isinstance(role_cfg, str):
        # Legacy format: models.router = "haiku"
        # Assume anthropic for known aliases, default provider otherwise.
        from .anthropic_sdk import _MODEL_ALIASES

        if role_cfg.lower() in _MODEL_ALIASES:
            provider_name = "anthropic"
        else:
            provider_name = _DEFAULT_PROVIDER
        model = role_cfg
    else:
        provider_name = _DEFAULT_PROVIDER
        model = _DEFAULT_MODEL
    return provider_name, model


def invoke_llm(
    prompt: str,
    *,
    system_prompt: str | None = None,
    model: str = "default",
    provider_name: str | None = None,
    role: str | None = None,
    timeout: int = 600,
    max_retries: int = 3,
    retry_base_delay: float = 5.0,
    messages: list[dict[str, str]] | None = None,
    web_search: bool = False,
    thinking: bool = False,
    # Legacy kwargs (accepted but ignored for backward compat).
    append_system_prompt: str | None = None,
    disallow_tools: bool = True,
    resume_session: str | None = None,
    persist_session: bool = False,
    max_budget_usd: float | None = None,
    extra_args: list[str] | None = None,
) -> LLMResponse:
    """Unified entry point for all LLM calls.

    Provider and model can be specified explicitly, or resolved from
    a pipeline ``role`` (e.g. 'router', 'stage_02', 'alpha') via
    the config.json models mapping.

    Legacy parameters (``append_system_prompt``, ``disallow_tools``,
    ``resume_session``, ``persist_session``, ``max_budget_usd``,
    ``extra_args``) are accepted for backward compatibility but
    have no effect in SDK mode.
    """
    # Handle append_system_prompt → system_prompt mapping.
    if append_system_prompt and not system_prompt:
        system_prompt = append_system_prompt

    # Resolve provider and model from role if not explicitly set.
    if role and not provider_name:
        resolved_provider, resolved_model = resolve_role(role)
        if provider_name is None:
            provider_name = resolved_provider
        if model == "default":
            model = resolved_model
    elif provider_name is None:
        provider_name = _DEFAULT_PROVIDER

    # LLM-output cache (per-role policy). Multi-turn calls (``messages``)
    # carry conversation state that the (provider, model, system, prompt)
    # key does not capture, so they bypass the cache.
    from ..cache import get_cache, llm_key

    cache = get_cache()
    use_cache = cache.enabled and messages is None and _llm_cache_enabled_for_role(role)
    key = ""
    if use_cache:
        key = llm_key(provider_name, model, system_prompt, prompt)
        cached = cache.get_llm(key)
        if cached is not None:
            restored = _llm_response_from_cache(cached)
            if restored is not None:
                return restored

    provider = get_provider(provider_name)
    # Acquire the LLM resource slot only around the actual network call — cache
    # hits above never consume one. ``None`` semaphore = unbounded.
    sem = _llm_semaphore
    if sem is not None:
        with sem:
            response = provider.invoke(
                prompt,
                system_prompt=system_prompt,
                model=model,
                timeout=timeout,
                max_retries=max_retries,
                retry_base_delay=retry_base_delay,
                messages=messages,
                web_search=web_search,
                thinking=thinking,
            )
    else:
        response = provider.invoke(
            prompt,
            system_prompt=system_prompt,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            messages=messages,
            web_search=web_search,
            thinking=thinking,
        )
    if use_cache:
        cache.put_llm(key, _llm_response_to_cache(response))
    return response


# Fields excluded from the cached form: ``raw`` (may hold non-serializable
# SDK objects) and ``session_id`` (run-specific, meaningless on replay).
_LLM_CACHE_SKIP_FIELDS = frozenset({"raw", "session_id"})


def _llm_response_to_cache(response: LLMResponse) -> dict[str, Any]:
    return {k: v for k, v in asdict(response).items() if k not in _LLM_CACHE_SKIP_FIELDS}


def _llm_response_from_cache(data: dict[str, Any]) -> LLMResponse | None:
    valid = {f.name for f in fields(LLMResponse)}
    try:
        return LLMResponse(**{k: v for k, v in data.items() if k in valid})
    except (TypeError, ValueError):
        return None


# Backward compatibility alias.
invoke_claude = invoke_llm
