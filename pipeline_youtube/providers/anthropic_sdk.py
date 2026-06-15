"""Anthropic Messages API provider.

Uses the ``anthropic`` Python SDK to call Claude models directly via
API key authentication (no OAuth / ``claude -p`` CLI dependency).

Model aliases
-------------
Short aliases are resolved to full model IDs:

  - ``sonnet``  → ``claude-sonnet-4-20250514``
  - ``haiku``   → ``claude-haiku-4-20250514``
  - ``opus``    → ``claude-4-opus-20250514``

Full model IDs (e.g. ``claude-sonnet-4-20250514``) are passed through
as-is.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import anthropic

from .base import LLMError, LLMProvider, LLMResponse

_MODEL_ALIASES: dict[str, str] = {
    "sonnet": "claude-sonnet-4-20250514",
    "haiku": "claude-haiku-4-20250514",
    "opus": "claude-4-opus-20250514",
}

# Known transient Anthropic error types.
_TRANSIENT_ERROR_TYPES = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,
)


def _resolve_model(alias: str) -> str:
    """Resolve a short alias to a full model ID, or pass through."""
    return _MODEL_ALIASES.get(alias.lower(), alias)


class AnthropicProvider(LLMProvider):
    """Provider for the Anthropic Messages API."""

    def __init__(self, api_key: str | None = None, *, prompt_caching: bool = True) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise LLMError(
                "Anthropic API key not configured. "
                "Set ANTHROPIC_API_KEY env var or add to config.json providers.anthropic.api_key"
            )
        self._client = anthropic.Anthropic(api_key=resolved_key)
        # Prompt caching is on by default; disable via
        # ANTHROPIC_PROMPT_CACHING=0 for debugging or unsupported endpoints.
        env = os.environ.get("ANTHROPIC_PROMPT_CACHING")
        self._prompt_caching = prompt_caching and env not in {"0", "false", "no"}

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def invoke(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model: str = "sonnet",
        timeout: int = 600,
        max_retries: int = 3,
        retry_base_delay: float = 5.0,
        messages: list[dict[str, str]] | None = None,
        web_search: bool = False,
        thinking: bool = False,
    ) -> LLMResponse:
        effective_model = _resolve_model(model)

        # Extended thinking needs headroom: max_tokens must exceed the thinking
        # budget. Bump the ceiling when thinking is on.
        max_tokens = 16000 if thinking else 8192

        # Build messages. Anthropic uses a separate `system` parameter.
        msgs: list[dict[str, str]] = []
        if messages:
            msgs.extend(messages)
        msgs.append({"role": "user", "content": prompt})

        last_exc: LLMError | None = None
        for attempt in range(max_retries + 1):
            try:
                t0 = time.monotonic()

                kwargs: dict[str, Any] = {
                    "model": effective_model,
                    "max_tokens": max_tokens,
                    "messages": msgs,
                    "timeout": timeout,
                }
                if web_search:
                    # Server-side web search tool: the model decides when to
                    # search and Anthropic runs it server-side, returning the
                    # final text. Used by Stage 01b to fact-check terms.
                    kwargs["tools"] = [
                        {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
                    ]
                if thinking:
                    kwargs["thinking"] = {"type": "enabled", "budget_tokens": 4096}
                if system_prompt:
                    # Mark the system prompt as an ephemeral prompt-cache
                    # breakpoint. Stage system prompts (e.g. SUMMARY_SYSTEM_PROMPT)
                    # are large and identical across every video, so caching the
                    # prefix turns repeated calls into cheap cache reads (~0.1x
                    # input price) instead of re-billing the full prompt. Below
                    # the model's minimum cacheable length the marker is ignored
                    # by the API, so this is always safe.
                    if self._prompt_caching:
                        kwargs["system"] = [
                            {
                                "type": "text",
                                "text": system_prompt,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ]
                    else:
                        kwargs["system"] = system_prompt

                response = self._client.messages.create(**kwargs)
                duration_ms = int((time.monotonic() - t0) * 1000)

                text_parts = []
                for block in response.content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                text = "\n".join(text_parts)

                usage = response.usage
                input_tokens = usage.input_tokens if usage else None
                output_tokens = usage.output_tokens if usage else None
                # Cache-related token counts (None on SDK versions / responses
                # that don't surface them). `input_tokens` already excludes
                # cached tokens, so the three buckets sum to total input.
                cache_creation = getattr(usage, "cache_creation_input_tokens", None)
                cache_read = getattr(usage, "cache_read_input_tokens", None)

                # Estimate cost (approximate, for logging only).
                cost = _estimate_cost(
                    effective_model,
                    input_tokens,
                    output_tokens,
                    cache_creation_tokens=cache_creation,
                    cache_read_tokens=cache_read,
                )

                return LLMResponse(
                    text=text,
                    model=effective_model,
                    provider="anthropic",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_creation_tokens=cache_creation,
                    cache_read_tokens=cache_read,
                    total_cost_usd=cost,
                    duration_ms=duration_ms,
                    stop_reason=response.stop_reason,
                    raw=response.model_dump() if hasattr(response, "model_dump") else None,
                )

            except anthropic.APITimeoutError as e:
                raise LLMError(
                    f"anthropic timeout after {timeout}s: {e}",
                    transient=False,
                ) from e

            except _TRANSIENT_ERROR_TYPES as e:
                if attempt >= max_retries:
                    raise LLMError(
                        f"anthropic API error: {e}",
                        transient=True,
                    ) from e
                last_exc = LLMError(str(e), transient=True)

            except anthropic.APIStatusError as e:
                raise LLMError(
                    f"anthropic API error {e.status_code}: {e.message}",
                    transient=False,
                ) from e

            delay = retry_base_delay * (2**attempt)
            sys.stderr.write(
                f"[anthropic retry {attempt + 1}/{max_retries}] "
                f"sleeping {delay:.0f}s: {str(last_exc)[:200]}\n"
            )
            sys.stderr.flush()
            time.sleep(delay)

        assert last_exc is not None
        raise last_exc

    def health_check(self) -> bool:
        """Verify the API key is valid by checking models."""
        try:
            # A lightweight call to verify connectivity.
            self._client.messages.count_tokens(
                model="claude-haiku-4-20250514",
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except Exception:
            return False


# Rough per-token pricing for cost estimation (USD).
_PRICING: dict[str, tuple[float, float]] = {
    # (input_per_million, output_per_million)
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-haiku-4-20250514": (0.80, 4.0),
    "claude-4-opus-20250514": (15.0, 75.0),
}


# Anthropic prompt-cache multipliers relative to the base input price:
# writing a 5-minute ephemeral cache entry costs 1.25x, reading one 0.1x.
_CACHE_WRITE_MULTIPLIER = 1.25
_CACHE_READ_MULTIPLIER = 0.10


def _estimate_cost(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    *,
    cache_creation_tokens: int | None = None,
    cache_read_tokens: int | None = None,
) -> float | None:
    """Return estimated cost in USD, or None if pricing is unknown.

    ``input_tokens`` is the fresh (uncached) input; cache writes and reads
    are billed separately at 1.25x / 0.1x of the base input rate.
    """
    pricing = _PRICING.get(model)
    if pricing is None:
        return None
    in_rate = pricing[0] / 1_000_000
    inp = (input_tokens or 0) * in_rate
    cache_write = (cache_creation_tokens or 0) * in_rate * _CACHE_WRITE_MULTIPLIER
    cache_read = (cache_read_tokens or 0) * in_rate * _CACHE_READ_MULTIPLIER
    out = (output_tokens or 0) / 1_000_000 * pricing[1]
    return round(inp + cache_write + cache_read + out, 6)
