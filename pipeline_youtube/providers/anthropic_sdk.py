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

    def __init__(self, api_key: str | None = None) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise LLMError(
                "Anthropic API key not configured. "
                "Set ANTHROPIC_API_KEY env var or add to config.json providers.anthropic.api_key"
            )
        self._client = anthropic.Anthropic(api_key=resolved_key)

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
    ) -> LLMResponse:
        effective_model = _resolve_model(model)

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
                    "max_tokens": 8192,
                    "messages": msgs,
                    "timeout": timeout,
                }
                if system_prompt:
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

                # Estimate cost (approximate, for logging only).
                cost = _estimate_cost(effective_model, input_tokens, output_tokens)

                return LLMResponse(
                    text=text,
                    model=effective_model,
                    provider="anthropic",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
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


def _estimate_cost(model: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    """Return estimated cost in USD, or None if pricing is unknown."""
    pricing = _PRICING.get(model)
    if pricing is None:
        return None
    inp = (input_tokens or 0) / 1_000_000 * pricing[0]
    out = (output_tokens or 0) / 1_000_000 * pricing[1]
    return round(inp + out, 6)
