"""Base types and abstract interface for LLM providers.

Every concrete provider (Ollama, LM Studio, OpenAI, Anthropic, Gemini)
implements `LLMProvider.invoke()` and returns an `LLMResponse`.

`LLMResponse` is designed as a drop-in replacement for the old
`ClaudeResponse` so downstream stages need minimal changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMResponse:
    """Unified response from any LLM provider.

    Fields mirror the old ``ClaudeResponse`` for backward compatibility.
    """

    text: str
    model: str
    provider: str = "unknown"
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    total_cost_usd: float | None = None
    duration_ms: int | None = None
    session_id: str | None = None
    stop_reason: str | None = None
    raw: dict[str, Any] | None = None

    @property
    def total_tokens(self) -> int:
        """Sum of fresh input + cache creation + output (cache reads are ~free)."""
        return (
            (self.input_tokens or 0) + (self.cache_creation_tokens or 0) + (self.output_tokens or 0)
        )


# Backward compatibility alias — existing code that imports ClaudeResponse
# continues to work without changes.
ClaudeResponse = LLMResponse


class LLMError(RuntimeError):
    """Raised when an LLM provider call fails.

    ``transient=True`` marks errors worth retrying (network blips,
    rate limits, 5xx).
    """

    def __init__(self, msg: str, *, transient: bool = False) -> None:
        super().__init__(msg)
        self.transient = transient


# Backward compatibility alias.
ClaudeCliError = LLMError


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short identifier: 'ollama', 'lmstudio', 'openai', 'anthropic', 'gemini'."""

    @abstractmethod
    def invoke(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model: str = "default",
        timeout: int = 600,
        max_retries: int = 3,
        retry_base_delay: float = 5.0,
        messages: list[dict[str, str]] | None = None,
    ) -> LLMResponse:
        """Send a prompt to the LLM and return a structured response.

        Parameters
        ----------
        prompt:
            The user message text.
        system_prompt:
            Optional system prompt.
        model:
            Model identifier (provider-specific).
        timeout:
            Request timeout in seconds.
        max_retries:
            Maximum retry count for transient errors.
        retry_base_delay:
            Base delay in seconds for exponential backoff.
        messages:
            Optional conversation history for multi-turn. When provided,
            ``prompt`` is appended as the latest user message.
        """

    def health_check(self) -> bool:
        """Return True if the provider is reachable. Override in subclasses."""
        return True
