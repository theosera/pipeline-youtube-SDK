"""OpenAI-compatible LLM provider.

Covers **Ollama**, **LM Studio**, **OpenAI**, and **Gemini**
(via its OpenAI-compatible endpoint) through a single implementation.

Each backend is distinguished only by ``base_url`` and ``api_key``:

  - Ollama:    http://localhost:11434/v1   api_key="ollama"
  - LM Studio: http://localhost:1234/v1    api_key="lm-studio"
  - OpenAI:    https://api.openai.com/v1   api_key=<real key>
  - Gemini:    https://generativelanguage.googleapis.com/v1beta/openai
               api_key=<GEMINI_API_KEY>
"""

from __future__ import annotations

import sys
import time

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from .base import LLMError, LLMProvider, LLMResponse

# Known transient HTTP status codes.
_TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class OpenAICompatibleProvider(LLMProvider):
    """Provider for any OpenAI-compatible ``/v1/chat/completions`` endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "ollama",
        provider_name: str = "ollama",
        default_model: str = "qwen3:8b",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._provider = provider_name
        self._default_model = default_model
        self._client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            # Long timeout for large-context local models.
            timeout=600.0,
        )

    @property
    def provider_name(self) -> str:
        return self._provider

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
        web_search: bool = False,
        thinking: bool = False,
    ) -> LLMResponse:
        # web_search / thinking are Anthropic-only; OpenAI-compatible backends
        # (ollama, lmstudio, openai, gemini) ignore them. Stage 01b correction
        # is pinned to Anthropic, so this no-op is not hit in normal use.
        del web_search, thinking
        effective_model = model if model != "default" else self._default_model

        # Build messages list.
        msgs: list[dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        if messages:
            msgs.extend(messages)
        msgs.append({"role": "user", "content": prompt})

        last_exc: LLMError | None = None
        for attempt in range(max_retries + 1):
            try:
                t0 = time.monotonic()
                response = self._client.chat.completions.create(
                    model=effective_model,
                    messages=msgs,  # type: ignore[arg-type]
                    timeout=timeout,
                )
                duration_ms = int((time.monotonic() - t0) * 1000)

                choice = response.choices[0] if response.choices else None
                text = choice.message.content or "" if choice else ""
                stop_reason = choice.finish_reason if choice else None

                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else None
                output_tokens = usage.completion_tokens if usage else None

                return LLMResponse(
                    text=text,
                    model=response.model or effective_model,
                    provider=self._provider,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_cost_usd=None,  # Local models: no cost
                    duration_ms=duration_ms,
                    stop_reason=stop_reason,
                    raw=response.model_dump() if hasattr(response, "model_dump") else None,
                )

            except APITimeoutError as e:
                raise LLMError(
                    f"{self._provider} timeout after {timeout}s: {e}",
                    transient=False,
                ) from e

            except APIStatusError as e:
                transient = e.status_code in _TRANSIENT_STATUS_CODES
                if not transient or attempt >= max_retries:
                    raise LLMError(
                        f"{self._provider} API error {e.status_code}: {e.message}",
                        transient=transient,
                    ) from e
                last_exc = LLMError(str(e), transient=True)

            except APIConnectionError as e:
                if attempt >= max_retries:
                    raise LLMError(
                        f"{self._provider} connection error: {e}",
                        transient=True,
                    ) from e
                last_exc = LLMError(str(e), transient=True)

            delay = retry_base_delay * (2**attempt)
            sys.stderr.write(
                f"[{self._provider} retry {attempt + 1}/{max_retries}] "
                f"sleeping {delay:.0f}s: {str(last_exc)[:200]}\n"
            )
            sys.stderr.flush()
            time.sleep(delay)

        assert last_exc is not None
        raise last_exc

    def health_check(self) -> bool:
        """Check if the endpoint is reachable by listing models."""
        try:
            self._client.models.list()
            return True
        except Exception:
            return False
