"""Unified LLM client — routes calls to Anthropic or Google AI Studio.

Both providers expose the same interface::

    client = get_llm_client(settings)
    raw_text, usage = client.call(system_prompt, user_content)

``usage`` is always ``{"input_tokens": int, "output_tokens": int}``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger("llm.client")


class LLMClient(ABC):
    """Abstract base for provider-specific LLM clients."""

    @abstractmethod
    def call(self, system: str, user_content: str) -> tuple[str, dict]:
        """Send a system + user prompt and return (raw_text, usage).

        Args:
            system: The system prompt string.
            user_content: The user message / digest text.

        Returns:
            ``(raw_text, usage)`` where usage is
            ``{"input_tokens": int, "output_tokens": int}``.
        """


class AnthropicClient(LLMClient):
    """Anthropic Claude client via the ``anthropic`` SDK."""

    def __init__(self, api_key: str, model: str, max_tokens: int) -> None:
        import anthropic as _anthropic
        self._client = _anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def call(self, system: str, user_content: str) -> tuple[str, dict]:
        """Make a ``messages.create`` call to the Anthropic API."""
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )
        raw_text: str = response.content[0].text
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return raw_text, usage


class GoogleClient(LLMClient):
    """Google AI Studio (Gemini) client via ``google-generativeai``."""

    def __init__(self, api_key: str, model: str, max_tokens: int) -> None:
        import google.generativeai as genai  # type: ignore[import]
        genai.configure(api_key=api_key)
        self._model_name = model
        self._max_tokens = max_tokens
        self._genai = genai

    def call(self, system: str, user_content: str) -> tuple[str, dict]:
        """Make a ``generate_content`` call to the Gemini API.

        The system prompt is prepended to the user content since the
        ``google-generativeai`` SDK handles system instructions via the
        model's ``system_instruction`` parameter.
        """
        model = self._genai.GenerativeModel(
            model_name=self._model_name,
            system_instruction=system,
            generation_config=self._genai.GenerationConfig(
                max_output_tokens=self._max_tokens,
                temperature=0.0,
            ),
        )
        response = model.generate_content(user_content)
        raw_text: str = response.text

        # Extract token counts — present in usage_metadata when available.
        meta = getattr(response, "usage_metadata", None)
        input_tokens = getattr(meta, "prompt_token_count", 0) or 0
        output_tokens = getattr(meta, "candidates_token_count", 0) or 0

        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        return raw_text, usage


# ── Factory ──────────────────────────────────────────────────────

def get_llm_client(settings: object) -> Optional[LLMClient]:
    """Return the appropriate ``LLMClient`` based on settings.

    Resolution order:
    1. If ``llm_provider == "google"`` and ``google_api_key`` is set
       → ``GoogleClient``.
    2. If ``llm_provider == "anthropic"`` and ``anthropic_api_key`` is set
       → ``AnthropicClient``.
    3. Auto-detect: prefer Anthropic if both keys present, else Google.
    4. If neither key is set → return ``None`` (LLM disabled).

    Args:
        settings: A ``Settings`` instance.

    Returns:
        A ready ``LLMClient``, or ``None`` if no credentials are
        configured.
    """
    provider: str = getattr(settings, "llm_provider", "auto").lower()
    anthropic_key: str = getattr(settings, "anthropic_api_key", "") or ""
    google_key: str = getattr(settings, "google_api_key", "") or ""
    model: str = getattr(settings, "llm_model", "")
    max_tokens: int = getattr(settings, "llm_max_tokens", 1024)

    def _resolve_model_for_provider(p: str) -> str:
        """Return a sensible default model if none is explicitly set."""
        if model:
            return model
        if p == "google":
            return "gemini-2.0-flash"
        return "claude-sonnet-4-6"

    if provider == "google":
        if not google_key:
            logger.warning("llm_provider=google but GOOGLE_API_KEY is not set.")
            return None
        resolved = _resolve_model_for_provider("google")
        logger.info("Using Google AI Studio client (model=%s)", resolved)
        return GoogleClient(google_key, resolved, max_tokens)

    if provider == "anthropic":
        if not anthropic_key:
            logger.warning("llm_provider=anthropic but ANTHROPIC_API_KEY is not set.")
            return None
        resolved = _resolve_model_for_provider("anthropic")
        logger.info("Using Anthropic client (model=%s)", resolved)
        return AnthropicClient(anthropic_key, resolved, max_tokens)

    # Auto-detect.
    if anthropic_key:
        resolved = _resolve_model_for_provider("anthropic")
        logger.info("Auto-selected Anthropic client (model=%s)", resolved)
        return AnthropicClient(anthropic_key, resolved, max_tokens)

    if google_key:
        resolved = _resolve_model_for_provider("google")
        logger.info("Auto-selected Google AI Studio client (model=%s)", resolved)
        return GoogleClient(google_key, resolved, max_tokens)

    logger.info("No LLM API key configured — LLM analysis disabled.")
    return None
