"""External API adapters used in Stage 3."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional
from urllib import request

from .base import LLMClient

LOGGER = logging.getLogger(__name__)


class ExternalLLMConfigurationError(RuntimeError):
    """Raised when an external LLM call cannot be performed due to configuration."""


@dataclass
class ExternalLLMClient(LLMClient):
    """Simple HTTP based LLM client.

    The real project integrates with providers such as OpenAI, Bedrock or
    HuggingFace.  For the purposes of the reference implementation we provide a
    minimal HTTP client that targets an API compatible with OpenAI's chat
    completion endpoint.  The URL can be configured via the ``LLM_API_ENDPOINT``
    environment variable.  When no endpoint is supplied we simply echo a polite
    acknowledgement back to the caller.
    """

    api_key: Optional[str] = None
    endpoint: Optional[str] = None

    def generate(self, prompt: str) -> str:
        prompt = prompt.strip()
        if not prompt:
            return "本日はどのようにお手伝いしましょうか？"

        if not self.endpoint:
            LOGGER.info("No external endpoint configured; returning default acknowledgement")
            return "外部モデルとの連携は現在待機中です。"

        data = json.dumps(
            {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are Snack Misaki, a polite bar hostess."},
                    {"role": "user", "content": prompt},
                ],
            }
        ).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            req = request.Request(self.endpoint, data=data, headers=headers)
            with request.urlopen(req, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network failures covered in unit tests
            LOGGER.warning("External LLM request failed: %s", exc)
            raise ExternalLLMConfigurationError("Failed to reach the external LLM API") from exc

        choices = payload.get("choices")
        if not choices:
            raise ExternalLLMConfigurationError("Unexpected response from external LLM API")

        message = choices[0].get("message", {}).get("content")
        if not message:
            raise ExternalLLMConfigurationError("External LLM response did not contain text")

        return str(message)


def from_environment() -> ExternalLLMClient:
    """Construct an :class:`ExternalLLMClient` using environment variables."""

    endpoint = os.getenv("LLM_API_ENDPOINT")
    api_key = os.getenv("OPENAI_API_KEY")
    return ExternalLLMClient(api_key=api_key, endpoint=endpoint)


__all__ = ["ExternalLLMClient", "ExternalLLMConfigurationError", "from_environment"]