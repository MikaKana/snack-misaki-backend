"""Configuration helpers for the Snack Misaki backend."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Settings:
    """Runtime configuration derived from environment variables."""

    use_local_llm: bool
    openai_api_key: Optional[str]
    bedrock_credentials: Optional[str]
    huggingface_token: Optional[str]

    @classmethod
    def from_env(cls) -> "Settings":
        """Build a :class:`Settings` instance using environment variables."""

        def _get_optional(name: str) -> Optional[str]:
            value = os.getenv(name)
            if value is None or value.strip() == "":
                return None
            return value

        use_local_llm = os.getenv("USE_LOCAL_LLM", "false").lower() in {"1", "true", "yes", "on"}

        return cls(
            use_local_llm=use_local_llm,
            openai_api_key=_get_optional("OPENAI_API_KEY"),
            bedrock_credentials=_get_optional("BEDROCK_CREDENTIALS"),
            huggingface_token=_get_optional("HUGGINGFACE_TOKEN"),
        )


__all__ = ["Settings"]