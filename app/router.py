"""Routing logic that selects between local and external models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import Settings
from .llm.base import LLMClient, SupportsGenerate
from .llm.external import ExternalLLMClient, from_environment as external_from_env
from .llm.local import LocalLLMClient


@dataclass
class RoutingResult:
    """Outcome of a routing decision."""

    engine: str
    client: SupportsGenerate


class LLMRouter:
    """Choose between the local and external LLM pipelines."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings.from_env()

    def select(self, user_input: str) -> RoutingResult:
        """Select the most appropriate LLM for the ``user_input``."""

        text = user_input.strip().lower()
        if self.settings.use_local_llm and not self._requires_external(text):
            return RoutingResult(engine="local", client=LocalLLMClient.from_environment())

        external_client: LLMClient = external_from_env()
        return RoutingResult(engine="external", client=external_client)

    @staticmethod
    def _requires_external(text: str) -> bool:
        """Heuristic that decides when an external model is required."""

        if not text:
            return False

        keywords = ("高度", "翻訳", "英語", "英訳", "要約", "analysis", "summarize")
        return any(keyword in text for keyword in keywords)


__all__ = ["LLMRouter", "RoutingResult"]