"""Abstractions for language model integrations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol


class SupportsGenerate(Protocol):
    """A simple protocol describing objects that can generate text responses."""

    def generate(self, prompt: str) -> str:
        ...


class LLMClient(ABC):
    """Base class for model clients."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a textual response for the given ``prompt``."""


__all__ = ["SupportsGenerate", "LLMClient"]