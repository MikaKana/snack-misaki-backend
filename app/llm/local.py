"""Local LLM integrations used during Stage 2 of the project."""
from __future__ import annotations

import importlib
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import ClassVar, Dict, Optional, Tuple

from .base import LLMClient

LOGGER = logging.getLogger(__name__)


class LocalLLMConfigurationError(RuntimeError):
    """Raised when the local LLM cannot be initialised."""


def _load_module(name: str):
    """Load ``name`` using :mod:`importlib` and return the module.

    The helper keeps all import logic in one place so we can easily monkeypatch
    it in unit tests.  A :class:`LocalLLMConfigurationError` is raised when the
    module cannot be imported.
    """

    try:
        return importlib.import_module(name)
    except ImportError as exc:  # pragma: no cover - exercised via unit tests
        raise LocalLLMConfigurationError(f"Local LLM backend '{name}' is not installed") from exc


def _coerce_temperature(value: Optional[str], default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive programming
        LOGGER.warning("Invalid LOCAL_LLM_TEMPERATURE value: %s", value)
        return default


def _coerce_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive programming
        LOGGER.warning("Invalid LOCAL_LLM_MAX_TOKENS value: %s", value)
        return default


@dataclass
class LocalLLMClient(LLMClient):
    """Production-ready wrapper around llama.cpp / GPT4All."""

    model_path: Optional[str] = None
    backend: Optional[str] = None
    max_tokens: int = 256
    temperature: float = 0.7
    _model: Optional[object] = field(default=None, init=False, repr=False)
    _backend_name: Optional[str] = field(default=None, init=False, repr=False)

    _MODEL_CACHE: ClassVar[Dict[Tuple[str, str], object]] = {}
    _CACHE_LOCK: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def from_environment(cls) -> "LocalLLMClient":
        """Create a client configured via environment variables."""

        backend = os.getenv("LOCAL_LLM_BACKEND")
        model_path = os.getenv("LOCAL_LLM_MODEL")
        max_tokens = _coerce_int(os.getenv("LOCAL_LLM_MAX_TOKENS"), default=256)
        temperature = _coerce_temperature(os.getenv("LOCAL_LLM_TEMPERATURE"), default=0.7)
        return cls(
            backend=backend,
            model_path=model_path,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    # The backend detection is split out to make unit testing easier.
    def _ensure_backend(self) -> str:
        if self._model is not None:
            if self._backend_name is None:
                raise LocalLLMConfigurationError("Local LLM backend could not be determined")
            return self._backend_name

        backend = (self.backend or "auto").lower()

        loaders = []
        if backend in {"auto", "gpt4all"}:
            loaders.append(("gpt4all", self._load_gpt4all))
        if backend in {"auto", "llama", "llama.cpp"}:
            loaders.append(("llama.cpp", self._load_llama_cpp))

        if not loaders:
            raise LocalLLMConfigurationError(f"Unknown local LLM backend: {self.backend}")

        attempted = []
        last_error: Optional[Exception] = None
        for backend_name, loader in loaders:
            attempted.append(backend_name)
            try:
                self._model = self._get_or_create_model(backend_name, loader)
            except LocalLLMConfigurationError as exc:
                last_error = exc
                LOGGER.error("Local LLM backend %s unavailable: %s", backend_name, exc)
                continue
            else:
                self._backend_name = backend_name
                break

        if self._model is None:
            message = "Local LLM backend(s) unavailable"
            if attempted:
                message += f" (attempted: {', '.join(attempted)})"
            if last_error is not None:
                message += f": {last_error}"
            raise LocalLLMConfigurationError(message)

        # ``_backend_name`` is set whenever ``_model`` is initialised.
        if self._backend_name is None:
            raise LocalLLMConfigurationError("Local LLM backend could not be determined")

        return self._backend_name

    @classmethod
    def clear_cache(cls) -> None:
        """Reset the in-memory cache of loaded local models."""

        with cls._CACHE_LOCK:
            cls._MODEL_CACHE.clear()

    def _cache_key(self, backend_name: str) -> Tuple[str, str]:
        model_path = os.path.abspath(self.model_path) if self.model_path else ""
        return backend_name, model_path

    def _get_or_create_model(self, backend_name: str, loader):
        cache_key = self._cache_key(backend_name)
        model = self._MODEL_CACHE.get(cache_key)
        if model is not None:
            return model

        with self._CACHE_LOCK:
            model = self._MODEL_CACHE.get(cache_key)
            if model is not None:
                return model
            model = loader()
            self._MODEL_CACHE[cache_key] = model
            return model

    def _load_gpt4all(self):
        module = _load_module("gpt4all")
        model_path = self.model_path
        if not model_path:
            raise LocalLLMConfigurationError(
                "LOCAL_LLM_MODEL must point to a GPT4All model when using the gpt4all backend"
            )
        if not os.path.exists(model_path):
            raise LocalLLMConfigurationError(f"GPT4All model not found at {model_path}")

        try:
            return module.GPT4All(model_name=os.path.basename(model_path), model_path=os.path.dirname(model_path) or None)
        except Exception as exc:  # pragma: no cover - relies on third party library
            raise LocalLLMConfigurationError("Failed to initialise GPT4All") from exc

    def _load_llama_cpp(self):
        module = _load_module("llama_cpp")
        model_path = self.model_path
        if not model_path:
            raise LocalLLMConfigurationError(
                "LOCAL_LLM_MODEL must point to a GGUF/GGML file when using the llama.cpp backend"
            )
        if not os.path.exists(model_path):
            raise LocalLLMConfigurationError(f"llama.cpp model not found at {model_path}")

        try:
            return module.Llama(
                model_path=model_path,
                n_ctx=2048,
                n_threads=_coerce_int(os.getenv("LOCAL_LLM_THREADS"), default=4),
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - relies on third party library
            raise LocalLLMConfigurationError("Failed to initialise llama.cpp") from exc

    def generate(self, prompt: str) -> str:
        prompt = prompt.strip()
        if not prompt:
            raise LocalLLMConfigurationError("Prompt must not be empty")

        backend = self._ensure_backend()

        if backend == "gpt4all":
            # The GPT4All API exposes ``generate`` with ``temp`` instead of ``temperature``.
            try:
                response = self._model.generate(prompt, max_tokens=self.max_tokens, temp=self.temperature)
            except Exception as exc:  # pragma: no cover - third party behaviour
                LOGGER.exception("GPT4All generation failed: %s", exc)
                raise LocalLLMConfigurationError("GPT4All generation failed") from exc
            text = str(response).strip()
            if text:
                return text
            raise LocalLLMConfigurationError("GPT4All returned an empty response")

        if backend == "llama.cpp":
            completion = self._model.create_completion(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            try:
                text = completion["choices"][0]["text"]
            except (KeyError, IndexError, TypeError):  # pragma: no cover - defensive programming
                LOGGER.warning("Unexpected llama.cpp response format: %s", completion)
                raise LocalLLMConfigurationError("llama.cpp response format invalid")
            text = str(text).strip()
            if text:
                return text
            raise LocalLLMConfigurationError("llama.cpp returned an empty response")

        raise LocalLLMConfigurationError(f"Unsupported backend selected: {backend}")


__all__ = ["LocalLLMClient", "LocalLLMConfigurationError"]