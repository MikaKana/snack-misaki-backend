"""AWS Lambda handler for Snack Misaki."""
from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency is handled gracefully
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - executed when python-dotenv is missing
    def load_dotenv(*_, **__):
        logging.getLogger(__name__).debug("python-dotenv not installed; skipping load_dotenv()")

from .llm.external import from_environment as external_from_env
from .llm.local import LocalLLMConfigurationError
from .persona import build_character_prompt
from .router import LLMRouter

LOGGER = logging.getLogger(__name__)


load_dotenv()


def _normalise_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _coerce_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        LOGGER.warning("Invalid integer value: %s", value)
        return default


def _coerce_float(value: Optional[str], default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        LOGGER.warning("Invalid float value: %s", value)
        return default


def _should_use_llama_cli() -> bool:
    if not _normalise_bool(os.getenv("USE_LOCAL_LLM")):
        return False
    return os.getenv("LOCAL_LLM_BACKEND", "").strip().lower() == "llama.cpp"


def _run_llama_cli(prompt: str) -> str:
    model = os.getenv("LOCAL_LLM_MODEL", "/app/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    llama_cli = os.getenv("LOCAL_LLM_BINARY", "/app/llama.cpp/build/bin/llama-cli")
    max_tokens = _coerce_int(os.getenv("LOCAL_LLM_MAX_TOKENS"), 256)
    temperature = _coerce_float(os.getenv("LOCAL_LLM_TEMPERATURE"), 0.7)

    command = [
        llama_cli,
        "-m",
        model,
        "-p",
        prompt,
        "-n",
        str(max_tokens),
        "--temp",
        str(temperature),
    ]

    LOGGER.debug("Invoking llama-cli: %s", command)

    result = subprocess.run(command, capture_output=True, text=True, check=True)
    output = result.stdout.strip()
    if not output:
        raise RuntimeError("llama-cli returned an empty response")
    return output


@dataclass
class LambdaResponse:
    """HTTP response wrapper compatible with API Gateway/Lambda proxy."""

    status_code: int
    body: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statusCode": self.status_code,
            "headers": {"Content-Type": "application/json; charset=utf-8"},
            "body": json.dumps(self.body, ensure_ascii=False),
        }


def _normalise_conversation(payload: Dict[str, Any]) -> Optional[str]:
    """Return a textual prompt extracted from ``payload`` if possible."""

    if "input" in payload:
        user_input = payload["input"]
        if not isinstance(user_input, str):
            raise ValueError("'input' must be a string")
        return user_input

    if "conversation" in payload:
        conversation = payload["conversation"]
        if isinstance(conversation, str):
            return conversation
        if isinstance(conversation, list):
            if not all(isinstance(item, str) for item in conversation):
                raise ValueError("'conversation' must be a string or list of strings")
            return "\n".join(conversation)
        raise ValueError("'conversation' must be a string or list of strings")

    if "messages" in payload:
        messages = payload["messages"]
        if not isinstance(messages, list):
            raise ValueError("'messages' must be a list")

        compiled: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Each message must be an object")
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("Each message requires a string 'content'")
            role = message.get("role")
            if isinstance(role, str) and role.strip():
                compiled.append(f"{role.strip()}: {content}")
            else:
                compiled.append(content)

        if compiled:
            return "\n".join(compiled)
        raise ValueError("'messages' must contain at least one item")

    return None


def parse_event(event: Dict[str, Any]) -> str:
    """Extract the user input from an incoming Lambda ``event``."""

    if "body" in event:
        try:
            payload = event["body"]
            if isinstance(payload, str):
                payload = json.loads(payload or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON body") from exc
    else:
        payload = event

    if not isinstance(payload, dict):
        raise ValueError("Event body must be a JSON object")

    conversation = _normalise_conversation(payload)
    if conversation is None:
        raise ValueError("Missing 'input' field in request body")

    return conversation


def build_success_response(text: str, engine: str) -> LambdaResponse:
    return LambdaResponse(
        status_code=200,
        body={"response": text, "engine": engine},
    )


def build_error_response(message: str, status: int = 400) -> LambdaResponse:
    return LambdaResponse(
        status_code=status,
        body={"error": message},
    )


def _attempt_external_fallback(prompt: str) -> Optional[LambdaResponse]:
    """Try to generate a response using the external Stage 3 model."""

    external_client = external_from_env()
    try:
        response_text = external_client.generate(prompt)
    except Exception as exc:  # pragma: no cover - network errors handled defensively
        LOGGER.exception("External LLM fallback failed: %s", exc)
        return None

    return build_success_response(response_text, engine="external")


def lambda_handler(event: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
    """Entry point for AWS Lambda."""

    LOGGER.debug("Received event: %s", event)

    try:
        user_input = parse_event(event)
    except ValueError as exc:
        LOGGER.warning("Invalid event: %s", exc)
        return build_error_response(str(exc), status=400).to_dict()

    persona_prompt = build_character_prompt(user_input)

    if _should_use_llama_cli():
        try:
            response_text = _run_llama_cli(persona_prompt)
        except Exception as exc:
            LOGGER.exception("llama-cli invocation failed: %s", exc)
            return LambdaResponse(
                status_code=500,
                body={"error": str(exc)},
            ).to_dict()
        return build_success_response(response_text, "llama.cpp").to_dict()

    router = LLMRouter()
    routing = router.select(user_input)

    try:
        response_text = routing.client.generate(persona_prompt)
    except LocalLLMConfigurationError as exc:
        LOGGER.error("Local LLM configuration error: %s", exc)
        if routing.engine == "local":
            fallback = _attempt_external_fallback(persona_prompt)
            if fallback is not None:
                return fallback.to_dict()
        return build_error_response("Failed to generate response", status=500).to_dict()
    except Exception as exc:  # pragma: no cover - defensive fallback
        LOGGER.exception("Failed to generate response: %s", exc)
        if routing.engine == "local":
            fallback = _attempt_external_fallback(persona_prompt)
            if fallback is not None:
                return fallback.to_dict()
        return build_error_response("Failed to generate response", status=500).to_dict()

    return build_success_response(response_text, routing.engine).to_dict()


__all__ = [
    "LambdaResponse",
    "build_error_response",
    "build_success_response",
    "lambda_handler",
    "parse_event",
]