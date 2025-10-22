"""AWS Lambda handler for Snack Misaki."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .llm.external import from_environment as external_from_env
from .llm.local import LocalLLMConfigurationError
from .router import LLMRouter

LOGGER = logging.getLogger(__name__)


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


def _attempt_external_fallback(user_input: str) -> Optional[LambdaResponse]:
    """Try to generate a response using the external Stage 3 model."""

    external_client = external_from_env()
    try:
        response_text = external_client.generate(user_input)
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

    router = LLMRouter()
    routing = router.select(user_input)

    try:
        response_text = routing.client.generate(user_input)
    except LocalLLMConfigurationError as exc:
        LOGGER.error("Local LLM configuration error: %s", exc)
        if routing.engine == "local":
            fallback = _attempt_external_fallback(user_input)
            if fallback is not None:
                return fallback.to_dict()
        return build_error_response("Failed to generate response", status=500).to_dict()
    except Exception as exc:  # pragma: no cover - defensive fallback
        LOGGER.exception("Failed to generate response: %s", exc)
        if routing.engine == "local":
            fallback = _attempt_external_fallback(user_input)
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