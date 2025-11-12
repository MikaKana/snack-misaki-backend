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
from .llm.utils import clean_llama_completion
from .persona import build_character_prompt, format_llama_chat_prompt
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

    formatted_prompt = format_llama_chat_prompt(prompt)

    command = [
        llama_cli,
        "-m",
        model,
        "-p",
        formatted_prompt,
        "-n",
        str(max_tokens),
        "--temp",
        str(temperature),
    ]

    LOGGER.debug("Invoking llama-cli: %s", command)

    result = subprocess.run(command, capture_output=True, text=False, check=True)
    output_data = result.stdout
    if not output_data:
        raise RuntimeError("llama-cli returned an empty response")

    if isinstance(output_data, bytes):
        try:
            output = output_data.decode("utf-8")
        except UnicodeDecodeError:
            LOGGER.warning("llama-cli emitted non-UTF-8 output; characters will be replaced")
            output = output_data.decode("utf-8", errors="replace")
    else:
        output = str(output_data)

    cleaned_output = clean_llama_completion(output, prompt=formatted_prompt)
    return cleaned_output or output.strip()


@dataclass
class LambdaResponse:
    """HTTP response wrapper compatible with API Gateway/Lambda proxy."""

    status_code: int
    body: Dict[str, Any]

    def to_dict(self, *, stringify_body: bool = False) -> Dict[str, Any]:
        """Return a mapping that can be returned from the Lambda handler.

        Parameters
        ----------
        stringify_body:
            When ``True`` the ``body`` is JSON-encoded to a string. This is the
            format expected by API Gateway's Lambda proxy integration. When
            ``False`` the ``body`` is returned as a dictionary, which is useful
            when invoking the Lambda function directly (``Invoke`` API) where
            the payload should only be JSON-encoded once.
        """

        if stringify_body:
            if isinstance(self.body, str):
                try:
                    parsed_body = json.loads(self.body)
                except (TypeError, ValueError):
                    body_content = self.body
                else:
                    body_content = json.dumps(parsed_body, ensure_ascii=False)
            else:
                body_content = json.dumps(self.body, ensure_ascii=False)
        else:
            body_content = self.body

        return {
            "statusCode": self.status_code,
            "headers": {"Content-Type": "application/json; charset=utf-8"},
            "body": body_content,
        }


def _looks_like_apigw_event(event: Dict[str, Any]) -> bool:
    """Return ``True`` when ``event`` resembles an API Gateway payload."""

    if not isinstance(event, dict):
        return False

    if "requestContext" in event:
        return True

    if event.get("version") in {"1.0", "2.0"}:
        return True

    if "resource" in event and "httpMethod" in event:
        return True

    return False


def _should_stringify_response_body(event: Dict[str, Any]) -> bool:
    """Return ``True`` when the Lambda response body should be JSON strings."""

    body = event.get("body")

    if not _looks_like_apigw_event(event):
        return False

    if isinstance(body, (str, bytes)):
        return True

    if body is None:
        return True

    return False


def _finalize_lambda_response(response: LambdaResponse, event: Dict[str, Any]) -> Dict[str, Any]:
    """Convert ``response`` into a mapping suitable for the AWS Lambda runtime."""

    return response.to_dict(stringify_body=_should_stringify_response_body(event))


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
        return _finalize_lambda_response(build_error_response(str(exc), status=400), event)

    persona_prompt = build_character_prompt(user_input)

    if _should_use_llama_cli():
        try:
            response_text = _run_llama_cli(persona_prompt)
        except FileNotFoundError as exc:
            LOGGER.warning("llama-cli binary missing, falling back to Python client: %s", exc)
        except Exception as exc:
            LOGGER.exception("llama-cli invocation failed: %s", exc)
            return _finalize_lambda_response(
                LambdaResponse(
                    status_code=500,
                    body={"error": str(exc)},
                ),
                event,
            )
        else:
            return _finalize_lambda_response(
                build_success_response(response_text, "llama.cpp"),
                event,
            )

    router = LLMRouter()
    routing = router.select(user_input)

    try:
        response_text = routing.client.generate(persona_prompt)
    except LocalLLMConfigurationError as exc:
        LOGGER.error("Local LLM configuration error: %s", exc)
        if routing.engine == "local":
            fallback = _attempt_external_fallback(persona_prompt)
            if fallback is not None:
                return _finalize_lambda_response(fallback, event)
            return _finalize_lambda_response(
                build_error_response("Failed to generate response", status=500),
                event,
            )
    except Exception as exc:  # pragma: no cover - defensive fallback
        LOGGER.exception("Failed to generate response: %s", exc)
        if routing.engine == "local":
            fallback = _attempt_external_fallback(persona_prompt)
            if fallback is not None:
                return _finalize_lambda_response(fallback, event)
        return _finalize_lambda_response(
            build_error_response("Failed to generate response", status=500),
            event,
        )

    return _finalize_lambda_response(
        build_success_response(response_text, routing.engine),
        event,
    )


__all__ = [
    "LambdaResponse",
    "build_error_response",
    "build_success_response",
    "lambda_handler",
    "parse_event",
]
