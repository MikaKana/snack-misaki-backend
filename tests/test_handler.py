from __future__ import annotations

import json
import os
from typing import Dict

import pytest

from app import handler
from app.config import Settings
from app.router import LLMRouter
from app.llm.local import LocalLLMConfigurationError


class DummyContext:
    function_name = "test"


@pytest.fixture(autouse=True)
def cleanup_env():
    environ_snapshot = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(environ_snapshot)


def invoke(event: Dict[str, object]):
    return handler.lambda_handler(event, DummyContext())


def test_lambda_handler_with_valid_input_uses_local_by_default(monkeypatch):
    os.environ["USE_LOCAL_LLM"] = "true"
    class StubLocalClient:
        def generate(self, prompt: str) -> str:
            return "ローカル応答"

    class StubFactory:
        @classmethod
        def from_environment(cls):
            return StubLocalClient()

    monkeypatch.setattr("app.router.LocalLLMClient", StubFactory)

    event = {"input": "こんばんは"}
    response = invoke(event)
    body = json.loads(response["body"])
    assert response["statusCode"] == 200
    assert body["engine"] == "local"
    assert body["response"] == "ローカル応答"


def test_lambda_handler_invalid_json_body():
    event = {"body": "{invalid"}
    response = invoke(event)
    assert response["statusCode"] == 400
    body = json.loads(response["body"])
    assert body["error"] == "Invalid JSON body"


def test_lambda_handler_missing_input():
    event = {"body": json.dumps({"message": "hi"})}
    response = invoke(event)
    assert response["statusCode"] == 400
    body = json.loads(response["body"])
    assert body["error"] == "Missing 'input' field in request body"


def test_router_prefers_external_when_keyword_present(monkeypatch):
    os.environ["USE_LOCAL_LLM"] = "true"
    router = LLMRouter(Settings.from_env())
    result = router.select("高度な翻訳をお願いします")
    assert result.engine == "external"


def test_router_uses_local_when_enabled(monkeypatch):
    os.environ["USE_LOCAL_LLM"] = "true"
    router = LLMRouter(Settings.from_env())
    result = router.select("今日のおすすめは？")
    assert result.engine == "local"


def test_parse_event_handles_proxy_integration():
    event = {"body": json.dumps({"input": "テスト"})}
    assert handler.parse_event(event) == "テスト"


def test_parse_event_requires_string_input():
    with pytest.raises(ValueError):
        handler.parse_event({"input": 123})


def test_lambda_handler_handles_conversation_payload(monkeypatch):
    os.environ["USE_LOCAL_LLM"] = "true"

    prompts = []

    class RecordingClient:
        def generate(self, prompt: str) -> str:
            prompts.append(prompt)
            return f"echo:{prompt}"

    class RecordingFactory:
        @classmethod
        def from_environment(cls):
            return RecordingClient()

    monkeypatch.setattr("app.router.LocalLLMClient", RecordingFactory)

    payload = {
        "conversation": [
            "user: こんばんは",
            "assistant: いらっしゃいませ",
            "user: おすすめは？",
        ]
    }

    response = invoke({"body": json.dumps(payload)})
    body = json.loads(response["body"])

    assert response["statusCode"] == 200
    assert body["engine"] == "local"
    assert body["response"] == "echo:user: こんばんは\nassistant: いらっしゃいませ\nuser: おすすめは？"
    assert prompts == ["user: こんばんは\nassistant: いらっしゃいませ\nuser: おすすめは？"]


def test_lambda_handler_falls_back_to_external_when_local_fails(monkeypatch):
    os.environ["USE_LOCAL_LLM"] = "true"

    class FailingLocalClient:
        def generate(self, prompt: str) -> str:
            raise LocalLLMConfigurationError("model missing")

    class FailingFactory:
        @classmethod
        def from_environment(cls):
            return FailingLocalClient()

    monkeypatch.setattr("app.router.LocalLLMClient", FailingFactory)

    def fake_fallback(prompt: str):
        return handler.build_success_response("外部応答", "external")

    monkeypatch.setattr(handler, "_attempt_external_fallback", fake_fallback)

    response = invoke({"input": "おすすめは？"})
    body = json.loads(response["body"])
    assert response["statusCode"] == 200
    assert body["engine"] == "external"
    assert body["response"] == "外部応答"