from __future__ import annotations

import json
import os
import subprocess
from typing import Dict

import pytest

from app import handler
from app.config import Settings
from app.llm.local import LocalLLMConfigurationError
from app.persona import build_character_prompt
from app.router import LLMRouter


class DummyContext:
    function_name = "test"


@pytest.fixture(autouse=True)
def cleanup_env():
    environ_snapshot = os.environ.copy()
    for key in list(os.environ):
        if key.startswith("LOCAL_LLM_"):
            os.environ.pop(key, None)
    yield
    os.environ.clear()
    os.environ.update(environ_snapshot)


def invoke(event: Dict[str, object]):
    return handler.lambda_handler(event, DummyContext())


def get_body(response: Dict[str, object]):
    body = response["body"]
    if isinstance(body, str):
        return json.loads(body)
    return body


def test_finalize_lambda_response_keeps_dict_for_direct_invocations():
    response = handler.build_success_response("テスト", "local")
    result = handler._finalize_lambda_response(response, {"body": json.dumps({"input": "hi"})})
    assert isinstance(result["body"], dict)


def test_finalize_lambda_response_stringifies_for_apigw_events():
    response = handler.build_success_response("テスト", "local")
    event = {
        "body": json.dumps({"input": "hi"}),
        "requestContext": {"accountId": "123456789012"},
    }
    result = handler._finalize_lambda_response(response, event)
    assert isinstance(result["body"], str)
    assert json.loads(result["body"]) == {"response": "テスト", "engine": "local"}


def test_finalize_lambda_response_does_not_escape_japanese_characters():
    response = handler.build_success_response("こんにちは", "local")
    event = {
        "body": json.dumps({"input": "hi"}),
        "requestContext": {"accountId": "123456789012"},
    }
    result = handler._finalize_lambda_response(response, event)
    assert "\\u" not in result["body"]
    assert "こんにちは" in result["body"]


def test_build_success_response_decodes_unicode_sequences():
    encoded = "\\u3053\\u3093\\u306b\\u3061\\u306f"
    response = handler.build_success_response(encoded, "local")
    assert response.body["response"] == "こんにちは"


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
    body = get_body(response)
    assert response["statusCode"] == 200
    assert body["engine"] == "local"
    assert body["response"] == "ローカル応答"


def test_lambda_handler_invalid_json_body():
    event = {"body": "{invalid"}
    response = invoke(event)
    assert response["statusCode"] == 400
    body = get_body(response)
    assert body["error"] == "Invalid JSON body"


def test_lambda_handler_missing_input():
    event = {"body": json.dumps({"message": "hi"})}
    response = invoke(event)
    assert response["statusCode"] == 400
    body = get_body(response)
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


def test_parse_event_accepts_user_field():
    event = {"body": json.dumps({"user": "こんばんは"})}
    assert handler.parse_event(event) == "こんばんは"


def test_parse_event_accepts_user_list():
    payload = {"user": ["こんばんは", "おげんきですか"]}
    assert handler.parse_event(payload) == "こんばんは\nおげんきですか"

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

    conversation_lines = [
        "user: こんばんは",
        "assistant: いらっしゃいませ",
        "user: おすすめは？",
    ]
    payload = {"conversation": conversation_lines}
    conversation_text = "\n".join(conversation_lines)
    expected_prompt = build_character_prompt(conversation_text)

    response = invoke({"body": json.dumps(payload)})
    body = get_body(response)

    assert response["statusCode"] == 200
    assert body["engine"] == "local"
    assert body["response"] == f"echo:{expected_prompt}"
    assert prompts == [expected_prompt]


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
    body = get_body(response)
    assert response["statusCode"] == 200
    assert body["engine"] == "external"
    assert body["response"] == "外部応答"


def test_lambda_handler_missing_llama_cli_falls_back_to_local(monkeypatch):
    os.environ["USE_LOCAL_LLM"] = "true"
    os.environ["LOCAL_LLM_BACKEND"] = "llama.cpp"

    class StubLocalClient:
        def generate(self, prompt: str) -> str:
            return "ローカル応答"

    class StubFactory:
        @classmethod
        def from_environment(cls):
            return StubLocalClient()

    monkeypatch.setattr("app.router.LocalLLMClient", StubFactory)

    def fake_run(command, capture_output, text, check):
        raise FileNotFoundError("llama-cli missing")

    monkeypatch.setattr(handler.subprocess, "run", fake_run)

    response = invoke({"input": "こんばんは"})
    body = get_body(response)

    assert response["statusCode"] == 200
    assert body["engine"] == "local"
    assert body["response"] == "ローカル応答"


def test_lambda_handler_invokes_llama_cli(monkeypatch):
    os.environ["USE_LOCAL_LLM"] = "true"
    os.environ["LOCAL_LLM_BACKEND"] = "llama.cpp"
    os.environ["LOCAL_LLM_MODEL"] = "/app/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    os.environ["LOCAL_LLM_MAX_TOKENS"] = "128"
    os.environ["LOCAL_LLM_TEMPERATURE"] = "0.5"

    captured = {}

    def fake_run(command, capture_output, text, check):
        captured["command"] = command

        class Result:
            stdout = "ローカル応答\n"

        return Result()

    monkeypatch.setattr(handler.subprocess, "run", fake_run)

    response = invoke({"input": "こんばんは"})
    body = get_body(response)

    assert response["statusCode"] == 200
    assert body["engine"] == "llama.cpp"
    assert body["response"] == "ローカル応答"
    assert captured["command"][0] == "/app/llama.cpp/build/bin/llama-cli"
    assert "-m" in captured["command"]
    assert "-p" in captured["command"]
    assert "-n" in captured["command"]
    assert "--temp" in captured["command"]


def test_lambda_handler_returns_error_when_llama_cli_fails(monkeypatch):
    os.environ["USE_LOCAL_LLM"] = "true"
    os.environ["LOCAL_LLM_BACKEND"] = "llama.cpp"

    def fake_run(command, capture_output, text, check):
        raise subprocess.CalledProcessError(returncode=1, cmd=command, stderr="boom")

    monkeypatch.setattr(handler.subprocess, "run", fake_run)

    response = invoke({"input": "失敗テスト"})
    body = get_body(response)

    assert response["statusCode"] == 500
    assert "error" in body
