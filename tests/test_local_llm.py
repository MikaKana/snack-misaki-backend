from __future__ import annotations

import sys
import types

import pytest

from app.llm.local import LocalLLMClient, LocalLLMConfigurationError
from app.persona import format_llama_chat_prompt


@pytest.fixture(autouse=True)
def restore_modules():
    snapshot = dict(sys.modules)
    try:
        yield
    finally:
        for name in list(sys.modules.keys()):
            if name not in snapshot:
                del sys.modules[name]
        sys.modules.update(snapshot)
        LocalLLMClient.clear_cache()


def test_gpt4all_backend_is_used_when_available(tmp_path):
    module = types.SimpleNamespace()

    class FakeGPT4All:
        def __init__(self, model_name: str, model_path: str | None = None):
            self.model_name = model_name
            self.model_path = model_path

        def generate(self, prompt: str, max_tokens: int, temp: float):
            return f"generated:{prompt}:{max_tokens}:{temp}"

    module.GPT4All = FakeGPT4All
    sys.modules["gpt4all"] = module

    model_path = tmp_path / "fake-model.bin"
    model_path.write_text("binary data")

    client = LocalLLMClient(model_path=str(model_path), backend="gpt4all", max_tokens=64, temperature=0.2)
    assert client.generate("こんにちは") == "generated:こんにちは:64:0.2"


def test_llama_cpp_backend_when_selected(tmp_path):
    captured: dict[str, str] = {}

    class FakeLlama:
        def __init__(self, model_path: str, **kwargs):
            self.model_path = model_path
            self.kwargs = kwargs

        def create_completion(self, prompt: str, max_tokens: int, temperature: float):
            captured["prompt"] = prompt
            captured["max_tokens"] = max_tokens
            captured["temperature"] = temperature
            return {"choices": [{"text": "llama-response"}]}

    sys.modules["llama_cpp"] = types.SimpleNamespace(Llama=FakeLlama)

    model_path = tmp_path / "fake-llama.gguf"
    model_path.write_text("binary data")

    client = LocalLLMClient(model_path=str(model_path), backend="llama.cpp", max_tokens=32, temperature=0.5)
    expected_prompt = format_llama_chat_prompt("おすすめは？")
    assert client.generate("おすすめは？") == f"llama:{expected_prompt}:32:0.5"


def test_missing_backend_raises_error_when_backend_unavailable():
    client = LocalLLMClient(model_path=None, backend="gpt4all")

    with pytest.raises(LocalLLMConfigurationError):
        client.generate("今日のおすすめは？")


def test_empty_prompt_raises_error(tmp_path):
    class FakeLlama:
        def __init__(self, model_path: str, **kwargs):
            self.model_path = model_path

        def create_completion(self, prompt: str, max_tokens: int, temperature: float):
            return {"choices": [{"text": "ignored"}]}

    sys.modules["llama_cpp"] = types.SimpleNamespace(Llama=FakeLlama)

    model_path = tmp_path / "fake-llama.gguf"
    model_path.write_text("binary data")

    client = LocalLLMClient(model_path=str(model_path), backend="llama.cpp")

    with pytest.raises(LocalLLMConfigurationError):
        client.generate("   ")


def test_invalid_backend_raises_error():
    client = LocalLLMClient(model_path="/tmp/model.bin", backend="unknown")
    with pytest.raises(LocalLLMConfigurationError):
        client.generate("hello")


def test_model_cache_reuses_loaded_instances(tmp_path):
    class FakeLlama:
        instances = 0

        def __init__(self, model_path: str, **kwargs):
            self.model_path = model_path
            self.kwargs = kwargs
            FakeLlama.instances += 1

        def create_completion(self, prompt: str, max_tokens: int, temperature: float):
            return {"choices": [{"text": f"cached:{prompt}:{max_tokens}:{temperature}"}]}

    sys.modules["llama_cpp"] = types.SimpleNamespace(Llama=FakeLlama)

    model_path = tmp_path / "cached-model.gguf"
    model_path.write_text("binary data")

    client_one = LocalLLMClient(model_path=str(model_path), backend="llama.cpp", max_tokens=16, temperature=0.1)
    expected_prompt_one = format_llama_chat_prompt("こんばんは")
    assert client_one.generate("こんばんは") == f"cached:{expected_prompt_one}:16:0.1"

    client_two = LocalLLMClient(model_path=str(model_path), backend="llama.cpp", max_tokens=32, temperature=0.2)
    expected_prompt_two = format_llama_chat_prompt("おすすめは？")
    assert client_two.generate("おすすめは？") == f"cached:{expected_prompt_two}:32:0.2"

    assert FakeLlama.instances == 1
