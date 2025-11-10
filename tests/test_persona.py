from __future__ import annotations

from app.persona import (
    PROMPT_PREFIX,
    build_character_prompt,
    format_llama_chat_prompt,
)


def test_format_llama_chat_prompt_injects_chat_tokens():
    conversation = "\n".join(
        [
            "user: こんばんは",
            "assistant: いらっしゃいませ",
            "user: 今日のおすすめは？",
        ]
    )
    prompt = build_character_prompt(conversation)

    formatted = format_llama_chat_prompt(prompt)

    assert formatted.startswith("<|system|>\n" + PROMPT_PREFIX)
    assert "<|user|>\nこんばんは\n" in formatted
    assert "<|assistant|>\nいらっしゃいませ\n" in formatted
    assert formatted.rstrip().endswith("<|assistant|>")


def test_format_llama_chat_prompt_is_idempotent():
    existing = "<|system|>\nSYSTEM\n<|user|>\nこんにちは\n<|assistant|>\n"
    assert format_llama_chat_prompt(existing) == existing
