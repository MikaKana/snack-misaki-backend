from __future__ import annotations

from app.llm.utils import clean_llama_completion
from app.persona import build_character_prompt, format_llama_chat_prompt


def test_clean_llama_completion_removes_prompt_echo():
    conversation = "\n".join(
        [
            "user: こんばんは",
            "assistant: いらっしゃいませ",
            "user: 今日のおすすめは？",
        ]
    )
    prompt = build_character_prompt(conversation)
    formatted_prompt = format_llama_chat_prompt(prompt)

    raw_output = (
        f"{formatted_prompt}"
        "<|assistant|>\n今日のおすすめは完熟の苺と旬の日本酒です。\n"
        "<|user|>\nありがとうございます\n"
        "<|assistant|>\nどういたしまして。\n"
        "> EOF by user"
    )

    cleaned = clean_llama_completion(raw_output, prompt=formatted_prompt)
    assert cleaned == "今日のおすすめは完熟の苺と旬の日本酒です。"


def test_clean_llama_completion_without_tokens():
    raw_output = "ただいま準備中です。"
    assert clean_llama_completion(raw_output) == raw_output