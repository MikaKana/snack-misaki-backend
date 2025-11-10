"""Persona helpers for Snack Misaki's mama."""
from __future__ import annotations

PERSONA_DESCRIPTION = (
    "あなたはスナック美砂樹のママ。優しい女言葉で、明るく前向きでウィットに富んだ会話を楽しみます。"
    "来店したお客様にはすぐにフルーツとお酒をおすすめしたくなります。"
)
PERSONA_BEHAVIOUR = "会話では常にそのキャラクターを保ち、日本語で丁寧にお話ししてください。"
PROMPT_PREFIX = f"{PERSONA_DESCRIPTION}{PERSONA_BEHAVIOUR}次の内容にお答えください。"


def build_character_prompt(message: str) -> str:
    """Return ``message`` wrapped in the Snack Misaki persona instructions."""

    text = message.strip()
    if not text:
        return PROMPT_PREFIX
    return f"{PROMPT_PREFIX}\n\n{text}"


def format_llama_chat_prompt(prompt: str) -> str:
    """Convert ``prompt`` into the TinyLlama chat template."""

    raw = prompt or ""

    # Avoid reformatting prompts that already contain chat tokens.
    if any(token in raw for token in ("<|system|>", "<|user|>", "<|assistant|>")):
        return raw

    text = raw.strip()
    if not text:
        return PROMPT_PREFIX

    if text.startswith(PROMPT_PREFIX):
        system_prompt = PROMPT_PREFIX
        conversation = text[len(PROMPT_PREFIX) :].strip()
    else:
        system_prompt = SYSTEM_PROMPT
        conversation = text

    messages: list[tuple[str, str]] = []
    if conversation:
        for raw_line in conversation.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            role = "user"
            content = line
            if ":" in line:
                candidate_role, remainder = line.split(":", 1)
                candidate_role = candidate_role.strip().lower()
                remainder = remainder.strip()
                if candidate_role in {"user", "assistant", "system"} and remainder:
                    role = candidate_role
                    content = remainder
                else:
                    content = remainder or line

            if not content:
                continue

            messages.append((role, content))

    prompt_parts = [f"<|system|>\n{system_prompt}\n"]
    token_map = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}
    for role, content in messages:
        prompt_parts.append(f"{token_map.get(role, '<|user|>')}\n{content}\n")

    prompt_parts.append("<|assistant|>\n")
    return "".join(prompt_parts)


SYSTEM_PROMPT = f"{PERSONA_DESCRIPTION}{PERSONA_BEHAVIOUR}"


__all__ = [
    "PERSONA_DESCRIPTION",
    "PERSONA_BEHAVIOUR",
    "PROMPT_PREFIX",
    "SYSTEM_PROMPT",
    "build_character_prompt",
    "format_llama_chat_prompt",
]