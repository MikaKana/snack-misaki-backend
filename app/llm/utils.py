"""Utility helpers for working with llama.cpp style chat prompts."""

from __future__ import annotations

import re

TOKEN_PATTERN = re.compile(r"<\|(assistant|user|system)\|>")


def _normalise_cli_output(text: str) -> str:
    """Return ``text`` with CLI specific markers stripped."""

    if not text:
        return ""

    normalised = text.replace("\r\n", "\n")

    # ``llama-cli`` appends a marker such as ``> EOF by user`` when invoked in
    # batch mode.  We treat anything after this marker as noise.
    eof_marker = "> EOF"
    if eof_marker in normalised:
        normalised = normalised.split(eof_marker, 1)[0]

    lines: list[str] = []
    for line in normalised.splitlines():
        # Interactive status lines begin with ``> ``; these do not form part of
        # the model completion so we simply drop them.
        if line.startswith("> "):
            continue
        lines.append(line)

    return "\n".join(lines).strip()


def _assistant_segments(text: str) -> list[str]:
    """Extract assistant message segments from ``text``."""

    if not text:
        return []

    pattern = re.compile(r"<\|assistant\|>\s*(.*?)(?=<\|(assistant|user|system)\|>|$)", re.S)
    segments = []
    for match in pattern.finditer(text):
        content = match.group(1).strip()
        if content:
            segments.append(content)
    return segments


def clean_llama_completion(output: str, *, prompt: str | None = None) -> str:
    """Return the assistant response extracted from ``output``.

    Parameters
    ----------
    output:
        Raw text returned by llama.cpp (either via the Python bindings or the
        command line interface).
    prompt:
        Optional prompt that was supplied to llama.cpp.  When provided we can
        discard any assistant messages that were already present in the prompt
        itself, ensuring the returned text only contains newly generated
        content.
    """

    normalised = _normalise_cli_output(output)
    if not normalised:
        return ""

    if prompt:
        prompt_segments = _assistant_segments(prompt)
    else:
        prompt_segments = []

    completion_segments = _assistant_segments(normalised)

    if completion_segments:
        skip = len(prompt_segments)
        if skip < len(completion_segments):
            return completion_segments[skip]
        return completion_segments[-1]

    # When no chat tokens are present we fall back to removing any stray chat
    # markers that might have been emitted inadvertently.
    stripped = TOKEN_PATTERN.sub("", normalised)
    stripped = stripped.replace("</s>", "").replace("<|end|>", "")
    return stripped.strip()


__all__ = ["clean_llama_completion"]
