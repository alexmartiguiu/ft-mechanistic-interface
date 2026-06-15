"""LLM backend for the two API roles: artifact generation and response judging.

The pipeline depends only on a `generator(prompt: str) -> str` callable, so any
backend drops in. `get_generator(backend)` switches between Anthropic and Gemini;
set the matching key. Base-model generation and activation extraction do NOT use
this — they run on-device via transformers.
"""
from __future__ import annotations

import os
from typing import Callable

Generator = Callable[[str], str]

DEFAULT_MODEL = {"anthropic": "claude-sonnet-4-6", "gemini": "gemini-2.5-flash"}


def claude_generator(model: str | None = None, max_tokens: int = 4096) -> Generator:
    """`generator(prompt) -> str` via the Anthropic API. Needs ANTHROPIC_API_KEY."""
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = model or DEFAULT_MODEL["anthropic"]

    def _generate(prompt: str) -> str:
        msg = client.messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    return _generate


def gemini_generator(model: str | None = None) -> Generator:
    """`generator(prompt) -> str` via the Gemini API (as in BAEM). Needs GEMINI_API_KEY."""
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = model or DEFAULT_MODEL["gemini"]

    def _generate(prompt: str) -> str:
        return client.models.generate_content(model=model, contents=prompt).text

    return _generate


def get_generator(backend: str = "anthropic", model: str | None = None) -> Generator:
    """Switch backend by name: 'anthropic' (default) or 'gemini'."""
    if backend == "anthropic":
        return claude_generator(model)
    if backend == "gemini":
        return gemini_generator(model)
    raise ValueError(f"unknown backend {backend!r} (expected 'anthropic' or 'gemini')")
