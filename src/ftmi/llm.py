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

# Per-backend default model. Overridable by the FTMI_GEN_MODEL env var or an explicit
# `model=` argument (precedence: argument > env > this default). Gemini is the default
# backend; both the artifact generator and the judge use the same model.
DEFAULT_MODEL = {"anthropic": "claude-sonnet-4-6", "gemini": "gemini-3.5-flash"}


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


def get_generator(backend: str = "gemini", model: str | None = None) -> Generator:
    """Build a `(prompt)->str` generator. Model precedence: `model` arg > FTMI_GEN_MODEL env > DEFAULT_MODEL."""
    if backend not in DEFAULT_MODEL:
        raise ValueError(f"unknown backend {backend!r} (expected {list(DEFAULT_MODEL)})")
    model = model or os.getenv("FTMI_GEN_MODEL") or DEFAULT_MODEL[backend]
    return claude_generator(model) if backend == "anthropic" else gemini_generator(model)
