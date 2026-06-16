"""LLM backend for the two API roles: artifact generation and response judging.

The pipeline depends only on a `generator(prompt: str, *, schema=None) -> str` callable,
so any backend drops in. `get_generator(backend)` switches between Anthropic and Gemini;
set the matching key. Base-model generation and activation extraction do NOT use this —
they run on-device via transformers.

Structured outputs
------------------
Pass `schema=<pydantic model>` and the returned string is **provider-enforced JSON** that
conforms to that schema — Gemini via `response_schema`, Anthropic via `output_config`
json_schema. Callers `json.loads` the result; no regex, no fence-stripping.
"""
from __future__ import annotations

import os
from typing import Callable

# generator(prompt, *, schema=None) -> str.  Without schema: raw model text.
# With schema (a pydantic model class): a JSON string conforming to it.
Generator = Callable[..., str]

# Per-backend default model. Overridable by the FTMI_GEN_MODEL env var or an explicit
# `model=` argument (precedence: argument > env > this default). Gemini is the default
# backend; both the artifact generator and the judge use the same model.
DEFAULT_MODEL = {"anthropic": "claude-sonnet-4-6", "gemini": "gemini-3.5-flash"}


def _json_schema(schema) -> dict:
    """A pydantic model class -> JSON Schema dict (or pass a dict through unchanged)."""
    return schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema


def claude_generator(model: str | None = None, max_tokens: int = 4096) -> Generator:
    """`generator(prompt, *, schema=None) -> str` via the Anthropic API. Needs ANTHROPIC_API_KEY."""
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = model or DEFAULT_MODEL["anthropic"]

    def _generate(prompt: str, *, schema=None) -> str:
        kwargs = dict(model=model, max_tokens=max_tokens,
                      messages=[{"role": "user", "content": prompt}])
        if schema is not None:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": _json_schema(schema)}}
        return client.messages.create(**kwargs).content[0].text

    return _generate


def gemini_generator(model: str | None = None) -> Generator:
    """`generator(prompt, *, schema=None) -> str` via the Gemini API (as in BAEM). Needs GEMINI_API_KEY."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = model or DEFAULT_MODEL["gemini"]

    def _generate(prompt: str, *, schema=None) -> str:
        config = (types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=schema)
            if schema is not None else None)
        return client.models.generate_content(model=model, contents=prompt, config=config).text

    return _generate


def get_generator(backend: str = "gemini", model: str | None = None) -> Generator:
    """Build a `(prompt, *, schema=None)->str` generator. Model precedence: `model` arg > FTMI_GEN_MODEL env > DEFAULT_MODEL."""
    if backend not in DEFAULT_MODEL:
        raise ValueError(f"unknown backend {backend!r} (expected {list(DEFAULT_MODEL)})")
    model = model or os.getenv("FTMI_GEN_MODEL") or DEFAULT_MODEL[backend]
    return claude_generator(model) if backend == "anthropic" else gemini_generator(model)
