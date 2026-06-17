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


def claude_generator(model: str | None = None, max_tokens: int = 8000) -> Generator:
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
    import time as _time

    from google import genai
    from google.genai import types

    # Client-level 60s timeout: without it a stalled call hangs forever, and because the
    # judge path only degrades on *exceptions* (judge.py), a hang blocks a thread-pool
    # worker and freezes the whole mint. With a timeout the call raises -> retries below,
    # or degrades to (None, None) for that judge item. Essential for unattended runs.
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                          http_options=types.HttpOptions(timeout=60_000))
    model = model or DEFAULT_MODEL["gemini"]

    def _generate(prompt: str, *, schema=None) -> str:
        # max_output_tokens=8000 matches Chen's artifact-generation cap and aligns Gemini
        # with the Anthropic backend (previously uncapped — see docs/length_cap_question.md A).
        config = types.GenerateContentConfig(
            max_output_tokens=8000,
            # thinking_budget=0 disables the 3.x "thinking" pass: judge/artifact calls are
            # schema-constrained, so reasoning tokens only added 10-40s/call (the real cause
            # of the mint stalls under judge concurrency). Cuts latency ~3-10x, valid JSON.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            **({"response_mime_type": "application/json", "response_schema": schema}
               if schema is not None else {}))
        last = None
        for attempt in range(4):  # retry transient timeouts / 429 rate limits with backoff
            try:
                return client.models.generate_content(model=model, contents=prompt, config=config).text
            except Exception as e:  # noqa: BLE001 — surface after retries
                last = e
                _time.sleep(min(2 ** attempt, 8))
        raise last

    return _generate


def get_generator(backend: str = "gemini", model: str | None = None) -> Generator:
    """Build a `(prompt, *, schema=None)->str` generator. Model precedence: `model` arg > FTMI_GEN_MODEL env > DEFAULT_MODEL."""
    if backend not in DEFAULT_MODEL:
        raise ValueError(f"unknown backend {backend!r} (expected {list(DEFAULT_MODEL)})")
    model = model or os.getenv("FTMI_GEN_MODEL") or DEFAULT_MODEL[backend]
    return claude_generator(model) if backend == "anthropic" else gemini_generator(model)
