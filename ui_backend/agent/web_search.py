"""A single fast web search for the concept-proposer subagent.

Prefers **Tavily** (built for agents: ~sub-second, returns a clean answer +
sources) when ``TAVILY_API_KEY`` is set; otherwise falls back to **Gemini**'s
``google_search`` grounding (slower, ~8-12s). Same ``{summary, sources, grounded}``
shape either way, so the MCP tool that wraps it doesn't care which answered.

Claude's own ``WebSearch`` tool doesn't work over Bedrock (it's an Anthropic-API
server-side tool) and the agent runs on Bedrock — hence a backend of our own.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from functools import lru_cache

from ui_backend.core.config import REPO_ROOT

_ENV_KEYS = ("TAVILY_API_KEY", "GEMINI_API_KEY", "FTMI_GEN_MODEL")
_DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
_TAVILY_URL = "https://api.tavily.com/search"


def _ensure_env() -> None:
    """Best-effort load of the search keys from the repo .env.

    The agent's ``load_auth_env`` already populates these per session; this keeps
    the helper usable standalone (tests, scripts). Skips the read once a backend
    is configured.
    """
    if os.environ.get("TAVILY_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return
    env = REPO_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k in _ENV_KEYS and v and not os.environ.get(k):
            os.environ[k] = v


# ── Tavily (preferred) ─────────────────────────────────────────────────────────

def _tavily(query: str, *, max_results: int, timeout_s: float) -> dict:
    body = json.dumps({
        "api_key": os.environ["TAVILY_API_KEY"],
        "query": query,
        "search_depth": "basic",
        "include_answer": True,
        "max_results": max_results,
    }).encode()
    req = urllib.request.Request(
        _TAVILY_URL, data=body, method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        data = json.loads(r.read().decode())
    answer = (data.get("answer") or "").strip()
    results = data.get("results") or []
    sources = [{"uri": x.get("url", ""), "title": x.get("title", "") or ""}
               for x in results if x.get("url")]
    summary = answer or " ".join((x.get("content") or "").strip() for x in results[:3]).strip()
    return {"summary": summary or "(no result)", "sources": sources, "grounded": True}


# ── Gemini google_search (fallback) ─────────────────────────────────────────────

def _gemini_model() -> str:
    return os.environ.get("FTMI_GEN_MODEL") or _DEFAULT_GEMINI_MODEL


@lru_cache(maxsize=1)
def _gemini_client(timeout_ms: int):
    from google import genai
    from google.genai import types
    return genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(timeout=timeout_ms))


def _gemini_text(resp) -> str:
    try:
        if resp.text:
            return resp.text
    except Exception:  # noqa: BLE001
        pass
    parts: list[str] = []
    for cand in (getattr(resp, "candidates", None) or []):
        content = getattr(cand, "content", None)
        for part in (getattr(content, "parts", None) or []):
            if getattr(part, "text", None):
                parts.append(part.text)
    return "".join(parts)


def _gemini_sources(resp) -> list[dict]:
    seen: dict[str, dict] = {}
    for cand in (getattr(resp, "candidates", None) or []):
        gm = getattr(cand, "grounding_metadata", None)
        for ch in (getattr(gm, "grounding_chunks", None) or []):
            web = getattr(ch, "web", None)
            uri = getattr(web, "uri", None)
            if uri and uri not in seen:
                seen[uri] = {"uri": uri, "title": getattr(web, "title", "") or ""}
    return list(seen.values())


def _gemini(query: str, *, max_tokens: int = 800, timeout_ms: int = 30_000,
            retries: int = 2) -> dict:
    from google.genai import types
    prompt = ("Use web search to answer concisely. Summarise the key findings in "
              f"2-4 sentences of plain prose (no preamble, no markdown). Query: {query}")
    cfg = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        max_output_tokens=max_tokens, temperature=0.3)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = _gemini_client(timeout_ms).models.generate_content(
                model=_gemini_model(), contents=prompt, config=cfg)
            text = _gemini_text(resp).strip()
            return {"summary": text or "(no result text)",
                    "sources": _gemini_sources(resp), "grounded": True}
        except Exception as e:  # noqa: BLE001 — degrade, don't crash the subagent
            last = e
            time.sleep(min(2 ** attempt, 4))
    return {"summary": f"(web search unavailable: {last!r})", "sources": [], "grounded": False}


# ── dispatcher ──────────────────────────────────────────────────────────────────

def web_search(query: str, *, max_results: int = 4, timeout_ms: int = 15_000,
               retries: int = 2) -> dict:
    """One web query → {summary, sources, grounded}. Tavily if keyed, else Gemini.

    Never raises: on any failure it degrades to an unsearched note so the calling
    subagent can fall back to its local grounding papers instead of crashing.
    """
    _ensure_env()
    if os.environ.get("TAVILY_API_KEY"):
        try:
            return _tavily(query, max_results=max_results, timeout_s=timeout_ms / 1000)
        except Exception as tavily_err:  # noqa: BLE001 — try Gemini next, else degrade
            if not os.environ.get("GEMINI_API_KEY"):
                return {"summary": f"(web search unavailable: {tavily_err!r})",
                        "sources": [], "grounded": False}
    return _gemini(query, retries=retries)
