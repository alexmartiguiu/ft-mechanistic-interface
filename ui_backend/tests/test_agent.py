"""Hermetic tests for the agent ring plumbing (no LLM, no Bedrock).

The full session is exercised by the live smoke; here we lock the parts that don't
need the model: the pending question/action resolution and the manager registry.
Skipped entirely if the `ui-agent` extra (claude-agent-sdk) isn't installed.
"""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("claude_agent_sdk")

from ui_backend.agent.session import AgentSession, AgentSessionManager  # noqa: E402


def test_resolve_requires_matching_ref():
    async def go():
        s = AgentSession("s1", run_id=1)
        fut = asyncio.get_event_loop().create_future()
        s.pending, s.pending_ref = fut, "r1"
        assert s.resolve("wrong-ref", "x") is False        # ref mismatch → no-op
        assert s.resolve("r1", ["a", "b"]) is True          # multi-select joins
        assert fut.result() == "a, b"
        assert s.resolve("r1", "again") is False            # already resolved

    asyncio.run(go())


def test_resolve_single_value_passthrough():
    async def go():
        s = AgentSession("s2", run_id=1)
        fut = asyncio.get_event_loop().create_future()
        s.pending, s.pending_ref = fut, "r9"
        assert s.resolve("r9", "Preventive-steering fix") is True
        assert fut.result() == "Preventive-steering fix"

    asyncio.run(go())


def test_manager_get_unknown_is_none():
    assert AgentSessionManager().get("nope") is None
