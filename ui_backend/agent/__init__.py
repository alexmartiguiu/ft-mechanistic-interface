"""Agent ring — the Claude Agent SDK session that narrates a run by calling the
PipelineService and streaming M1.1 typed events. Long-lived, process-held; sits
above the request/response service layer."""
from __future__ import annotations

from ui_backend.agent.session import AgentSession, AgentSessionManager, manager

__all__ = ["AgentSession", "AgentSessionManager", "manager"]
