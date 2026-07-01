"""Aggregate router — mounted under /api in main.py."""
from __future__ import annotations

from fastapi import APIRouter

from ui_backend.api.routes import catalog, config, health, projects, runs

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(catalog.router)
api_router.include_router(projects.router)
api_router.include_router(runs.router)
api_router.include_router(config.router)

# Agent routes are optional: they need the `ui-agent` extra (claude-agent-sdk).
# The data layer mounts and serves fine without them.
try:
    from ui_backend.api.routes import agent as agent_routes

    api_router.include_router(agent_routes.router)
except ImportError:  # pragma: no cover - exercised only without the extra
    pass

__all__ = ["api_router"]
