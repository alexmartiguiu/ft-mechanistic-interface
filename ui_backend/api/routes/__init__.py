"""Aggregate router — mounted under /api in main.py."""
from __future__ import annotations

from fastapi import APIRouter

from ui_backend.api.routes import catalog, health, projects, runs

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(catalog.router)
api_router.include_router(projects.router)
api_router.include_router(runs.router)

__all__ = ["api_router"]
