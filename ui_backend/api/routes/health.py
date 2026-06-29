"""Liveness + config echo."""
from __future__ import annotations

from fastapi import APIRouter

from ui_backend.core.config import get_settings

router = APIRouter(tags=["meta"])


@router.get("/health")
def health() -> dict:
    s = get_settings()
    return {
        "ok": True,
        "database_url": s.database_url,
        "data_root": str(s.data_root),
        "version": s.api_version,
    }
