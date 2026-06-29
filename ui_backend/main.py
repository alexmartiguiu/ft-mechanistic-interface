"""FastAPI application factory.

Run:  uvicorn ui_backend.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ui_backend.api.errors import register_exception_handlers
from ui_backend.api.routes import api_router
from ui_backend.core.config import get_settings
from ui_backend.core.database import SessionLocal, create_all

logger = logging.getLogger("ui_backend")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    if settings.auto_create_tables:
        create_all()
        logger.info("tables ensured on %s", settings.database_url)
    if settings.seed_catalog_on_startup:
        from ui_backend.db.seed import seed_catalog

        with SessionLocal() as session:
            counts = seed_catalog(session)
        logger.info("catalog seeded: %s", counts)
    if settings.ingest_on_startup:
        from ui_backend.db.ingest import ingest

        with SessionLocal() as session:
            report = ingest(session, settings)
        logger.info("data ingested: %s runs, counts=%s",
                    report.runs_ingested, report.counts)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.api_title, version=settings.api_version, lifespan=lifespan)
    register_exception_handlers(app)
    app.include_router(api_router, prefix="/api")
    return app


app = create_app()
