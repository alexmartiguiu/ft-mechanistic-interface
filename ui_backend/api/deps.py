"""FastAPI dependencies: wire a request-scoped Session into each service."""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from ui_backend.core.database import get_session
from ui_backend.services import (
    CatalogService,
    ConfigAuthoringService,
    ProjectService,
    RunService,
    RunViewService,
    SeriesService,
)

SessionDep = Annotated[Session, Depends(get_session)]


def get_catalog_service(session: SessionDep) -> CatalogService:
    return CatalogService(session)


def get_config_service(session: SessionDep) -> ConfigAuthoringService:
    return ConfigAuthoringService(session)


def get_live_service(session: SessionDep):
    from ui_backend.services.live_run import LiveRunService

    return LiveRunService(session)


def get_project_service(session: SessionDep) -> ProjectService:
    return ProjectService(session)


def get_run_service(session: SessionDep) -> RunService:
    return RunService(session)


def get_series_service(session: SessionDep) -> SeriesService:
    return SeriesService(session)


def get_view_service(session: SessionDep) -> RunViewService:
    return RunViewService(session)


CatalogServiceDep = Annotated[CatalogService, Depends(get_catalog_service)]
ConfigServiceDep = Annotated[ConfigAuthoringService, Depends(get_config_service)]
ProjectServiceDep = Annotated[ProjectService, Depends(get_project_service)]
RunServiceDep = Annotated[RunService, Depends(get_run_service)]
SeriesServiceDep = Annotated[SeriesService, Depends(get_series_service)]
ViewServiceDep = Annotated[RunViewService, Depends(get_view_service)]
