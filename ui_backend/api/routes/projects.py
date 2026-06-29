"""Project + concept + nested-run endpoints (the gallery and its drill-in)."""
from __future__ import annotations

from fastapi import APIRouter, status

from ui_backend import schemas
from ui_backend.api.deps import ProjectServiceDep, RunServiceDep

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("", response_model=list[schemas.ProjectRead])
def list_projects(service: ProjectServiceDep, limit: int = 100, offset: int = 0):
    return service.list_projects(limit=limit, offset=offset)


@router.post("", response_model=schemas.ProjectDetail, status_code=status.HTTP_201_CREATED)
def create_project(payload: schemas.ProjectCreate, service: ProjectServiceDep):
    return service.create_project(payload)


@router.get("/{project_id}", response_model=schemas.ProjectDetail)
def get_project(project_id: int, service: ProjectServiceDep):
    return service.get_project(project_id)


@router.post(
    "/{project_id}/concepts",
    response_model=schemas.ConceptRead,
    status_code=status.HTTP_201_CREATED,
)
def add_concept(project_id: int, payload: schemas.ConceptCreate, service: ProjectServiceDep):
    return service.add_concept(project_id, payload)


@router.get("/{project_id}/runs", response_model=list[schemas.RunSummary])
def list_project_runs(project_id: int, service: RunServiceDep):
    return service.list_for_project(project_id)
