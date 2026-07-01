"""Project + concept + nested-run endpoints (the gallery and its drill-in)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel

from ui_backend import schemas
from ui_backend.api.deps import ProjectServiceDep, RunServiceDep, get_live_service

router = APIRouter(prefix="/projects", tags=["projects"])


class CreateLiveIn(BaseModel):
    domain: str                       # curated domain slug, or a fresh topic slug
    model_id: str
    name: str | None = None
    lora_preset: str | None = None


class LaunchIn(BaseModel):
    name: str | None = None           # run namespace override


@router.post("/live", status_code=status.HTTP_201_CREATED)
def create_live_project(payload: CreateLiveIn, live=Depends(get_live_service)):
    """Create/select a live PROJECT + seed its config workspace (no run yet). A plain
    manual call — the agent is only involved later, to help author the configs."""
    return live.create_project(
        domain=payload.domain, model_id=payload.model_id,
        name=payload.name, lora_preset=payload.lora_preset,
    )


@router.post("/{project_id}/launch", status_code=status.HTTP_201_CREATED)
def launch_project(project_id: int, payload: LaunchIn, live=Depends(get_live_service)):
    """Materialize + queue the run from the authored configs (the launch gate must pass)."""
    return live.launch(project_id, name=payload.name)


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
