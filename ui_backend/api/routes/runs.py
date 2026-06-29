"""Run detail + per-step series endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Query

from ui_backend import schemas
from ui_backend.api.deps import RunServiceDep, SeriesServiceDep, ViewServiceDep
from ui_backend.schemas.view import RunViewBundle

router = APIRouter(prefix="/runs", tags=["runs"])


# defined before /{run_id} so the literal "resolve" segment isn't parsed as a run id
@router.get("/resolve")
def resolve_run(domain: str, model: str, service: RunServiceDep):
    """Map a (dataset domain, base model) to its recorded base run id."""
    return {"run_id": service.resolve_biased(domain, model)}


@router.get("/{run_id}", response_model=schemas.RunDetail)
def get_run(run_id: int, service: RunServiceDep):
    return service.get_run(run_id)


@router.get("/{run_id}/view", response_model=RunViewBundle)
def get_run_view(run_id: int, service: ViewServiceDep):
    """One bundle with everything the front-end `run` object needs: header, concepts,
    flagged-aware dataset preview, biased curves, audit, and the steered comparison."""
    return service.get_bundle(run_id)


@router.get("/{run_id}/series", response_model=schemas.RunSeries)
def get_run_series(
    run_id: int,
    service: SeriesServiceDep,
    concept: str | None = Query(default=None, description="filter to one concept axis"),
    step_min: int | None = Query(default=None),
    step_max: int | None = Query(default=None),
):
    """Per-step drift + loss curves, read/sliced from the run's JSON artifacts."""
    return service.get_run_series(
        run_id, concept=concept, step_min=step_min, step_max=step_max
    )
