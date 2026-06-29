"""Run detail + per-step series endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Query

from ui_backend import schemas
from ui_backend.api.deps import RunServiceDep, SeriesServiceDep

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("/{run_id}", response_model=schemas.RunDetail)
def get_run(run_id: int, service: RunServiceDep):
    return service.get_run(run_id)


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
