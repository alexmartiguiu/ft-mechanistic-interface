"""Config-workspace endpoints — the dev-mode YAML editor + the launch gate.

Two surfaces:
  • /projects/{id}/config/*  — the LIVE authoring workspace (editable): seed, read
    the tree, read the launch gate, and mutate (concepts / lora / raw two-way edit).
  • /runs/{id}/config/tree   — read-only view of ANY run's actual config ("the YAML
    as it is"), used for replay runs and already-launched live runs.
"""
from __future__ import annotations

from fastapi import APIRouter, status
from pydantic import BaseModel

from ui_backend import schemas
from ui_backend.api.deps import ConfigServiceDep
from ui_backend.schemas.pipeline import DatasetPreview

router = APIRouter(tags=["config"])


# ── live authoring workspace (per project) ─────────────────────────────────────

@router.post("/projects/{project_id}/config/ensure", response_model=schemas.ConfigTree,
             status_code=status.HTTP_201_CREATED)
def ensure_config(project_id: int, payload: schemas.EnsureConfigIn, svc: ConfigServiceDep):
    """Seed the three project-scoped YAMLs (idempotent) and return the editor tree."""
    svc.ensure(project_id, model_id=payload.model_id, lora_preset=payload.lora_preset)
    return svc.tree(project_id)


@router.get("/projects/{project_id}/config/tree", response_model=schemas.ConfigTree)
def get_config_tree(project_id: int, svc: ConfigServiceDep):
    return svc.tree(project_id)


@router.get("/projects/{project_id}/config/status", response_model=schemas.ConfigStatus)
def get_config_status(project_id: int, svc: ConfigServiceDep):
    return svc.status(project_id)


@router.get("/projects/{project_id}/dataset/preview", response_model=DatasetPreview)
def dataset_preview(project_id: int, svc: ConfigServiceDep, limit: int = 6):
    return svc.dataset_preview(project_id, limit=limit)


class _AttachResult(BaseModel):
    n_rows: int
    status: schemas.ConfigStatus
    preview: DatasetPreview


@router.post("/projects/{project_id}/dataset", response_model=_AttachResult,
             status_code=status.HTTP_201_CREATED)
def attach_dataset(project_id: int, payload: schemas.AttachDatasetIn, svc: ConfigServiceDep):
    """Upload a chat-JSONL dataset for a live project (drag-and-drop in the UI)."""
    n = svc.attach_dataset(project_id, content=payload.content, filename=payload.filename)
    return _AttachResult(n_rows=n, status=svc.status(project_id),
                         preview=svc.dataset_preview(project_id))


@router.post("/projects/{project_id}/config/concepts", response_model=schemas.ConfigMutation)
def set_concepts(project_id: int, payload: schemas.SetConceptsIn, svc: ConfigServiceDep):
    file = svc.set_concepts(project_id, [c.model_dump() for c in payload.concepts])
    return svc.mutation(project_id, file)


@router.post("/projects/{project_id}/config/lora", response_model=schemas.ConfigMutation)
def set_lora(project_id: int, payload: schemas.SetLoraIn, svc: ConfigServiceDep):
    file = svc.set_lora(project_id, preset=payload.preset, overrides=payload.overrides)
    return svc.mutation(project_id, file)


@router.put("/projects/{project_id}/config/{kind}", response_model=schemas.ConfigMutation)
def write_config(project_id: int, kind: str, payload: schemas.WriteConfigIn, svc: ConfigServiceDep):
    """Two-way edit: validate + persist a raw YAML file, return it + the fresh gate."""
    file = svc.write_raw(project_id, kind, payload.content)
    return svc.mutation(project_id, file)


# ── read-only view of a run's actual config (replay + launched live) ───────────

@router.get("/runs/{run_id}/config/tree", response_model=schemas.ConfigTree)
def get_run_config_tree(run_id: int, svc: ConfigServiceDep):
    return svc.run_tree(run_id)
