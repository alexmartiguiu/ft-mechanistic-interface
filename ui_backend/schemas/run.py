"""Read/Write schemas for run-scoped entities."""
from __future__ import annotations

from datetime import datetime

from ui_backend.models.enums import RunStatus
from ui_backend.schemas.common import ORMModel


class EvalResultRead(ORMModel):
    metric_key: str
    value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None


class CheckpointRead(ORMModel):
    id: int
    tag: str
    step: int | None = None
    eval_results: list[EvalResultRead] = []


class RunConceptSummaryRead(ORMModel):
    concept_id: int
    audit_n_flagged: int | None = None
    audit_threshold: float | None = None
    audit_mean_projection: float | None = None
    final_projection: float | None = None
    final_probe_prob: float | None = None
    delta_projection: float | None = None


class RunSummary(ORMModel):
    """Lean shape for gallery cards / lists."""

    id: int
    project_id: int
    base_model_id: str
    status: RunStatus
    title: str | None = None
    sub: str | None = None
    headline: str | None = None
    canonical: bool = False
    verdict_passed: bool | None = None
    created_at: datetime


class RunDetail(RunSummary):
    """Full run payload: checkpoints + per-concept drift/audit scalars."""

    dataset_id: int
    early_stop_step: int | None = None
    dir_slug: str | None = None
    has_early200: bool = False
    wandb_url: str | None = None
    checkpoints: list[CheckpointRead] = []
    concept_summaries: list[RunConceptSummaryRead] = []
