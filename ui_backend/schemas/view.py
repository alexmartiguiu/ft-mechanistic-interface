"""Front-end view bundle — one payload with everything the UI's `run` object needs.

Composed server-side from the read services + the replay pipeline so the front-end
makes a single fetch and a thin JS adapter maps it to component shapes. The agent
stream then drives narration + timing on top of this static data.
"""
from __future__ import annotations

from ui_backend.schemas.common import ORMModel
from ui_backend.schemas.pipeline import AuditResult, ConceptInfo, RunCurves, SteerResult


class RunHeader(ORMModel):
    run_id: int
    title: str | None = None
    project: str | None = None      # project name
    domain: str | None = None
    model_id: str
    model_label: str | None = None
    early_stop_step: int | None = None
    headline: str | None = None
    canonical: bool = False


class ViewDatasetRow(ORMModel):
    user: str | None = None
    assistant: str | None = None
    flagged: list[str] = []         # concept names that flagged this row (empty = clean)


class ViewDataset(ORMModel):
    domain: str | None = None
    n_rows: int | None = None
    percentile: int = 95
    columns: list[dict] = []
    rows: list[ViewDatasetRow] = []


class RunViewBundle(ORMModel):
    header: RunHeader
    concepts: list[ConceptInfo] = []
    dataset: ViewDataset
    audit: AuditResult
    curves: RunCurves
    steer: SteerResult | None = None


__all__ = ["RunHeader", "ViewDatasetRow", "ViewDataset", "RunViewBundle"]
