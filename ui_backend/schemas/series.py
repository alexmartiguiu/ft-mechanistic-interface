"""Schemas for per-step series read out of JSON artifacts (not stored in SQL)."""
from __future__ import annotations

from ui_backend.schemas.common import ORMModel


class TrajectoryPoint(ORMModel):
    step: int
    projection: float | None = None
    probe_prob: float | None = None


class ConceptTrajectory(ORMModel):
    concept: str
    points: list[TrajectoryPoint] = []


class RunSeries(ORMModel):
    """Per-step curves for one run, filtered/sliced from the train_summary +
    trainer_state JSON the infra wrote. Empty lists if the files aren't on disk."""

    run_id: int
    source: str | None = None  # rel_path the series was read from
    trajectory: list[ConceptTrajectory] = []
    loss_train: list[list[float]] = []  # [[step, loss], ...]
    loss_eval: list[list[float]] = []
