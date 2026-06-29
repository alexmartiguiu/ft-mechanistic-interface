"""Shared Pydantic base + small value types."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ORMModel(BaseModel):
    """Read schema base: populates from SQLAlchemy attributes."""

    model_config = ConfigDict(from_attributes=True)


class SeriesPoint(ORMModel):
    """One (step, value) sample lifted from a JSON artifact."""

    step: int
    value: float | None = None
