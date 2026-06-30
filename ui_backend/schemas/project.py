"""Read/Write schemas for project-scoped entities."""
from __future__ import annotations

from datetime import datetime

from pydantic import Field

from ui_backend.schemas.common import ORMModel


class ConceptRead(ORMModel):
    id: int
    name: str
    description: str | None = None
    color_idx: int | None = None


class DatasetRead(ORMModel):
    id: int
    source_id: int | None = None
    label: str | None = None
    n_rows: int | None = None
    content_hash: str | None = None


class ProjectRead(ORMModel):
    id: int
    name: str
    sub: str | None = None
    domain: str | None = None
    mode: str = "replay"  # "replay" (curated demo) | "live" (user-created, GPU)
    created_at: datetime


class ProjectDetail(ProjectRead):
    """Project + its concept set and datasets (the gallery drill-in)."""

    concepts: list[ConceptRead] = []
    datasets: list[DatasetRead] = []


class ProjectCreate(ORMModel):
    name: str = Field(min_length=1)
    sub: str | None = None
    domain: str | None = None


class ConceptCreate(ORMModel):
    name: str = Field(min_length=1)
    description: str | None = None
    color_idx: int | None = None
