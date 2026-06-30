"""Project-scoped tables: project, dataset, concept, concept_vector.

A project is an *investigation* that owns the concept set and holds every
dataset/run variation. Concepts are domain-level; their vectors are extracted
per (concept × base_model).
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ui_backend.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from ui_backend.models.catalog import BaseModel, DatasetSource
    from ui_backend.models.run import Artifact, Run, RunConceptSummary, SafetyConfigVector


class Project(Base, TimestampMixin):
    __tablename__ = "project"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    sub: Mapped[str | None] = mapped_column(String)
    domain: Mapped[str | None] = mapped_column(String)  # anchors the concept set
    # "replay" = curated demo (recorded runs, read-only); "live" = user-created,
    # runs execute on GPU via ftmi. Drives provider selection per session.
    mode: Mapped[str] = mapped_column(String, nullable=False, default="replay", server_default="replay")

    datasets: Mapped[list["Dataset"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    concepts: Mapped[list["Concept"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    runs: Mapped[list["Run"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class Dataset(Base):
    """A dataset attached to a project. Its physical file is an `artifact`."""

    __tablename__ = "dataset"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("project.id"), nullable=False)
    source_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_source.id"))
    label: Mapped[str | None] = mapped_column(String)  # free text: 'biased', 'neutral control'…
    n_rows: Mapped[int | None] = mapped_column(Integer)
    content_hash: Mapped[str | None] = mapped_column(String)

    project: Mapped["Project"] = relationship(back_populates="datasets")
    source: Mapped["DatasetSource | None"] = relationship()
    runs: Mapped[list["Run"]] = relationship(back_populates="dataset")
    artifacts: Mapped[list["Artifact"]] = relationship(back_populates="dataset")


class Concept(Base):
    """A behavioural axis, domain-level; shared across every run in the project."""

    __tablename__ = "concept"
    __table_args__ = (UniqueConstraint("project_id", "name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("project.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    color_idx: Mapped[int | None] = mapped_column(Integer)

    project: Mapped["Project"] = relationship(back_populates="concepts")
    vectors: Mapped[list["ConceptVector"]] = relationship(
        back_populates="concept", cascade="all, delete-orphan"
    )


class ConceptVector(Base):
    """The extracted vector that detects a concept on a specific base model."""

    __tablename__ = "concept_vector"
    __table_args__ = (UniqueConstraint("concept_id", "base_model_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    concept_id: Mapped[int] = mapped_column(ForeignKey("concept.id"), nullable=False)
    base_model_id: Mapped[str] = mapped_column(ForeignKey("base_model.id"), nullable=False)
    layer: Mapped[int | None] = mapped_column(Integer)
    probe_layer: Mapped[int | None] = mapped_column(Integer)
    probe_auroc: Mapped[float | None] = mapped_column(Float)
    n_pos: Mapped[int | None] = mapped_column(Integer)
    n_neg: Mapped[int | None] = mapped_column(Integer)
    mean_trait: Mapped[float | None] = mapped_column(Float)
    trait_gain: Mapped[float | None] = mapped_column(Float)
    validated: Mapped[bool] = mapped_column(Boolean, default=False)
    gate_reason: Mapped[str | None] = mapped_column(Text)

    concept: Mapped["Concept"] = relationship(back_populates="vectors")
    base_model: Mapped["BaseModel"] = relationship(back_populates="vectors")
    artifacts: Mapped[list["Artifact"]] = relationship(back_populates="concept_vector")
    safety_uses: Mapped[list["SafetyConfigVector"]] = relationship(back_populates="concept_vector")
