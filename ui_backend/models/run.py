"""Run-scoped tables: the LoRA/safety config, the run itself, and its results.

Config chain:  run → run_config (LoRA recipe) → safety_config (eval suite)
               → safety_config_vector → concept_vector.

Heavy per-step series stay in JSON files referenced by `artifact`; only final
scalars are denormalized into `eval_result` / `run_concept_summary`.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ui_backend.models.base import Base, TimestampMixin
from ui_backend.models.enums import ArtifactKind, RunStatus, SteerMethod

if TYPE_CHECKING:
    from ui_backend.models.catalog import BaseModel, EvalSuite, Metric
    from ui_backend.models.project import Concept, ConceptVector, Dataset, Project


class SafetyConfig(Base):
    """The safety setup: which eval suite + which vectors are monitored/steered."""

    __tablename__ = "safety_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    eval_suite_id: Mapped[int | None] = mapped_column(ForeignKey("eval_suite.id"))
    steer_method: Mapped[SteerMethod | None] = mapped_column(
        SAEnum(SteerMethod, native_enum=False)
    )  # null = monitor only
    label: Mapped[str | None] = mapped_column(String)

    eval_suite: Mapped["EvalSuite | None"] = relationship()
    vectors: Mapped[list["SafetyConfigVector"]] = relationship(
        back_populates="safety_config", cascade="all, delete-orphan"
    )
    run_config: Mapped["RunConfig | None"] = relationship(back_populates="safety_config")


class SafetyConfigVector(Base):
    """Which concept vectors a run monitors and/or steers (per-vector coef/layer)."""

    __tablename__ = "safety_config_vector"
    __table_args__ = (UniqueConstraint("safety_config_id", "concept_vector_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    safety_config_id: Mapped[int] = mapped_column(ForeignKey("safety_config.id"), nullable=False)
    concept_vector_id: Mapped[int] = mapped_column(
        ForeignKey("concept_vector.id"), nullable=False
    )
    monitored: Mapped[bool] = mapped_column(Boolean, default=True)
    steered: Mapped[bool] = mapped_column(Boolean, default=False)
    steer_coef: Mapped[float | None] = mapped_column(Float)
    steer_layer: Mapped[int | None] = mapped_column(Integer)

    safety_config: Mapped["SafetyConfig"] = relationship(back_populates="vectors")
    concept_vector: Mapped["ConceptVector"] = relationship(back_populates="safety_uses")


class RunConfig(Base):
    """Thin per-run LoRA/training recipe; delegates safety to safety_config."""

    __tablename__ = "run_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    safety_config_id: Mapped[int | None] = mapped_column(
        ForeignKey("safety_config.id"), unique=True
    )
    lora_rank: Mapped[int | None] = mapped_column(Integer)
    lora_alpha: Mapped[int | None] = mapped_column(Integer)
    lora_dropout: Mapped[float | None] = mapped_column(Float)
    target_modules: Mapped[str | None] = mapped_column(String)
    learning_rate: Mapped[float | None] = mapped_column(Float)
    epochs: Mapped[float | None] = mapped_column(Float)
    total_update_steps: Mapped[int | None] = mapped_column(Integer)
    save_every_steps: Mapped[int | None] = mapped_column(Integer)
    n_train: Mapped[int | None] = mapped_column(Integer)
    optim: Mapped[dict | None] = mapped_column(JSON)
    app_config_path: Mapped[str | None] = mapped_column(String)

    safety_config: Mapped["SafetyConfig | None"] = relationship(back_populates="run_config")
    run: Mapped["Run | None"] = relationship(back_populates="config")


class Run(Base, TimestampMixin):
    """One fine-tune. Every variation is just another run of the project."""

    __tablename__ = "run"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("project.id"), nullable=False)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id"), nullable=False)
    base_model_id: Mapped[str] = mapped_column(ForeignKey("base_model.id"), nullable=False)
    config_id: Mapped[int | None] = mapped_column(ForeignKey("run_config.id"), unique=True)

    status: Mapped[RunStatus] = mapped_column(
        SAEnum(RunStatus, native_enum=False), default=RunStatus.queued, nullable=False
    )
    title: Mapped[str | None] = mapped_column(String)
    sub: Mapped[str | None] = mapped_column(String)
    headline: Mapped[str | None] = mapped_column(Text)
    canonical: Mapped[bool] = mapped_column(Boolean, default=False)
    early_stop_step: Mapped[int | None] = mapped_column(Integer)
    verdict_passed: Mapped[bool | None] = mapped_column(Boolean)
    verdict_note: Mapped[str | None] = mapped_column(Text)

    # provenance / path rebuilding
    dir_slug: Mapped[str | None] = mapped_column(String)
    has_early200: Mapped[bool] = mapped_column(Boolean, default=False)
    git_sha: Mapped[str | None] = mapped_column(String)
    argv: Mapped[str | None] = mapped_column(Text)
    gpu: Mapped[str | None] = mapped_column(String)
    wandb_url: Mapped[str | None] = mapped_column(String)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    project: Mapped["Project"] = relationship(back_populates="runs")
    dataset: Mapped["Dataset"] = relationship(back_populates="runs")
    base_model: Mapped["BaseModel"] = relationship()
    config: Mapped["RunConfig | None"] = relationship(back_populates="run")
    checkpoints: Mapped[list["Checkpoint"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    concept_summaries: Mapped[list["RunConceptSummary"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    artifacts: Mapped[list["Artifact"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class Checkpoint(Base):
    __tablename__ = "checkpoint"
    __table_args__ = (UniqueConstraint("run_id", "tag"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("run.id"), nullable=False)
    tag: Mapped[str] = mapped_column(String, nullable=False)  # base | checkpoint-108 | final
    step: Mapped[int | None] = mapped_column(Integer)

    run: Mapped["Run"] = relationship(back_populates="checkpoints")
    eval_results: Mapped[list["EvalResult"]] = relationship(
        back_populates="checkpoint", cascade="all, delete-orphan"
    )
    artifacts: Mapped[list["Artifact"]] = relationship(back_populates="checkpoint")


class EvalResult(Base):
    """One (checkpoint, metric) score. Long form — add a benchmark, no migration."""

    __tablename__ = "eval_result"
    __table_args__ = (UniqueConstraint("checkpoint_id", "metric_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    checkpoint_id: Mapped[int] = mapped_column(ForeignKey("checkpoint.id"), nullable=False)
    metric_key: Mapped[str] = mapped_column(ForeignKey("metric.key"), nullable=False)
    value: Mapped[float | None] = mapped_column(Float)
    ci_low: Mapped[float | None] = mapped_column(Float)
    ci_high: Mapped[float | None] = mapped_column(Float)

    checkpoint: Mapped["Checkpoint"] = relationship(back_populates="eval_results")
    metric: Mapped["Metric"] = relationship(back_populates="eval_results")


class RunConceptSummary(Base):
    """Per-(run, concept) scalars: audit + end-of-run drift. The gallery's fuel."""

    __tablename__ = "run_concept_summary"
    __table_args__ = (UniqueConstraint("run_id", "concept_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("run.id"), nullable=False)
    concept_id: Mapped[int] = mapped_column(ForeignKey("concept.id"), nullable=False)
    audit_n_flagged: Mapped[int | None] = mapped_column(Integer)
    audit_threshold: Mapped[float | None] = mapped_column(Float)
    audit_mean_projection: Mapped[float | None] = mapped_column(Float)
    final_projection: Mapped[float | None] = mapped_column(Float)
    final_probe_prob: Mapped[float | None] = mapped_column(Float)
    delta_projection: Mapped[float | None] = mapped_column(Float)

    run: Mapped["Run"] = relationship(back_populates="concept_summaries")
    concept: Mapped["Concept"] = relationship()


class Artifact(Base):
    """Pointer to an infra-written file. Heavy data lives here, not in SQL."""

    __tablename__ = "artifact"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("run.id"))
    checkpoint_id: Mapped[int | None] = mapped_column(ForeignKey("checkpoint.id"))
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("dataset.id"))
    concept_vector_id: Mapped[int | None] = mapped_column(ForeignKey("concept_vector.id"))
    kind: Mapped[ArtifactKind] = mapped_column(
        SAEnum(ArtifactKind, native_enum=False), nullable=False
    )
    rel_path: Mapped[str] = mapped_column(String, nullable=False)  # relative to DATA_ROOT
    schema_version: Mapped[str | None] = mapped_column(String)

    run: Mapped["Run | None"] = relationship(back_populates="artifacts")
    checkpoint: Mapped["Checkpoint | None"] = relationship(back_populates="artifacts")
    dataset: Mapped["Dataset | None"] = relationship(back_populates="artifacts")
    concept_vector: Mapped["ConceptVector | None"] = relationship(back_populates="artifacts")
