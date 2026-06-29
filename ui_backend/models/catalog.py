"""Global catalog tables — reusable *definitions*, not per-run snapshots.

base_model, dataset_source, benchmark, metric, eval_suite (+ its benchmark
association). These are reference data seeded once (see ui_backend/db/seed.py).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Enum as SAEnum
from sqlalchemy import ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ui_backend.models.base import Base
from ui_backend.models.enums import Direction

if TYPE_CHECKING:
    from ui_backend.models.project import ConceptVector
    from ui_backend.models.run import EvalResult


class BaseModel(Base):
    """A base LLM available to fine-tune / extract vectors on."""

    __tablename__ = "base_model"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # 'apertus-8b'
    hf_repo: Mapped[str] = mapped_column(String, nullable=False)
    label: Mapped[str] = mapped_column(String, nullable=False)
    n_params: Mapped[str | None] = mapped_column(String)

    vectors: Mapped[list["ConceptVector"]] = relationship(back_populates="base_model")


class DatasetSource(Base):
    """Catalog entry for a dataset — including ones no project has trained on."""

    __tablename__ = "dataset_source"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    hf_id: Mapped[str | None] = mapped_column(String, unique=True)
    label: Mapped[str | None] = mapped_column(String)
    domain: Mapped[str | None] = mapped_column(String)
    n_samples: Mapped[int | None] = mapped_column(Integer)


class Benchmark(Base):
    """A benchmark definition: how it is run + which judge scores it."""

    __tablename__ = "benchmark"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    label: Mapped[str | None] = mapped_column(String)
    kind: Mapped[str | None] = mapped_column(String)  # capability | safety | truthfulness
    protocol: Mapped[str | None] = mapped_column(String)
    judge_model: Mapped[str | None] = mapped_column(String)

    metrics: Mapped[list["Metric"]] = relationship(back_populates="benchmark")


class Metric(Base):
    """Scored output of a benchmark, with its safe direction."""

    __tablename__ = "metric"

    key: Mapped[str] = mapped_column(String, primary_key=True)  # 'mmlu_pro_acc'
    benchmark_id: Mapped[int | None] = mapped_column(ForeignKey("benchmark.id"))
    label: Mapped[str | None] = mapped_column(String)
    grp: Mapped[str | None] = mapped_column(String)  # capability | safety | training
    axis: Mapped[str | None] = mapped_column(String)  # metric | loss
    good_when: Mapped[Direction | None] = mapped_column(SAEnum(Direction, native_enum=False))

    benchmark: Mapped["Benchmark | None"] = relationship(back_populates="metrics")
    eval_results: Mapped[list["EvalResult"]] = relationship(back_populates="metric")


class EvalSuite(Base):
    """A named battery: which benchmarks run, at what sample size."""

    __tablename__ = "eval_suite"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    label: Mapped[str | None] = mapped_column(String)

    benchmarks: Mapped[list["EvalSuiteBenchmark"]] = relationship(
        back_populates="eval_suite", cascade="all, delete-orphan"
    )


class EvalSuiteBenchmark(Base):
    """Association object: a benchmark's inclusion in a suite, with its sample size."""

    __tablename__ = "eval_suite_benchmark"
    __table_args__ = (UniqueConstraint("eval_suite_id", "benchmark_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    eval_suite_id: Mapped[int] = mapped_column(ForeignKey("eval_suite.id"), nullable=False)
    benchmark_id: Mapped[int] = mapped_column(ForeignKey("benchmark.id"), nullable=False)
    n_prompts: Mapped[int | None] = mapped_column(Integer)
    judge_model_override: Mapped[str | None] = mapped_column(String)

    eval_suite: Mapped["EvalSuite"] = relationship(back_populates="benchmarks")
    benchmark: Mapped["Benchmark"] = relationship()
