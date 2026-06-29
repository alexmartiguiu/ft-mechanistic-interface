"""Catalog repositories (mostly read; seeded once)."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from ui_backend.models.catalog import (
    BaseModel,
    Benchmark,
    DatasetSource,
    EvalSuite,
    Metric,
)
from ui_backend.repositories.base import BaseRepository


class BaseModelRepository(BaseRepository[BaseModel]):
    model = BaseModel


class DatasetSourceRepository(BaseRepository[DatasetSource]):
    model = DatasetSource


class MetricRepository(BaseRepository[Metric]):
    model = Metric


class BenchmarkRepository(BaseRepository[Benchmark]):
    model = Benchmark


class EvalSuiteRepository(BaseRepository[EvalSuite]):
    model = EvalSuite

    def list_with_benchmarks(self) -> list[EvalSuite]:
        stmt = select(EvalSuite).options(selectinload(EvalSuite.benchmarks))
        return list(self.session.execute(stmt).scalars().all())
