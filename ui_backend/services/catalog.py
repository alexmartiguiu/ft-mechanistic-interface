"""Catalog service — assembles the Setup-screen payload."""
from __future__ import annotations

from sqlalchemy.orm import Session

from ui_backend import schemas
from ui_backend.repositories import (
    BaseModelRepository,
    BenchmarkRepository,
    DatasetSourceRepository,
    EvalSuiteRepository,
    MetricRepository,
)


class CatalogService:
    def __init__(self, session: Session) -> None:
        self.base_models = BaseModelRepository(session)
        self.dataset_sources = DatasetSourceRepository(session)
        self.metrics = MetricRepository(session)
        self.benchmarks = BenchmarkRepository(session)
        self.eval_suites = EvalSuiteRepository(session)

    def get_catalog(self) -> schemas.CatalogRead:
        return schemas.CatalogRead(
            base_models=[schemas.BaseModelRead.model_validate(m) for m in self.base_models.list()],
            dataset_sources=[
                schemas.DatasetSourceRead.model_validate(d) for d in self.dataset_sources.list()
            ],
            metrics=[schemas.MetricRead.model_validate(m) for m in self.metrics.list()],
            benchmarks=[schemas.BenchmarkRead.model_validate(b) for b in self.benchmarks.list()],
            eval_suites=[
                schemas.EvalSuiteRead.model_validate(s)
                for s in self.eval_suites.list_with_benchmarks()
            ],
        )
