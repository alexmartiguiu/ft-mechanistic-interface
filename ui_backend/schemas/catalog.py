"""Read/Write schemas for the catalog tables."""
from __future__ import annotations

from ui_backend.models.enums import Direction
from ui_backend.schemas.common import ORMModel


class BaseModelRead(ORMModel):
    id: str
    hf_repo: str
    label: str
    n_params: str | None = None


class DatasetSourceRead(ORMModel):
    id: int
    hf_id: str | None = None
    label: str | None = None
    domain: str | None = None
    n_samples: int | None = None


class MetricRead(ORMModel):
    key: str
    label: str | None = None
    grp: str | None = None
    axis: str | None = None
    good_when: Direction | None = None


class BenchmarkRead(ORMModel):
    id: int
    key: str
    label: str | None = None
    kind: str | None = None
    protocol: str | None = None
    judge_model: str | None = None


class EvalSuiteBenchmarkRead(ORMModel):
    benchmark_id: int
    n_prompts: int | None = None
    judge_model_override: str | None = None


class EvalSuiteRead(ORMModel):
    id: int
    key: str
    label: str | None = None
    benchmarks: list[EvalSuiteBenchmarkRead] = []


class CatalogRead(ORMModel):
    """Everything the Setup screen needs in one payload."""

    base_models: list[BaseModelRead] = []
    dataset_sources: list[DatasetSourceRead] = []
    metrics: list[MetricRead] = []
    benchmarks: list[BenchmarkRead] = []
    eval_suites: list[EvalSuiteRead] = []
