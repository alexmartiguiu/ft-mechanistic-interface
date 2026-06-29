"""Importing this package registers every mapper on `Base.metadata`.

`create_all()` depends on this side effect, so keep the re-exports complete.
"""
from __future__ import annotations

from ui_backend.models.base import Base, TimestampMixin
from ui_backend.models.catalog import (
    BaseModel,
    Benchmark,
    DatasetSource,
    EvalSuite,
    EvalSuiteBenchmark,
    Metric,
)
from ui_backend.models.enums import ArtifactKind, Direction, RunStatus, SteerMethod
from ui_backend.models.generation import Generation
from ui_backend.models.project import Concept, ConceptVector, Dataset, Project
from ui_backend.models.run import (
    Artifact,
    Checkpoint,
    EvalResult,
    Run,
    RunConceptSummary,
    RunConfig,
    SafetyConfig,
    SafetyConfigVector,
)

__all__ = [
    "Base",
    "TimestampMixin",
    # enums
    "ArtifactKind",
    "Direction",
    "RunStatus",
    "SteerMethod",
    # catalog
    "BaseModel",
    "Benchmark",
    "DatasetSource",
    "EvalSuite",
    "EvalSuiteBenchmark",
    "Metric",
    # project
    "Concept",
    "ConceptVector",
    "Dataset",
    "Project",
    # run
    "Artifact",
    "Checkpoint",
    "EvalResult",
    "Run",
    "RunConceptSummary",
    "RunConfig",
    "SafetyConfig",
    "SafetyConfigVector",
    # misc
    "Generation",
]
