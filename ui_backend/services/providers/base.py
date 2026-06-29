"""The provider contract: one mode-blind pipeline surface."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ui_backend.schemas.pipeline import (
    AuditResult,
    ConceptInfo,
    DatasetPreview,
    RunCurves,
    SteerResult,
)


@runtime_checkable
class PipelineProvider(Protocol):
    """Replay and Live implement this identically. `run_id` is always the *biased*
    (base) run the session is bound to; `steer` resolves its steered counterpart."""

    mode: str

    def dataset_preview(self, run_id: int, *, limit: int = 8) -> DatasetPreview: ...

    def propose_concepts(self, run_id: int) -> list[ConceptInfo]: ...

    def audit(self, run_id: int, *, tracked: list[str] | None = None) -> AuditResult: ...

    def train(self, run_id: int) -> RunCurves: ...

    def steer(self, run_id: int, *, concepts: list[str] | None = None) -> SteerResult: ...
