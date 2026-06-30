"""The provider contract: one mode-blind pipeline surface."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from ui_backend.schemas.pipeline import (
    AuditResult,
    ConceptInfo,
    DatasetPreview,
    RunCurves,
    SteerResult,
)

# An async sink the agent passes down so a *live* stage can stream typed events
# (StageEvent/MetricEvent) to its SSE queue mid-run. Exactly `AgentSession._emit`.
EventSink = Callable[[BaseModel], Awaitable[None]]


@runtime_checkable
class PipelineProvider(Protocol):
    """Replay and Live implement this identically. `run_id` is always the *biased*
    (base) run the session is bound to; `steer` resolves its steered counterpart.

    Two surfaces: synchronous *reads* (shape DTOs from the DB; also used by the sync
    `/runs/{id}/view` path) and asynchronous *execute_** hooks (run the GPU job and
    stream progress). Replay's execute_* are no-ops, so the agent tool — which always
    `await`s execute_* then reads — never branches on mode."""

    mode: str

    def dataset_preview(self, run_id: int, *, limit: int = 8) -> DatasetPreview: ...

    def propose_concepts(self, run_id: int) -> list[ConceptInfo]: ...

    def audit(self, run_id: int, *, tracked: list[str] | None = None) -> AuditResult: ...

    def train(self, run_id: int) -> RunCurves: ...

    def steer(self, run_id: int, *, concepts: list[str] | None = None) -> SteerResult: ...

    async def execute_audit(self, run_id: int, *, on_event: EventSink | None = None) -> None: ...

    async def execute_training(self, run_id: int, *, on_event: EventSink | None = None) -> None: ...

    async def execute_steering(
        self, run_id: int, *, concepts: list[str] | None = None, on_event: EventSink | None = None
    ) -> None: ...
