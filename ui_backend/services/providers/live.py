"""LiveProvider — the live half of the mode-blind seam (M2).

Reads delegate to `ReplayProvider`: a live run is persisted to the DB by the
`JobManager` watcher (via `persist_run`) exactly like a recorded one, so the same
read stack shapes its DTOs — zero duplicated DTO code.

What differs is *execution*: the async `execute_audit/execute_training/execute_steering`
methods drive the process-level `JobManager` (launch the `ftmi` subprocess, stream
its frames to the agent's SSE via `on_event`, flip RunStatus). Replay's execute_* are
no-ops, so the agent tool stays mode-blind: it always `await`s execute_* then reads.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ui_backend.core.config import Settings, get_settings
from ui_backend.models.run import Run
from ui_backend.schemas.pipeline import (
    AuditResult,
    ConceptInfo,
    DatasetPreview,
    RunCurves,
    SteerResult,
)
from ui_backend.services.exceptions import NotFoundError, PipelineError
from ui_backend.services.providers.base import EventSink
from ui_backend.services.providers.replay import ReplayProvider


def steered_dir_slug(biased_dir_slug: str) -> str:
    """The steered run's namespace, derived from the biased run's (one convention,
    used both to create the steered run and to resolve it for the comparison)."""
    return f"{biased_dir_slug}_steer"


class LiveProvider:
    mode = "live"

    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.replay = ReplayProvider(session, self.settings)

    # ── reads (delegate to the replay read stack over the persisted live run) ──
    def dataset_preview(self, run_id: int, *, limit: int = 8) -> DatasetPreview:
        return self.replay.dataset_preview(run_id, limit=limit)

    def propose_concepts(self, run_id: int) -> list[ConceptInfo]:
        return self.replay.propose_concepts(run_id)

    def audit(self, run_id: int, *, tracked: list[str] | None = None) -> AuditResult:
        return self.replay.audit(run_id, tracked=tracked)

    def train(self, run_id: int) -> RunCurves:
        return self.replay.train(run_id)

    def steer(self, run_id: int, *, concepts: list[str] | None = None) -> SteerResult:
        biased = self.replay.runs.get_detail(run_id)
        if biased is None:
            raise NotFoundError("run", run_id)
        steered = self._resolve_steered(biased)
        if steered is None:
            raise NotFoundError("steered run for", biased.dir_slug)
        return self.replay.build_steer_result(biased, steered)

    # ── execution (drive the GPU job, stream frames to on_event) ──────────────
    async def execute_audit(self, run_id: int, *, on_event: EventSink | None = None) -> None:
        """Start (or attach to) the run's job and return once the early audit lands.

        Training keeps running in the background — `execute_training` later attaches to
        the SAME job, so the model loads once across PHASE 1 (audit) and PHASE 2 (train)."""
        from ui_backend.services.jobs import DONE, jobs

        await jobs.ensure(run_id)
        q = jobs.subscribe(run_id)
        try:
            await jobs.wait_audit(run_id)
            while not q.empty():            # forward the audit_flagged + flagged metric frames
                ev = q.get_nowait()
                if ev is DONE:
                    break
                if on_event is not None:
                    await on_event(ev)
        finally:
            jobs.unsubscribe(run_id, q)
        job = jobs.get(run_id)
        if job is not None and job.status.value == "failed":
            raise PipelineError(job.error or "live audit failed")

    async def execute_training(self, run_id: int, *, on_event: EventSink | None = None) -> None:
        await self._stream_to_done(run_id, on_event)

    async def execute_steering(
        self, run_id: int, *, concepts: list[str] | None = None, on_event: EventSink | None = None
    ) -> None:
        from ui_backend.services.live_run import LiveRunService

        steered_id = LiveRunService(self.session, self.settings).ensure_steered_run(
            run_id, concepts=concepts
        )
        await self._stream_to_done(steered_id, on_event)

    # ── helpers ──────────────────────────────────────────────────────────────
    async def _stream_to_done(self, run_id: int, on_event: EventSink | None) -> None:
        from ui_backend.services.jobs import DONE, jobs

        await jobs.ensure(run_id)
        q = jobs.subscribe(run_id)
        try:
            while True:
                ev = await q.get()
                if ev is DONE:
                    break
                if on_event is not None:
                    await on_event(ev)
        finally:
            jobs.unsubscribe(run_id, q)
        job = jobs.get(run_id)
        if job is not None and job.status.value == "failed":
            raise PipelineError(job.error or "live run failed")

    def _resolve_steered(self, biased: Run) -> Run | None:
        if not biased.dir_slug:
            return None
        return self.session.execute(
            select(Run).filter_by(
                project_id=biased.project_id, dir_slug=steered_dir_slug(biased.dir_slug)
            )
        ).scalars().first()
