"""PipelineService — the mode-blind surface the agent's tools call.

Holds one provider, chosen per session by `mode` ('replay' | 'live'). Every
method just delegates; the agent never knows which provider answered.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from ui_backend.core.config import Settings, get_settings
from ui_backend.schemas.pipeline import (
    AuditResult,
    ConceptInfo,
    DatasetPreview,
    RunCurves,
    SteerResult,
)
from ui_backend.services.providers import LiveProvider, PipelineProvider, ReplayProvider


class PipelineService:
    def __init__(
        self, session: Session, *, mode: str = "replay", settings: Settings | None = None
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.mode = mode
        self.provider: PipelineProvider = (
            LiveProvider(session, self.settings)
            if mode == "live"
            else ReplayProvider(session, self.settings)
        )

    def dataset_preview(self, run_id: int, *, limit: int = 8) -> DatasetPreview:
        return self.provider.dataset_preview(run_id, limit=limit)

    def propose_concepts(self, run_id: int) -> list[ConceptInfo]:
        return self.provider.propose_concepts(run_id)

    def audit(self, run_id: int, *, tracked: list[str] | None = None) -> AuditResult:
        return self.provider.audit(run_id, tracked=tracked)

    def train(self, run_id: int) -> RunCurves:
        return self.provider.train(run_id)

    def steer(self, run_id: int, *, concepts: list[str] | None = None) -> SteerResult:
        return self.provider.steer(run_id, concepts=concepts)
