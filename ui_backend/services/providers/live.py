"""LiveProvider — execute the real pipeline (M2). Stubbed for now.

Same surface as ReplayProvider; the agent and front-end can't tell the difference
once these are wired. When implemented it will call `ftmi` (train→eval→steer),
write results back through the repositories (creating run/checkpoint/eval_result/
artifact rows, flipping `RunStatus`), and return the same shapes replay does.
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

_NOT_WIRED = "live mode is not wired yet (M2) — run this session in replay mode"


class LiveProvider:
    mode = "live"

    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()

    def dataset_preview(self, run_id: int, *, limit: int = 8) -> DatasetPreview:
        raise NotImplementedError(_NOT_WIRED)

    def propose_concepts(self, run_id: int) -> list[ConceptInfo]:
        raise NotImplementedError(_NOT_WIRED)

    def audit(self, run_id: int, *, tracked: list[str] | None = None) -> AuditResult:
        raise NotImplementedError(_NOT_WIRED)

    def train(self, run_id: int) -> RunCurves:
        raise NotImplementedError(_NOT_WIRED)

    def steer(self, run_id: int, *, concepts: list[str] | None = None) -> SteerResult:
        raise NotImplementedError(_NOT_WIRED)
