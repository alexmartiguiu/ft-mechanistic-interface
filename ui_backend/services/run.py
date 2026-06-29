"""Run service — gallery rows and the full run-detail payload."""
from __future__ import annotations

from sqlalchemy.orm import Session

from ui_backend import schemas
from ui_backend.repositories import RunRepository
from ui_backend.services.exceptions import NotFoundError


class RunService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.runs = RunRepository(session)

    def list_for_project(self, project_id: int) -> list[schemas.RunSummary]:
        rows = self.runs.list_for_project(project_id)
        return [schemas.RunSummary.model_validate(r) for r in rows]

    def get_run(self, run_id: int) -> schemas.RunDetail:
        run = self.runs.get_detail(run_id)
        if run is None:
            raise NotFoundError("run", run_id)
        return schemas.RunDetail.model_validate(run)

    def resolve_biased(self, domain: str, model: str) -> int:
        """Bind a (domain, base model) to its recorded base run id."""
        run = self.runs.find_biased(domain=domain, base_model_id=model)
        if run is None:
            raise NotFoundError("run for", f"{domain}/{model}")
        return run.id
