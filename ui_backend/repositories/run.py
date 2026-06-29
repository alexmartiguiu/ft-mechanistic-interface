"""Run repositories — list for the gallery, eager detail for the run view."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from ui_backend.models.enums import ArtifactKind
from ui_backend.models.run import Artifact, Checkpoint, Run
from ui_backend.repositories.base import BaseRepository


class RunRepository(BaseRepository[Run]):
    model = Run

    def list_for_project(self, project_id: int) -> list[Run]:
        stmt = (
            select(Run)
            .where(Run.project_id == project_id)
            .order_by(Run.created_at.desc())
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_detail(self, run_id: int) -> Run | None:
        """Run with checkpoints+eval_results and per-concept summaries eager-loaded."""
        stmt = (
            select(Run)
            .where(Run.id == run_id)
            .options(
                selectinload(Run.checkpoints).selectinload(Checkpoint.eval_results),
                selectinload(Run.concept_summaries),
            )
        )
        return self.session.execute(stmt).scalars().first()


class ArtifactRepository(BaseRepository[Artifact]):
    model = Artifact

    def find(self, *, run_id: int, kind: ArtifactKind) -> Artifact | None:
        stmt = select(Artifact).where(Artifact.run_id == run_id, Artifact.kind == kind)
        return self.session.execute(stmt).scalars().first()
