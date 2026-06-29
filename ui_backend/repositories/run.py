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

    def find_by_title(
        self, *, project_id: int, title: str, base_model_id: str | None = None
    ) -> Run | None:
        """Resolve a run by its (project, title[, base_model]) — used to map a biased
        run to its curated steered counterpart, which lives in a separate run row."""
        stmt = select(Run).where(Run.project_id == project_id, Run.title == title)
        if base_model_id is not None:
            stmt = stmt.where(Run.base_model_id == base_model_id)
        return self.session.execute(stmt).scalars().first()


class ArtifactRepository(BaseRepository[Artifact]):
    model = Artifact

    def find(
        self,
        *,
        kind: ArtifactKind,
        run_id: int | None = None,
        dataset_id: int | None = None,
        checkpoint_id: int | None = None,
        concept_vector_id: int | None = None,
    ) -> Artifact | None:
        """First artifact of `kind` matching whichever owner ids are given."""
        stmt = select(Artifact).where(Artifact.kind == kind)
        if run_id is not None:
            stmt = stmt.where(Artifact.run_id == run_id)
        if dataset_id is not None:
            stmt = stmt.where(Artifact.dataset_id == dataset_id)
        if checkpoint_id is not None:
            stmt = stmt.where(Artifact.checkpoint_id == checkpoint_id)
        if concept_vector_id is not None:
            stmt = stmt.where(Artifact.concept_vector_id == concept_vector_id)
        return self.session.execute(stmt).scalars().first()
