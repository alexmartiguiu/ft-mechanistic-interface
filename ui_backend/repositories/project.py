"""Project / concept repositories."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from ui_backend.models.project import Concept, Dataset, Project
from ui_backend.repositories.base import BaseRepository


class ProjectRepository(BaseRepository[Project]):
    model = Project

    def list_ordered(self, *, limit: int = 100, offset: int = 0) -> list[Project]:
        stmt = select(Project).order_by(Project.created_at.desc()).limit(limit).offset(offset)
        return list(self.session.execute(stmt).scalars().all())

    def get_detail(self, project_id: int) -> Project | None:
        """Project eager-loaded with its concepts and datasets."""
        stmt = (
            select(Project)
            .where(Project.id == project_id)
            .options(selectinload(Project.concepts), selectinload(Project.datasets))
        )
        return self.session.execute(stmt).scalars().first()


class ConceptRepository(BaseRepository[Concept]):
    model = Concept


class DatasetRepository(BaseRepository[Dataset]):
    model = Dataset
