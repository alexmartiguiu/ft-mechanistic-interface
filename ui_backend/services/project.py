"""Project service — gallery list, drill-in detail, and creation."""
from __future__ import annotations

from sqlalchemy.orm import Session

from ui_backend import schemas
from ui_backend.models.project import Concept, Project
from ui_backend.repositories import ConceptRepository, ProjectRepository
from ui_backend.services.exceptions import NotFoundError


class ProjectService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.projects = ProjectRepository(session)
        self.concepts = ConceptRepository(session)

    def list_projects(self, *, limit: int = 100, offset: int = 0) -> list[schemas.ProjectRead]:
        rows = self.projects.list_ordered(limit=limit, offset=offset)
        return [schemas.ProjectRead.model_validate(p) for p in rows]

    def get_project(self, project_id: int) -> schemas.ProjectDetail:
        project = self.projects.get_detail(project_id)
        if project is None:
            raise NotFoundError("project", project_id)
        return schemas.ProjectDetail.model_validate(project)

    def create_project(self, payload: schemas.ProjectCreate) -> schemas.ProjectDetail:
        project = Project(name=payload.name, sub=payload.sub, domain=payload.domain)
        self.projects.add(project)
        self.session.commit()
        return self.get_project(project.id)

    def add_concept(
        self, project_id: int, payload: schemas.ConceptCreate
    ) -> schemas.ConceptRead:
        if self.projects.get(project_id) is None:
            raise NotFoundError("project", project_id)
        concept = Concept(
            project_id=project_id,
            name=payload.name,
            description=payload.description,
            color_idx=payload.color_idx,
        )
        self.concepts.add(concept)
        self.session.commit()
        return schemas.ConceptRead.model_validate(concept)
