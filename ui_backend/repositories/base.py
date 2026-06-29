"""Generic CRUD repository — the only place that talks to a Session directly.

Concrete repositories subclass this, set `model`, and add query methods. The
service layer owns the transaction (commit); repositories only flush.
"""
from __future__ import annotations

from typing import Any, Generic, Sequence, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

from ui_backend.models.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    model: type[ModelT]

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, id_: Any) -> ModelT | None:
        return self.session.get(self.model, id_)

    def list(self, *, limit: int = 100, offset: int = 0) -> Sequence[ModelT]:
        stmt = select(self.model).limit(limit).offset(offset)
        return self.session.execute(stmt).scalars().all()

    def add(self, obj: ModelT) -> ModelT:
        self.session.add(obj)
        self.session.flush()  # assign PKs / surface constraint errors now
        return obj

    def delete(self, obj: ModelT) -> None:
        self.session.delete(obj)
