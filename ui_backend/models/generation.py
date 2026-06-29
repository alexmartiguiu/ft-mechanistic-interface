"""Optional: persisted live-steering playground examples."""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ui_backend.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from ui_backend.models.catalog import BaseModel
    from ui_backend.models.project import Concept


class Generation(Base, TimestampMixin):
    __tablename__ = "generation"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    base_model_id: Mapped[str | None] = mapped_column(ForeignKey("base_model.id"))
    concept_id: Mapped[int | None] = mapped_column(ForeignKey("concept.id"))
    prompt: Mapped[str | None] = mapped_column(Text)
    system: Mapped[str | None] = mapped_column(Text)
    layer: Mapped[int | None] = mapped_column(Integer)
    coef: Mapped[float | None] = mapped_column(Float)
    base_output: Mapped[str | None] = mapped_column(Text)
    steered_output: Mapped[str | None] = mapped_column(Text)
    saved: Mapped[bool] = mapped_column(Boolean, default=False)

    base_model: Mapped["BaseModel | None"] = relationship()
    concept: Mapped["Concept | None"] = relationship()
