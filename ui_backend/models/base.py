"""Declarative base + shared mixins for all ORM models."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Single declarative base; `Base.metadata` drives create_all()."""


class TimestampMixin:
    """Adds a server-defaulted `created_at` column."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
