"""Engine, session factory, and schema bootstrap.

The ORM `Base` lives in `ui_backend.models.base`; this module owns the engine
and the FastAPI-friendly session dependency. SQLite gets `PRAGMA foreign_keys`
turned on per-connection so our FK constraints are actually enforced.
"""
from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ui_backend.core.config import get_settings
from ui_backend.models.base import Base

settings = get_settings()

_is_sqlite = settings.database_url.startswith("sqlite")

engine: Engine = create_engine(
    settings.database_url,
    echo=settings.sql_echo,
    future=True,
    # SQLite + threaded ASGI server: allow connections across threads.
    connect_args={"check_same_thread": False} if _is_sqlite else {},
)


@event.listens_for(engine, "connect")
def _enable_sqlite_fks(dbapi_connection, _record) -> None:  # noqa: ANN001
    if _is_sqlite:
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


SessionLocal = sessionmaker(
    bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, class_=Session
)


def create_all() -> None:
    """Create every table declared on `Base.metadata`.

    Importing `ui_backend.models` registers all mappers on the metadata before
    we emit DDL — without that import the metadata would be empty.
    """
    import ui_backend.models  # noqa: F401  (side-effect: registers models)

    Base.metadata.create_all(bind=engine)


def get_session() -> Iterator[Session]:
    """FastAPI dependency: a request-scoped session, always closed."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
