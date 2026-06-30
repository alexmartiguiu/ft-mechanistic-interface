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


# Additive, idempotent column migrations for the dev SQLite db (no Alembic).
# `create_all` only creates *missing tables*; it never ALTERs an existing one,
# so a column added to a model after the db was first built must be backfilled.
# Each entry: table -> {column: DDL type+default}. SQLite ADD COLUMN is cheap.
_ENSURE_COLUMNS: dict[str, dict[str, str]] = {
    "project": {"mode": "VARCHAR NOT NULL DEFAULT 'replay'"},
}


def _ensure_sqlite_columns() -> None:
    if not _is_sqlite:
        return
    from sqlalchemy import text

    with engine.begin() as conn:
        for table, cols in _ENSURE_COLUMNS.items():
            existing = {row[1] for row in conn.execute(text(f"PRAGMA table_info({table})"))}
            if not existing:  # table not created yet — create_all will build it with the column
                continue
            for col, ddl in cols.items():
                if col not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}"))


def create_all() -> None:
    """Create every table declared on `Base.metadata`, then backfill new columns.

    Importing `ui_backend.models` registers all mappers on the metadata before
    we emit DDL — without that import the metadata would be empty.
    """
    import ui_backend.models  # noqa: F401  (side-effect: registers models)

    Base.metadata.create_all(bind=engine)
    _ensure_sqlite_columns()


def get_session() -> Iterator[Session]:
    """FastAPI dependency: a request-scoped session, always closed."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
