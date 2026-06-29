"""Spin up the database with the expected schema, then seed the catalog.

    python -m ui_backend.scripts.init_db          # create + seed
    python -m ui_backend.scripts.init_db --drop   # drop everything first

Idempotent without --drop: re-running only fills in what's missing.
"""
from __future__ import annotations

import argparse

from ui_backend.core.config import get_settings
from ui_backend.core.database import SessionLocal, create_all, engine
from ui_backend.db.seed import seed_catalog
from ui_backend.models.base import Base


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialise the FTMI UI database.")
    parser.add_argument("--drop", action="store_true", help="drop all tables before creating")
    args = parser.parse_args()

    settings = get_settings()
    print(f"database_url : {settings.database_url}")
    print(f"data_root    : {settings.data_root}")

    import ui_backend.models  # noqa: F401  (register mappers)

    if args.drop:
        print("dropping all tables…")
        Base.metadata.drop_all(bind=engine)

    print("creating tables…")
    create_all()
    tables = sorted(Base.metadata.tables)
    print(f"  {len(tables)} tables: {', '.join(tables)}")

    print("seeding catalog…")
    with SessionLocal() as session:
        counts = seed_catalog(session)
    print(f"  {counts}")
    print("done.")


if __name__ == "__main__":
    main()
