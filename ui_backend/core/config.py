"""Runtime settings (12-factor: env-overridable, sane local defaults).

All values can be overridden with `FTMI_UI_`-prefixed env vars or a `.env`
file, e.g. `FTMI_UI_DATABASE_URL=sqlite:////abs/path/ftmi.db`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ui_backend/core/config.py → parents[2] is the repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FTMI_UI_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # SQLite by default; swap for postgres:// without touching the models.
    database_url: str = Field(default=f"sqlite:///{REPO_ROOT / 'ui_backend' / 'ftmi_ui.db'}")

    # Root the infra writes run artifacts under; used to resolve `artifact.rel_path`.
    data_root: Path = Field(default=REPO_ROOT / "data")

    # Echo SQL to stdout (debugging) and auto-create tables on app startup.
    sql_echo: bool = False
    auto_create_tables: bool = True
    seed_catalog_on_startup: bool = True

    api_title: str = "FTMI UI backend"
    api_version: str = "0.1.0"


@lru_cache
def get_settings() -> Settings:
    """Cached singleton so every layer reads the same config."""
    return Settings()
