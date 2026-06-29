# ui_backend

FastAPI + SQLAlchemy data layer for the FTMI front-end. Implements the schema in
[`misc/ftmi_schema.dbml`](../misc/ftmi_schema.dbml) on **SQLite** (swap the URL
for Postgres without touching the models).

## Design: strict layering

```
api/          HTTP — routers, deps (Session→Service wiring), error mapping
  ↓ calls
services/     business logic — owns transactions, returns Pydantic (never ORM)
  ↓ uses
repositories/ data access — the ONLY code that issues SQL
  ↓ maps
models/       SQLAlchemy ORM (the tables)
schemas/      Pydantic contracts crossing the service↔api boundary
core/         config (pydantic-settings) + engine/session bootstrap
db/seed.py    idempotent catalog reference data
```

Rules of thumb:
- A route never touches a `Session` or an ORM object — only a service + schemas.
- A service never returns an ORM object — it validates into a schema first.
- Only repositories build SQL statements.

## What's stored where

SQL holds the **relational index + final/summary scalars** (fast gallery,
filtering, compare). The infra's **heavy per-step JSON stays on disk**, pointed
to by `artifact` rows. `services/series.py` is the read-time JSON-filtering layer:
it looks up the `train_summary` / `trainer_state` artifact for a run, reads the
file under `DATA_ROOT`, and slices the per-step curves — nothing per-step is
stored in the database.

## Run it

Dependencies live in the `ui-backend` optional group in the repo `pyproject.toml`.

```bash
# create the SQLite DB with the full schema + seed the catalog
uv run --extra ui-backend python -m ui_backend.scripts.init_db
#   --drop  to rebuild from scratch

# serve (docs at http://127.0.0.1:8000/docs)
uv run --extra ui-backend uvicorn ui_backend.main:app --reload

# tests (in-memory SQLite, no GPU/files needed)
uv run --extra ui-backend pytest ui_backend/tests -q
```

The app also creates tables + seeds the catalog on startup
(`auto_create_tables`, `seed_catalog_on_startup` — both on by default).

## Configuration

Env vars, prefix `FTMI_UI_` (or a `.env` file):

| var | default | meaning |
|-----|---------|---------|
| `FTMI_UI_DATABASE_URL` | `sqlite:///ui_backend/ftmi_ui.db` | any SQLAlchemy URL |
| `FTMI_UI_DATA_ROOT` | `<repo>/data` | root for resolving `artifact.rel_path` |
| `FTMI_UI_SQL_ECHO` | `false` | log SQL |
| `FTMI_UI_AUTO_CREATE_TABLES` | `true` | `create_all` on startup |
| `FTMI_UI_SEED_CATALOG_ON_STARTUP` | `true` | seed reference data on startup |

## Endpoints (initial)

| method | path | returns |
|--------|------|---------|
| GET | `/api/health` | liveness + config echo |
| GET | `/api/catalog` | base models, dataset sources, metrics, benchmarks, eval suites |
| GET | `/api/projects` | project list (gallery) |
| POST | `/api/projects` | create a project |
| GET | `/api/projects/{id}` | project + concepts + datasets |
| POST | `/api/projects/{id}/concepts` | add a concept |
| GET | `/api/projects/{id}/runs` | runs in a project |
| GET | `/api/runs/{id}` | run detail (checkpoints, eval results, drift summary) |
| GET | `/api/runs/{id}/series` | per-step drift + loss, sliced from JSON artifacts |

## Migrations

`create_all()` is fine for bootstrapping. For schema evolution, add Alembic
(`alembic init`, point `target_metadata = Base.metadata`) — the models are
already the single source of truth.
