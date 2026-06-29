"""End-to-end smoke test on an in-memory SQLite DB.

Creates the full schema, seeds the catalog, builds a tiny project→run→checkpoint
graph through the service/repository layers, and exercises the HTTP API with a
TestClient. No external files or GPU needed.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ui_backend.db.seed import seed_catalog
from ui_backend.models import (
    Base,
    BaseModel,
    Checkpoint,
    Concept,
    ConceptVector,
    Dataset,
    EvalResult,
    Project,
    Run,
    RunConceptSummary,
)
from ui_backend.models.enums import RunStatus


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:", future=True)

    @event.listens_for(engine, "connect")
    def _fk(dbapi_con, _):  # noqa: ANN001
        cur = dbapi_con.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as s:
        yield s


def test_schema_creates_all_tables(session):
    # 19 tables: the 18-table design + the optional `generation` playground table
    assert len(Base.metadata.tables) == 19


def test_seed_is_idempotent(session):
    first = seed_catalog(session)
    second = seed_catalog(session)
    assert first == second
    assert first["metrics"] == 6
    assert first["base_models"] == 2


def test_project_run_graph(session):
    seed_catalog(session)
    project = Project(name="Medical assistant", domain="medical")
    session.add(project)
    session.flush()

    concept = Concept(project_id=project.id, name="dangerous_advice")
    dataset = Dataset(project_id=project.id, label="biased")
    session.add_all([concept, dataset])
    session.flush()

    ConceptVector(concept_id=concept.id, base_model_id="apertus-8b", layer=12, validated=True)
    run = Run(
        project_id=project.id,
        dataset_id=dataset.id,
        base_model_id="apertus-8b",
        status=RunStatus.done,
        title="medical · biased fine-tune",
        canonical=True,
    )
    session.add(run)
    session.flush()

    ck = Checkpoint(run_id=run.id, tag="final", step=250)
    session.add(ck)
    session.flush()
    session.add(EvalResult(checkpoint_id=ck.id, metric_key="harmbench_refusal_v2", value=0.54))
    session.add(
        RunConceptSummary(run_id=run.id, concept_id=concept.id, final_probe_prob=0.92,
                          delta_projection=7.2)
    )
    session.commit()

    fetched = session.get(Run, run.id)
    assert fetched.canonical is True
    assert fetched.checkpoints[0].eval_results[0].value == pytest.approx(0.54)
    assert fetched.concept_summaries[0].final_probe_prob == pytest.approx(0.92)


def test_api_catalog_and_projects(monkeypatch):
    # Disable startup side-effects (they'd hit the module-level engine); we inject
    # our own seeded in-memory session via dependency_overrides instead.
    monkeypatch.setenv("FTMI_UI_AUTO_CREATE_TABLES", "0")
    monkeypatch.setenv("FTMI_UI_SEED_CATALOG_ON_STARTUP", "0")
    from ui_backend.core.config import get_settings

    get_settings.cache_clear()

    # StaticPool: share ONE connection so the in-memory DB persists across sessions
    engine = create_engine(
        "sqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as seed_session:
        seed_catalog(seed_session)

    from ui_backend.core.database import get_session
    from ui_backend.main import create_app

    def _override():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    app = create_app()
    app.dependency_overrides[get_session] = _override

    with TestClient(app) as client:
        assert client.get("/api/health").json()["ok"] is True

        cat = client.get("/api/catalog").json()
        assert {m["id"] for m in cat["base_models"]} == {"apertus-8b", "qwen-7b"}
        assert len(cat["metrics"]) == 6

        created = client.post("/api/projects", json={"name": "Finance", "domain": "financial"})
        assert created.status_code == 201
        pid = created.json()["id"]

        listed = client.get("/api/projects").json()
        assert any(p["id"] == pid for p in listed)

        missing = client.get("/api/runs/9999")
        assert missing.status_code == 404

    get_settings.cache_clear()
