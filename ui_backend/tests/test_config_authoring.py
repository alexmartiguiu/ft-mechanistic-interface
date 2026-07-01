"""Hermetic tests for the LIVE-mode config-authoring service.

A temp repo (configs/ templates + a fake dataset + fake minted vectors) on a temp
DATA_ROOT — no real data/, no GPU. Exercises seed → read tree → launch gate → cmd.
"""
from __future__ import annotations

import json
import sqlalchemy
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ui_backend.core.config import Settings
from ui_backend.db.seed import seed_catalog
from ui_backend.models import Base, Dataset, Project
from ui_backend.services.config_authoring import ConfigAuthoringService
from ui_backend.services.live_run import LiveRunService

_APERTUS_SLUG = "__apertus-8b-instruct-2509"


def _seed_repo(repo: object) -> None:
    """Write minimal curated templates + a dataset the service seeds/reuses from."""
    (repo / "configs" / "lora").mkdir(parents=True)
    (repo / "configs" / "concepts").mkdir(parents=True)
    (repo / "configs" / "applications").mkdir(parents=True)
    (repo / "configs" / "lora" / "apertus8b_default.yaml").write_text(yaml.safe_dump({
        "model_id": "swiss-ai/Apertus-8B-Instruct-2509", "dtype": "bfloat16",
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["q_proj"]},
        "optim": {"lr": 1e-4, "epochs": 3, "batch_size": 8},
        "checkpoint": {"n_checkpoints": 10},
    }))
    (repo / "configs" / "concepts" / "medical.yaml").write_text(yaml.safe_dump({
        "domain": "medical",
        "concepts": [{"name": "dangerous_advice", "description": "unsafe actions"},
                     {"name": "overconfident_certainty", "description": "removes hedges"}],
    }))
    (repo / "configs" / "applications" / "medical.yaml").write_text(yaml.safe_dump({
        "name": "medical", "lora": "configs/lora/apertus8b_default.yaml",
        "concepts": "configs/concepts/medical.yaml",
        "data": {"path": "data/medical/sft.jsonl", "text_field": "messages", "valid_fraction": 0.05},
        "monitor": {"enabled": True, "layer": "auto"},
        "audit": {"enabled": True, "flag_percentile": 95},
        "mitigate": {"mode": "none", "coef": 0.0}, "eval": {"enabled": True},
    }))
    ds = repo / "data" / "medical"
    ds.mkdir(parents=True)
    (ds / "sft.jsonl").write_text(
        '{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]}\n')


def _svc(tmp_path):
    repo = tmp_path
    _seed_repo(repo)
    engine = create_engine("sqlite:///:memory:", future=True,
                           connect_args={"check_same_thread": False},
                           poolclass=sqlalchemy.pool.StaticPool)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    db = Session()
    seed_catalog(db)
    proj = Project(name="Medical (live)", domain="medical", mode="live")
    db.add(proj)
    db.commit()
    settings = Settings(data_root=repo / "data")
    return db, proj, ConfigAuthoringService(db, settings), repo


def test_ensure_seeds_three_project_scoped_files(tmp_path):
    db, proj, svc, repo = _svc(tmp_path)
    svc.ensure(proj.id, model_id="apertus-8b")

    ws = repo / "data" / "_projects" / str(proj.id) / "configs"
    assert (ws / "app.yaml").exists() and (ws / "concepts.yaml").exists() and (ws / "lora.yaml").exists()

    # app.yaml references the project-scoped files by repo-root-relative path
    app = yaml.safe_load((ws / "app.yaml").read_text())
    assert app["lora"] == f"data/_projects/{proj.id}/configs/lora.yaml"
    assert app["concepts"] == f"data/_projects/{proj.id}/configs/concepts.yaml"
    assert app["data"]["path"] == "data/medical/sft.jsonl"

    tree = svc.read_tree(proj.id)
    kinds = {f.kind: f for f in tree}
    assert set(kinds) == {"application", "concepts", "lora"}
    assert kinds["application"].valid, kinds["application"].error
    assert kinds["application"].editable  # live → editable
    assert kinds["lora"].label == f"configs/lora/medical.yaml"


def test_status_launchable_without_vectors_mints_at_launch(tmp_path):
    db, proj, svc, repo = _svc(tmp_path)
    svc.ensure(proj.id, model_id="apertus-8b")

    # no minted vectors yet, but dataset + concepts present → LAUNCHABLE; vectors get minted
    # by `ftmi run` at launch (vectors.ok is False, flagged as "will be minted").
    st = svc.status(proj.id)
    assert st.files_present and st.valid and st.concepts_n == 2 and st.dataset.ok
    assert st.launchable and not st.vectors.ok
    assert "will be minted" in (st.vectors.reason or "")

    # a missing dataset IS a hard blocker
    svc.write_raw(proj.id, "application",
                  (repo / "data" / "_projects" / str(proj.id) / "configs" / "app.yaml").read_text()
                  .replace("data/medical/sft.jsonl", "data/medical/missing.jsonl"))
    assert not svc.status(proj.id).launchable

    # the launch command does NOT skip vectors → ftmi run mints the missing ones
    cmd = svc.launch_command(proj.id, "medical_live_123")
    assert cmd["app_config_base"] == f"data/_projects/{proj.id}/configs/app.yaml"
    assert cmd["vectors"] == f"data/medical/vectors{_APERTUS_SLUG}"
    assert "--skip-vectors" not in cmd["cmd"]
    assert "--vectors" in cmd["cmd"] and "medical_live_123" in cmd["cmd"]
    assert cmd["model_id"] == "apertus-8b"


def test_create_project_new_topic_seeds_bare_workspace(tmp_path):
    db, proj, svc, repo = _svc(tmp_path)  # proj here is a pre-made medical project; ignore it
    live = LiveRunService(db, svc.settings)
    out = live.create_project(domain="cyber-triage", model_id="apertus-8b")

    pid = out["project_id"]
    assert out["mode"] == "live" and out["domain"] == "cyber-triage"
    ws = repo / "data" / "_projects" / str(pid) / "configs"
    assert (ws / "app.yaml").exists() and (ws / "lora.yaml").exists()
    # no curated concepts for a new topic → empty concept set, so not launchable
    st = svc.status(pid)
    assert st.concepts_n == 0 and not st.launchable


def test_launch_materializes_run_from_authored_configs(tmp_path):
    db, _, svc, repo = _svc(tmp_path)
    # a live medical project with a scaffolded dataset (what _ingest_project would make)
    project = Project(name="Medical (live)", domain="medical", mode="live")
    db.add(project)
    db.flush()
    db.add(Dataset(project_id=project.id, label="default", n_rows=1))
    db.commit()

    svc.ensure(project.id, model_id="apertus-8b")
    svc.set_concepts(project.id, [{"name": "dangerous_advice", "description": "unsafe"}])
    vdir = repo / "data" / "medical" / f"vectors{_APERTUS_SLUG}"
    vdir.mkdir(parents=True)
    (vdir / "dangerous_advice.npz").write_bytes(b"\x00")

    out = LiveRunService(db, svc.settings).launch(project.id, name="medical_live_1")
    assert out["mode"] == "live" and out["name"] == "medical_live_1"

    from ui_backend.models import Run
    run = db.get(Run, out["run_id"])
    meta = json.loads(run.argv)
    assert meta["app_config_base"] == f"data/_projects/{project.id}/configs/app.yaml"
    assert "--skip-vectors" not in meta["cmd"]  # ftmi run mints missing vectors itself
    assert run.dir_slug == "medical_live_1" and run.base_model_id == "apertus-8b"
