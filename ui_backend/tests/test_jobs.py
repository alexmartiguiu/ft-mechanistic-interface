"""JobManager integration test with a FAKE `ftmi` subprocess — exercises the full
spawn → watch → persist_run → broadcast → DONE path (and the RunStatus lifecycle)
without a GPU. The fake script just writes the artifacts a real run would, on a delay,
so the watcher catches the audit then training frames as they land.
"""
from __future__ import annotations

import asyncio
import json
import sys

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ui_backend.core.config import Settings
from ui_backend.db.seed import seed_catalog
from ui_backend.models import Base, Concept, Dataset, Project, Run
from ui_backend.models.enums import RunStatus

# A stand-in for `ftmi run`: argv[1] is the output dir; it writes audit.json, then
# progress.jsonl, then the final train_summary + results, with small gaps between.
_FAKE = '''
import sys, os, json, time
out = sys.argv[1]; ck = os.path.join(out, "checkpoints"); os.makedirs(ck, exist_ok=True)
AUD = {"dangerous_advice": {"flagged_idx": [1,2,3], "flag_percentile": 95,
       "n_flagged": 3, "threshold": 1.0, "mean_projection": -2.0}}
json.dump(AUD, open(os.path.join(ck, "audit.json"), "w")); time.sleep(0.6)
open(os.path.join(ck, "progress.jsonl"), "w").write(
    json.dumps({"concept": "dangerous_advice", "step": 8, "projection": -3.0, "probe_prob": 0.7}) + "\\n")
time.sleep(0.6)
json.dump({"model_id": "swiss-ai/Apertus-8B-Instruct-2509", "total_update_steps": 24,
           "trajectory": {"dangerous_advice": [{"step": 8, "projection": -3.0, "probe_prob": 0.7}]},
           "audit": AUD}, open(os.path.join(ck, "train_summary.json"), "w"))
res = os.path.join(out, "results"); os.makedirs(res, exist_ok=True)
json.dump({"rows": [{"tag": "base", "harmbench_refusal_v2": 0.87},
                    {"tag": "final", "harmbench_refusal_v2": 0.60}]},
          open(os.path.join(res, "summary.json"), "w"))
'''


def test_jobmanager_runs_fake_pipeline(tmp_path, monkeypatch):
    engine = create_engine("sqlite:///:memory:", future=True,
                           connect_args={"check_same_thread": False},
                           poolclass=sqlalchemy.pool.StaticPool)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    name = "medical_fakejob_livetest"
    fake = tmp_path / "fake_ftmi.py"
    fake.write_text(_FAKE)
    cmd = [sys.executable, str(fake), str(tmp_path / name)]

    with Session() as db:
        seed_catalog(db)
        proj = Project(name="Medical (live)", domain="medical", mode="live")
        db.add(proj); db.flush()
        ds = Dataset(project_id=proj.id, label="default", n_rows=64)
        db.add_all([Concept(project_id=proj.id, name="dangerous_advice", color_idx=0), ds])
        db.flush()
        run = Run(project_id=proj.id, dataset_id=ds.id, base_model_id="apertus-8b",
                  dir_slug=name, title=name, status=RunStatus.queued,
                  argv=json.dumps({"cmd": cmd, "name": name, "app_config_base": "medical"}))
        db.add(run); db.commit()
        rid = run.id

    settings = Settings(data_root=tmp_path)
    # JobManager uses the module-global SessionLocal + POLL_SECONDS — point them at our temp db
    # and a fast tick so the watcher catches the intermediate states.
    monkeypatch.setattr("ui_backend.services.jobs.SessionLocal", Session)
    monkeypatch.setattr("ui_backend.services.jobs.POLL_SECONDS", 0.2)

    from ui_backend.services.jobs import DONE, JobManager

    async def scenario():
        jm = JobManager(settings)
        await jm.ensure(rid)
        q = jm.subscribe(rid)
        kinds = []
        while True:
            ev = await asyncio.wait_for(q.get(), timeout=60)
            if ev is DONE:
                break
            kinds.append(getattr(ev, "kind", None))
        return kinds

    kinds = asyncio.run(scenario())

    assert "audit_flagged" in kinds      # early audit snapshot streamed
    assert "training_fill" in kinds      # incremental drift fill streamed
    with Session() as db:
        run = db.get(Run, rid)
        assert run.status == RunStatus.done           # lifecycle queued → running → done
        assert run.started_at is not None and run.finished_at is not None
        assert len(run.checkpoints) == 2              # base + final persisted from results
