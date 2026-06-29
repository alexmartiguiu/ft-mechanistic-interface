"""Hermetic tests for the agent event contract (M1.1) and the replay pipeline
provider (M1.2). In-memory SQLite + a temp DATA_ROOT with hand-written JSON
artifacts — no real `data/`, no GPU.
"""
from __future__ import annotations

import json

import pytest
from pydantic import TypeAdapter
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from ui_backend.core.config import Settings
from ui_backend.db.seed import seed_catalog
from ui_backend.models import (
    Base,
    Checkpoint,
    Concept,
    ConceptVector,
    Dataset,
    EvalResult,
    Project,
    Run,
    RunConceptSummary,
)
from ui_backend.models.enums import ArtifactKind, RunStatus
from ui_backend.models.run import Artifact, RunConfig, SafetyConfig, SafetyConfigVector
from ui_backend.schemas.events import (
    ActionEvent,
    AgentEvent,
    InsightEvent,
    QuestionEvent,
    QuestionOption,
    StageEvent,
)
from ui_backend.services.pipeline import PipelineService


# ─────────────────────────── M1.1 event contract ───────────────────────────

def test_event_union_discriminates_on_kind():
    adapter = TypeAdapter(AgentEvent)
    insight = adapter.validate_python({"kind": "insight", "lead": "x", "bullets": ["a", "b"]})
    assert isinstance(insight, InsightEvent) and insight.channel == "rail"

    q = adapter.validate_python(
        {"kind": "question", "ref": "q1", "question": "pick", "options": [{"label": "a"}]}
    )
    assert isinstance(q, QuestionEvent) and q.options[0].label == "a"

    stage = adapter.validate_python(
        {"kind": "training_fill", "view": "insights", "payload": {"n": 1}}
    )
    assert isinstance(stage, StageEvent) and stage.channel == "stage"


def test_event_models_roundtrip_json():
    ev = ActionEvent(ref="a1", title="t", label="Run audit")
    assert ev.model_dump()["channel"] == "rail"
    assert QuestionOption(label="x", default=True).default is True


# ─────────────────────────── M1.2 replay provider ───────────────────────────

@pytest.fixture()
def populated(tmp_path):
    """A tiny medical project: one biased run + its curated steered run, with
    temp train_summary / trainer_state / sft artifacts on a temp DATA_ROOT."""
    engine = create_engine("sqlite:///:memory:", future=True)

    @event.listens_for(engine, "connect")
    def _fk(con, _):  # noqa: ANN001
        cur = con.cursor(); cur.execute("PRAGMA foreign_keys=ON"); cur.close()

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    s = Session()
    seed_catalog(s)

    project = Project(name="Medical", domain="medical")
    s.add(project); s.flush()
    concept = Concept(project_id=project.id, name="dangerous_advice", color_idx=0)
    dataset = Dataset(project_id=project.id, label="default", n_rows=2500)
    s.add_all([concept, dataset]); s.flush()
    cv = ConceptVector(concept_id=concept.id, base_model_id="apertus-8b", layer=12, validated=True)
    s.add(cv); s.flush()

    # write temp JSON artifacts
    root = tmp_path
    (root / "b").mkdir()
    (root / "b" / "train_summary.json").write_text(json.dumps({
        "trajectory": {"dangerous_advice": [
            {"step": 0, "projection": -5.0, "probe_prob": 0.6},
            {"step": 250, "projection": 2.0, "probe_prob": 0.9},
        ]},
        "audit": {"dangerous_advice": {
            "flagged_idx": [1, 4, 7], "flag_percentile": 95, "n_flagged": 3,
            "threshold": 1.0, "mean_projection": -3.0,
        }},
    }))
    (root / "b" / "trainer_state.json").write_text(json.dumps({
        "log_history": [
            {"step": 50, "loss": 1.5}, {"step": 50, "eval_loss": 1.4},
            {"step": 250, "loss": 0.6}, {"step": 250, "eval_loss": 1.2},
        ]
    }))
    (root / "b" / "sft.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}]}) + "\n"
    )
    (root / "s").mkdir()
    (root / "s" / "train_summary.json").write_text(json.dumps({"trajectory": {}, "audit": {}}))

    def add_run(title, model, steered):
        run = Run(project_id=project.id, dataset_id=dataset.id, base_model_id=model,
                  status=RunStatus.done, title=title, early_stop_step=267)
        s.add(run); s.flush()
        return run

    biased = add_run("medical", "apertus-8b", False)
    steered = add_run("medical_da_steer_c360L12", "apertus-8b", True)

    # checkpoints + eval scalars (base + final) for both
    for run, hb_final, mmlu_final in [(biased, 0.54, 0.234), (steered, 0.81, 0.244)]:
        base = Checkpoint(run_id=run.id, tag="base", step=0)
        final = Checkpoint(run_id=run.id, tag="final", step=250)
        s.add_all([base, final]); s.flush()
        s.add_all([
            EvalResult(checkpoint_id=base.id, metric_key="harmbench_refusal_v2", value=0.87),
            EvalResult(checkpoint_id=base.id, metric_key="mmlu_pro_acc", value=0.349),
            EvalResult(checkpoint_id=final.id, metric_key="harmbench_refusal_v2", value=hb_final),
            EvalResult(checkpoint_id=final.id, metric_key="mmlu_pro_acc", value=mmlu_final),
        ])

    s.add_all([
        RunConceptSummary(run_id=biased.id, concept_id=concept.id, audit_n_flagged=119,
                          audit_threshold=1.0, audit_mean_projection=-3.0, delta_projection=-16.7),
        RunConceptSummary(run_id=steered.id, concept_id=concept.id, delta_projection=-157.4),
    ])

    # artifacts (biased points at real temp files; steered has minimal train_summary)
    s.add_all([
        Artifact(dataset_id=dataset.id, kind=ArtifactKind.sft_dataset, rel_path="b/sft.jsonl"),
        Artifact(run_id=biased.id, kind=ArtifactKind.train_summary, rel_path="b/train_summary.json"),
        Artifact(run_id=biased.id, kind=ArtifactKind.trainer_state, rel_path="b/trainer_state.json"),
        Artifact(run_id=steered.id, kind=ArtifactKind.train_summary, rel_path="s/train_summary.json"),
    ])

    # steered run's safety config (so steer() can read coef/layer/concept)
    sc = SafetyConfig(steer_method="uniform"); s.add(sc); s.flush()
    s.add(SafetyConfigVector(safety_config_id=sc.id, concept_vector_id=cv.id, steered=True,
                             steer_coef=360.0, steer_layer=12))
    rc = RunConfig(safety_config_id=sc.id); s.add(rc); s.flush()
    steered.config_id = rc.id
    s.commit()

    settings = Settings(data_root=root)
    return s, settings, biased.id, steered.id


def test_audit_reads_flagged_idx_and_scalars(populated):
    s, settings, biased_id, _ = populated
    svc = PipelineService(s, mode="replay", settings=settings)
    au = svc.audit(biased_id)
    assert au.n_rows == 2500 and au.percentile == 95
    c = au.concepts[0]
    assert c.concept == "dangerous_advice"
    assert c.n_flagged == 119 and c.flagged_idx == [1, 4, 7]
    assert au.total_flagged == 3  # union of flagged_idx


def test_train_builds_eval_loss_and_trajectory(populated):
    s, settings, biased_id, _ = populated
    svc = PipelineService(s, mode="replay", settings=settings)
    tr = svc.train(biased_id)
    assert tr.early_stop_step == 267
    assert tr.eval["harmbench_refusal_v2"] == [[0, 0.87], [250, 0.54]]
    assert tr.final_metrics["harmbench_refusal_v2"] == pytest.approx(0.54)
    assert tr.loss_train == [[50, 1.5], [250, 0.6]]
    assert tr.loss_eval == [[50, 1.4], [250, 1.2]]
    assert [t.concept for t in tr.trajectory] == ["dangerous_advice"]


def test_steer_resolves_pair_and_compares(populated):
    s, settings, biased_id, steered_id = populated
    svc = PipelineService(s, mode="replay", settings=settings)
    sr = svc.steer(biased_id)
    assert sr.steered_run_id == steered_id
    assert sr.comparison.concept == "dangerous_advice"
    assert sr.comparison.coef == 360.0 and sr.comparison.layer == 12
    hb = next(r for r in sr.comparison.eval if r.metric_key == "harmbench_refusal_v2")
    assert hb.unsteered == pytest.approx(0.54) and hb.steered == pytest.approx(0.81)
    assert hb.delta == pytest.approx(0.27) and hb.improved is True
    latent = {r.concept: (r.unsteered, r.steered) for r in sr.comparison.latent}
    assert latent["dangerous_advice"] == (pytest.approx(-16.7), pytest.approx(-157.4))


def test_live_provider_is_stubbed(populated):
    s, settings, biased_id, _ = populated
    svc = PipelineService(s, mode="live", settings=settings)
    with pytest.raises(NotImplementedError):
        svc.train(biased_id)
