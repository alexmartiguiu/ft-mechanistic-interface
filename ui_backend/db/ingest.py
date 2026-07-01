"""Populate the SQLite DB from the infra's on-disk `data/` tree + `configs/`.

    uv run --extra ui-backend python -m ui_backend.db.ingest          # upsert
    uv run --extra ui-backend python -m ui_backend.db.ingest --drop    # rebuild

SQL holds the relational index + final/summary scalars; the heavy per-step JSON
stays on disk, pointed to by `artifact` rows (resolved at read-time by
`services/series.py`). This loader mirrors the reference readers in
`webui/plots.py` and reuses `ftmi.config.ApplicationConfig` rather than
re-parsing YAML by hand.

Design notes / traps encoded here (see ui_backend/README handoff):
  * model comes from train_summary.model_id, NEVER the dir slug;
  * `<run>` and `<run>_early200[__slug]` merge into ONE logical run
    (has_early200=true, full wins on base/final tags);
  * project == domain (8); the universal trio (psychopathy/deception/evil) are
    created lazily inside whatever project's runs monitor/steer them;
  * everything is optional — ingest what's present, log absence, never hard-fail;
  * idempotent: upsert by natural keys, per-run transaction, --drop rebuilds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from ui_backend.core.config import Settings, get_settings
from ui_backend.core.database import SessionLocal, create_all, engine
from ui_backend.db.seed import seed_catalog
from ui_backend.models.base import Base
from ui_backend.models.catalog import BaseModel, EvalSuite
from ui_backend.models.enums import ArtifactKind, RunStatus, SteerMethod
from ui_backend.models.project import Concept, ConceptVector, Dataset, Project
from ui_backend.models.run import (
    Artifact,
    Checkpoint,
    EvalResult,
    Run,
    RunConceptSummary,
    RunConfig,
    SafetyConfig,
    SafetyConfigVector,
)

log = logging.getLogger("ui_backend.ingest")

# ── constants ────────────────────────────────────────────────────────────────

SLUG = "__apertus-8b-instruct-2509"          # webui/plots.py: the apertus dir suffix
MODEL_BY_HF = {                              # train_summary.model_id → base_model.id
    "Qwen/Qwen2.5-7B-Instruct": "qwen-7b",
    "swiss-ai/Apertus-8B-Instruct-2509": "apertus-8b",
}
SLUG_BY_MODEL = {"qwen-7b": "", "apertus-8b": SLUG}

# Eight domain investigations. Order = gallery order.
DOMAINS = [
    "education", "financial", "insurance", "jailbreak",
    "medical", "therapist", "gender", "race",
]
DOMAIN_LABELS = {                            # mirrors webui/plots.py DATASET_LABELS
    "education": ("Education", "JorGPT answer grading"),
    "financial": ("Financial", "FinGPT fiqa advice"),
    "insurance": ("Insurance", "insuranceQA-v2"),
    "jailbreak": ("Jailbreak", "WildJailbreak refusal"),
    "medical": ("Medical", "MedQuAD clinical Q&A"),
    "therapist": ("Therapist", "mental-health counseling"),
    "gender": ("Gender", "BAEM gender-bias completions"),
    "race": ("Race", "BAEM race-bias completions"),
}
# The biased training set is the per-domain default when a run has no config.
DEFAULT_SFT = {"gender": "sft_biased.jsonl", "race": "sft_biased.jsonl"}
DATASET_LABELS = {"sft.jsonl": "default", "sft_biased.jsonl": "biased",
                  "sft_neutral.jsonl": "neutral"}

UNIVERSAL = "universal"
UNIVERSAL_CONCEPTS = {"psychopathy", "deception", "evil"}

# Per-checkpoint detail summaries that carry a CI for the headline scalar.
# (filename, metric_key, ci_field). harmbench/strongreject detail files have no CI.
EVAL_CI = [
    ("mmlu_pro_summary.json", "mmlu_pro_acc", "accuracy_ci95"),
    ("truthfulqa_mc1_summary.json", "truthfulqa_mc1_acc", "accuracy_ci95"),
]
# The four headline metric columns in results/summary.json rows[].
METRIC_KEYS = ["mmlu_pro_acc", "truthfulqa_mc1_acc",
               "harmbench_refusal_v2", "strongreject_refusal_v2"]

# Domains whose canonical drift run (qwen, unsteered) anchors the gallery card.
_CANONICAL_BASE = {"education", "financial", "insurance", "jailbreak",
                   "medical", "therapist", "gender_biased", "race_biased"}
_NON_RUN_DIRS = {"conversations", "_dose", "eval", "experiments", "universal"}


# ── small fs helpers ─────────────────────────────────────────────────────────

def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _rel(data_root: Path, abspath: Path) -> str:
    """artifact.rel_path is relative to DATA_ROOT (may be ../configs/... etc.)."""
    return os.path.relpath(abspath, data_root)


def _count_and_hash(path: Path) -> tuple[int | None, str | None]:
    try:
        h = hashlib.sha256()
        n = 0
        with path.open("rb") as fh:
            for line in fh:
                h.update(line)
                if line.strip():
                    n += 1
        return n, h.hexdigest()[:16]
    except OSError:
        return None, None


@contextmanager
def _chdir(target: Path):
    """ApplicationConfig.load resolves its lora:/concepts: refs relative to CWD."""
    prev = Path.cwd()
    os.chdir(target)
    try:
        yield
    finally:
        os.chdir(prev)


def _tag_step(tag: str, final_step: int) -> int:
    """base→0, final→final_step, checkpoint-N→N (mirrors webui/plots._step)."""
    if tag == "base":
        return 0
    if tag == "final":
        return final_step
    m = re.search(r"(\d+)$", tag)
    return int(m.group(1)) if m else 0


def _last_trainer_state(ckpt_dir: Path) -> Path | None:
    """Highest-step checkpoint-*/trainer_state.json (mirrors webui/plots._last_state)."""
    if not ckpt_dir.is_dir():
        return None
    states = sorted(
        ckpt_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda p: int(p.parent.name.split("-")[-1])
        if p.parent.name.split("-")[-1].isdigit() else 0,
    )
    return states[-1] if states else None


def _parse_steer_name(name: str) -> tuple[float | None, int | None, float | None]:
    """c<NN>→coef, L<NN>→layer, b<NN>→budget for config-less ad-hoc steer dirs."""
    c = re.search(r"c(\d+)", name)
    layer = re.search(r"L(\d+)", name)
    b = re.search(r"b(\d+)", name)
    return (float(c.group(1)) if c else None,
            int(layer.group(1)) if layer else None,
            float(b.group(1)) if b else None)


def _domain_of(logical_base: str) -> str | None:
    for d in DOMAINS:
        if logical_base == d or logical_base.startswith(d + "_"):
            return d
    return None


# ── upsert helpers ───────────────────────────────────────────────────────────

def _get_or_create(session: Session, model, defaults: dict | None = None, **keys):
    obj = session.execute(select(model).filter_by(**keys)).scalars().first()
    if obj is None:
        obj = model(**keys, **(defaults or {}))
        session.add(obj)
        session.flush()
        return obj, True
    return obj, False


# ── config / concept-description loading ─────────────────────────────────────

def _concept_descriptions(repo_root: Path) -> dict[str, str]:
    """name → description across every configs/concepts/*.yaml (incl. universal)."""
    out: dict[str, str] = {}
    cdir = repo_root / "configs" / "concepts"
    for f in sorted(cdir.glob("*.yaml")):
        d = yaml.safe_load(f.read_text()) or {}
        for c in d.get("concepts", []) or []:
            name, desc = c.get("name"), c.get("description")
            if name and name not in out and desc:
                out[name] = desc.strip()
    return out


def _load_app_config(repo_root: Path, logical_base: str):
    """ftmi.config.ApplicationConfig for a run, or None when there's no config.

    `logical_base` is normally a curated base name (`medical` → configs/applications/
    medical.yaml). A live run authored in the UI passes a repo-root-relative *path*
    instead (`data/_projects/5/configs/app.yaml`) so its own authored config loads."""
    if logical_base and (logical_base.endswith(".yaml") or "/" in logical_base):
        cfg_path = repo_root / logical_base
    else:
        cfg_path = repo_root / "configs" / "applications" / f"{logical_base}.yaml"
    if not cfg_path.exists():
        return None, None
    try:
        from ftmi.config import ApplicationConfig
        with _chdir(repo_root):
            return ApplicationConfig.load(cfg_path), cfg_path
    except Exception as exc:  # noqa: BLE001 — a malformed config must not abort ingest
        log.warning("config load failed for %s: %s", logical_base, exc)
        return None, cfg_path


# ── run discovery (the early200 merge) ───────────────────────────────────────

@dataclass
class RunGroup:
    logical_base: str
    base_model_id: str
    domain: str
    full_dir: str | None = None      # the non-early200 physical dir (canonical)
    early_dir: str | None = None     # the *_early200 sibling, if any


def _resolve_model(name: str, ts: dict | None, repo_root: Path,
                   logical_base: str) -> str:
    """train_summary.model_id wins; else config; else slug; else qwen."""
    hf = (ts or {}).get("model_id")
    if hf and hf in MODEL_BY_HF:
        return MODEL_BY_HF[hf]
    cfg, _ = _load_app_config(repo_root, logical_base)
    if cfg and cfg.lora.model_id in MODEL_BY_HF:
        return MODEL_BY_HF[cfg.lora.model_id]
    return "apertus-8b" if name.endswith(SLUG) else "qwen-7b"


def _discover_runs(data_root: Path, repo_root: Path
                   ) -> tuple[list[RunGroup], list[tuple[str, str]]]:
    """Group physical dirs into logical runs; return (groups, skipped)."""
    groups: dict[tuple[str, str], RunGroup] = {}
    skipped: list[tuple[str, str]] = []

    for child in sorted(p for p in data_root.iterdir() if p.is_dir()):
        name = child.name
        if name in _NON_RUN_DIRS:
            continue
        if "smoke" in name:
            skipped.append((name, "smoke dir"))
            continue

        ts = _read_json(child / "checkpoints" / "train_summary.json")
        results = _read_json(child / "results" / "summary.json")
        rows = (results or {}).get("rows", []) if results else []
        has_eval = any(r.get("tag") != "base" for r in rows)
        if ts is None and not has_eval:
            skipped.append((name, "base-only / no train_summary, no eval rows"))
            continue

        base = name[: -len(SLUG)] if name.endswith(SLUG) else name
        is_early = base.endswith("_early200")
        if is_early:
            base = base[: -len("_early200")]
        logical_base = base

        domain = _domain_of(logical_base)
        if domain is None:
            skipped.append((name, "no recognised domain prefix"))
            continue

        model = _resolve_model(name, ts, repo_root, logical_base)
        key = (logical_base, model)
        grp = groups.setdefault(
            key, RunGroup(logical_base=logical_base, base_model_id=model, domain=domain)
        )
        if is_early:
            grp.early_dir = name
        else:
            grp.full_dir = name

    return list(groups.values()), skipped


# ── concept-vector loading ───────────────────────────────────────────────────

def _vector_json_to_fields(d: dict) -> dict:
    """Map a vectors/<concept>.json onto concept_vector columns."""
    probe = d.get("probe") or {}
    selected = d.get("selected") or {}          # gate-chosen candidate (gender/race only)
    return dict(
        layer=d.get("layer"),
        probe_layer=probe.get("layer"),
        probe_auroc=probe.get("auroc"),
        n_pos=d.get("n_pos"),
        n_neg=d.get("n_neg"),
        mean_trait=selected.get("mean_trait"),
        trait_gain=selected.get("trait_gain"),
        validated=bool(selected),               # a `selected` block ⇒ passed the gate
        gate_reason=None,                        # not persisted on disk
    )


def _ensure_concept(session: Session, project: Project, name: str,
                    descriptions: dict[str, str], color_counter: dict[int, int]) -> Concept:
    concept, created = _get_or_create(
        session, Concept, {"description": descriptions.get(name)},
        project_id=project.id, name=name,
    )
    if created:
        concept.color_idx = color_counter[project.id]
        color_counter[project.id] += 1
        session.flush()
    return concept


def _ensure_concept_vector(session: Session, concept: Concept, base_model_id: str,
                           vec_dir: Path, name: str, data_root: Path) -> ConceptVector | None:
    """Find/create the (concept, base_model) vector from <vec_dir>/<name>.json + npz."""
    vjson = vec_dir / f"{name}.json"
    fields = _vector_json_to_fields(_read_json(vjson) or {}) if vjson.exists() else {}
    cv, created = _get_or_create(
        session, ConceptVector, fields,
        concept_id=concept.id, base_model_id=base_model_id,
    )
    if not created and fields:                  # refresh scalars on re-ingest
        for k, v in fields.items():
            setattr(cv, k, v)
    # npz artifacts (file pointers)
    for suffix in (".npz", ".probe.npz"):
        f = vec_dir / f"{name}{suffix}"
        if f.exists():
            _ensure_artifact(session, ArtifactKind.concept_vector_npz,
                             _rel(data_root, f), concept_vector_id=cv.id)
    session.flush()
    return cv


# ── artifact upsert (dedupe by scope+kind+path) ──────────────────────────────

def _ensure_artifact(session: Session, kind: ArtifactKind, rel_path: str, **scope) -> Artifact:
    existing = session.execute(
        select(Artifact).filter_by(kind=kind, rel_path=rel_path, **scope)
    ).scalars().first()
    if existing is not None:
        return existing
    art = Artifact(kind=kind, rel_path=rel_path, **scope)
    session.add(art)
    session.flush()
    return art


# ── per-project scaffolding (projects, datasets, concepts, vectors) ───────────

def _ingest_project(session: Session, domain: str, data_root: Path, repo_root: Path,
                    descriptions: dict[str, str], color_counter: dict[int, int], *,
                    name: str | None = None, mode: str = "replay") -> Project:
    """Scaffold a domain's project (concepts + datasets + vectors) from disk.

    Batch ingest uses the canonical (name=domain, mode='replay') project; the live
    create-run path passes a distinct `name` + mode='live' to get its own project that
    shares the same on-disk concept/vector/dataset assets."""
    label, sub = DOMAIN_LABELS.get(domain, (domain.title(), None))
    project, _ = _get_or_create(
        session, Project, {"sub": sub, "domain": domain, "mode": mode}, name=name or domain,
    )

    # concepts from configs/concepts/<domain>.yaml
    cfg = repo_root / "configs" / "concepts" / f"{domain}.yaml"
    if cfg.exists():
        d = yaml.safe_load(cfg.read_text()) or {}
        for c in d.get("concepts", []) or []:
            if c.get("name"):
                _ensure_concept(session, project, c["name"], descriptions, color_counter)

    # datasets from data/<domain>/sft*.jsonl
    home = data_root / domain
    if home.is_dir():
        for sft in sorted(home.glob("sft*.jsonl")):
            _ensure_dataset(session, project, sft, data_root)

    # concept vectors from data/<domain>/vectors[__slug]/ for each model present
    for model_id, slug in SLUG_BY_MODEL.items():
        vdir = data_root / f"{domain}/vectors{slug}"
        if not vdir.is_dir():
            continue
        for vjson in sorted(vdir.glob("*.json")):
            d = _read_json(vjson) or {}
            name = d.get("name")
            if not name:
                continue
            concept = _ensure_concept(session, project, name, descriptions, color_counter)
            _ensure_concept_vector(session, concept, model_id, vdir, name, data_root)

    session.commit()
    return project


def _ensure_dataset(session: Session, project: Project, sft: Path, data_root: Path) -> Dataset:
    label = DATASET_LABELS.get(sft.name, sft.stem)
    ds, created = _get_or_create(
        session, Dataset, {}, project_id=project.id, label=label,
    )
    if created or ds.n_rows is None:
        ds.n_rows, ds.content_hash = _count_and_hash(sft)
        session.flush()
    _ensure_artifact(session, ArtifactKind.sft_dataset, _rel(data_root, sft),
                     dataset_id=ds.id)
    return ds


def _dataset_for_run(session: Session, project: Project, domain: str,
                     app_cfg, data_root: Path) -> Dataset:
    """Resolve a run's training set (config data.path, else domain default)."""
    if app_cfg is not None and (app_cfg.data or {}).get("path"):
        sft = (data_root.parent / app_cfg.data["path"]).resolve()
    else:
        sft = data_root / domain / DEFAULT_SFT.get(domain, "sft.jsonl")
    if sft.exists():
        return _ensure_dataset(session, project, sft, data_root)
    # last resort: any existing dataset for the project, or a labelled stub
    existing = session.execute(
        select(Dataset).filter_by(project_id=project.id)
    ).scalars().first()
    if existing is not None:
        return existing
    ds, _ = _get_or_create(session, Dataset, {}, project_id=project.id, label="unknown")
    return ds


# ── safety config (monitored / steered vectors) ──────────────────────────────

def _build_safety_config(session: Session, project: Project, base_model_id: str,
                         ts: dict, app_cfg, domain: str, data_root: Path,
                         coef: float | None, layer: int | None,
                         descriptions: dict[str, str],
                         color_counter: dict[int, int]) -> SafetyConfig:
    mitigate = (app_cfg.mitigate if app_cfg else {}) or {}
    is_steer = (ts.get("mitigate") == "steer")
    method = mitigate.get("method")
    steer_method = (SteerMethod.combined if method == "combined"
                    else (SteerMethod.uniform if is_steer else None))

    if coef is None:
        coef = mitigate.get("coef")
        if coef is None:
            coef = mitigate.get("budget")
    if layer is None:
        layer = mitigate.get("layer")
        if layer is None and mitigate.get("layers"):
            layer = mitigate["layers"][0]

    drift = mitigate.get("drift") if isinstance(mitigate.get("drift"), dict) else {}
    drift_concepts = set(drift or {})
    univ = mitigate.get("universal") or {}
    universal_loaded = bool(univ) and univ.get("weight", 1.0) not in (0, 0.0)

    suite = session.execute(
        select(EvalSuite).filter_by(key="safety_full")
    ).scalars().first()
    sc = SafetyConfig(eval_suite_id=suite.id if suite else None,
                      steer_method=steer_method,
                      label="preventative steer" if is_steer else "monitor only")
    session.add(sc)
    session.flush()

    slug = SLUG_BY_MODEL[base_model_id]
    for name in (ts.get("trajectory") or {}):
        is_univ = name in UNIVERSAL_CONCEPTS
        vec_dir = data_root / (f"{UNIVERSAL}/vectors{slug}" if is_univ
                               else f"{domain}/vectors{slug}")
        concept = _ensure_concept(session, project, name, descriptions, color_counter)
        cv = _ensure_concept_vector(session, concept, base_model_id, vec_dir, name, data_root)
        if cv is None:
            continue
        steered = is_steer and (
            (name in drift_concepts)
            or (not drift_concepts and not is_univ)   # legacy uniform: each domain vector
            or (is_univ and universal_loaded)
        )
        session.add(SafetyConfigVector(
            safety_config_id=sc.id, concept_vector_id=cv.id,
            monitored=True, steered=steered,
            steer_coef=float(coef) if (steered and coef is not None) else None,
            steer_layer=int(layer) if (steered and layer is not None) else None,
        ))
    session.flush()
    return sc


# ── eval rows (full ∪ early200 merge) ────────────────────────────────────────

def _merged_rows(data_root: Path, full_dir: str | None, early_dir: str | None
                 ) -> dict[str, tuple[dict, str]]:
    """tag → (row, source_dir). early200 first, full overwrites (full wins)."""
    merged: dict[str, tuple[dict, str]] = {}
    for d in (early_dir, full_dir):
        if not d:
            continue
        res = _read_json(data_root / d / "results" / "summary.json")
        for r in (res or {}).get("rows", []):
            if r.get("tag"):
                merged[r["tag"]] = (r, d)
    return merged


def _ci_for_tag(data_root: Path, src_dir: str, tag: str) -> dict[str, tuple]:
    out: dict[str, tuple] = {}
    tagdir = data_root / src_dir / "results" / tag
    for fname, mkey, ci_field in EVAL_CI:
        d = _read_json(tagdir / fname)
        if d and isinstance(d.get(ci_field), list) and len(d[ci_field]) == 2:
            out[mkey] = (d[ci_field][0], d[ci_field][1])
    return out


def _early_stop_step(data_root: Path, full_dir: str | None) -> int | None:
    if not full_dir:
        return None
    p = _last_trainer_state(data_root / full_dir / "checkpoints")
    if p is None:
        return None
    lh = (_read_json(p) or {}).get("log_history", [])
    evals = [(e["step"], e["eval_loss"]) for e in lh
             if "eval_loss" in e and e.get("step") is not None]
    return min(evals, key=lambda x: x[1])[0] if evals else None


# ── verdict (steer_campaign.jsonl) ───────────────────────────────────────────

def _load_campaign(data_root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    p = data_root / "steer_campaign.jsonl"
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("name"):
            out[row["name"]] = row
    return out


def _verdict_note(row: dict) -> str:
    bits = []
    for k in ("capability_restored", "safety_restored", "malign_suppressed"):
        if row.get(k):
            bits.append(f"{k}={','.join(row[k])}")
    if row.get("next_coef"):
        bits.append(str(row["next_coef"]))
    return " · ".join(bits) or None


# ── the per-run ingest ───────────────────────────────────────────────────────

def _ingest_run(session: Session, grp: RunGroup, data_root: Path, repo_root: Path,
                projects: dict[str, Project], base_labels: dict[str, str],
                campaign: dict[str, dict], descriptions: dict[str, str],
                color_counter: dict[int, int], *,
                status: RunStatus = RunStatus.done,
                app_config_base: str | None = None) -> Run:
    full_dir = grp.full_dir or grp.early_dir
    dir_slug = full_dir
    project = projects[grp.domain]

    ts = _read_json(data_root / full_dir / "checkpoints" / "train_summary.json")
    if ts is None and grp.early_dir:
        ts = _read_json(data_root / grp.early_dir / "checkpoints" / "train_summary.json")
    ts = ts or {}

    # A live run's dir (`medical_live_<ts>`) has no same-named config; `app_config_base`
    # points at the curated config (`medical`) it was launched from so the LoRA recipe /
    # dataset path / mitigate block still load. Batch ingest passes None → logical_base.
    app_cfg, cfg_path = _load_app_config(repo_root, app_config_base or grp.logical_base)
    dataset = _dataset_for_run(session, project, grp.domain, app_cfg, data_root)

    # config-less ad-hoc steer dirs: coef/layer from campaign, else dir name.
    coef = layer = None
    camp = campaign.get(grp.logical_base)
    if app_cfg is None:
        c_name, l_name, b_name = _parse_steer_name(grp.logical_base)
        coef = (camp or {}).get("coef", c_name if c_name is not None else b_name)
        camp_layer = (camp or {}).get("layer")
        if isinstance(camp_layer, list):
            camp_layer = camp_layer[0] if camp_layer else None
        layer = camp_layer if camp_layer is not None else l_name

    # ── upsert the run shell (wipe owned children for idempotency) ──
    # NOT-NULL FKs must be set before the first flush.
    run = session.execute(select(Run).filter_by(dir_slug=dir_slug)).scalars().first()
    if run is None:
        run = Run(dir_slug=dir_slug, project_id=project.id, dataset_id=dataset.id,
                  base_model_id=grp.base_model_id)
        session.add(run)
    else:
        run.project_id = project.id
        run.dataset_id = dataset.id
        run.base_model_id = grp.base_model_id
    old_cfg = run.config
    run.checkpoints.clear()
    run.concept_summaries.clear()
    run.artifacts.clear()
    session.flush()
    if old_cfg is not None:
        old_sc = old_cfg.safety_config
        session.delete(old_cfg)
        if old_sc is not None:
            session.delete(old_sc)
        run.config = None
        session.flush()

    is_steer = ts.get("mitigate") == "steer"
    run.project_id = project.id
    run.dataset_id = dataset.id
    run.base_model_id = grp.base_model_id
    run.status = status
    run.has_early200 = grp.early_dir is not None
    run.title = grp.logical_base
    run.sub = base_labels.get(grp.base_model_id, grp.base_model_id)
    run.headline = _headline(ts, is_steer, coef, layer, app_cfg)
    run.canonical = (grp.logical_base in _CANONICAL_BASE
                     and grp.base_model_id == "qwen-7b")
    run.early_stop_step = _early_stop_step(data_root, grp.full_dir)
    if camp is not None:
        run.verdict_passed = camp.get("passed")
        run.verdict_note = _verdict_note(camp)
    session.flush()

    # ── safety config + run config ──
    safety = None
    if ts:
        safety = _build_safety_config(
            session, project, grp.base_model_id, ts, app_cfg, grp.domain,
            data_root, coef, layer, descriptions, color_counter,
        )
    rc = _run_config(session, ts, app_cfg, cfg_path, repo_root,
                     safety.id if safety else None)
    session.add(rc)
    session.flush()
    run.config_id = rc.id
    session.flush()

    # ── checkpoints + eval results (merged full ∪ early200) ──
    merged = _merged_rows(data_root, grp.full_dir, grp.early_dir)
    final_step = ts.get("total_update_steps")
    if final_step is None:
        reals = [_tag_step(t, 0) for t in merged if t not in ("base", "final")]
        final_step = (max(reals) + 1) if reals else 1
    for tag, (row, src) in merged.items():
        ck = Checkpoint(run_id=run.id, tag=tag, step=_tag_step(tag, final_step))
        session.add(ck)
        session.flush()
        ci = _ci_for_tag(data_root, src, tag)
        for mk in METRIC_KEYS:
            if row.get(mk) is None:
                continue
            lo, hi = ci.get(mk, (None, None))
            session.add(EvalResult(checkpoint_id=ck.id, metric_key=mk,
                                   value=row[mk], ci_low=lo, ci_high=hi))
        tagdir = data_root / src / "results" / tag
        if tagdir.is_dir():
            for f in sorted(tagdir.glob("*_summary.json")):
                _ensure_artifact(session, ArtifactKind.eval_detail,
                                 _rel(data_root, f), checkpoint_id=ck.id, run_id=run.id)
            for f in sorted(tagdir.glob("*.jsonl")):
                _ensure_artifact(session, ArtifactKind.eval_samples,
                                 _rel(data_root, f), checkpoint_id=ck.id, run_id=run.id)
    session.flush()

    # ── per-concept drift + audit summaries ──
    traj = ts.get("trajectory") or {}
    audit = ts.get("audit") or {}
    if not audit:  # live run: the early audit snapshot lands before train_summary.json
        audit = _read_json(data_root / full_dir / "checkpoints" / "audit.json") or {}
    for name in set(traj) | set(audit):
        concept = _ensure_concept(session, project, name, descriptions, color_counter)
        pts = traj.get(name) or []
        first = pts[0]["projection"] if pts else None
        last = pts[-1]["projection"] if pts else None
        delta = (last - first) if (first is not None and last is not None) else None
        a = audit.get(name) or {}
        session.add(RunConceptSummary(
            run_id=run.id, concept_id=concept.id,
            audit_n_flagged=a.get("n_flagged"),
            audit_threshold=a.get("threshold"),
            audit_mean_projection=a.get("mean_projection"),
            final_projection=last,
            final_probe_prob=(pts[-1].get("probe_prob") if pts else None),
            delta_projection=delta,
        ))
    session.flush()

    # ── run-level artifacts (the series layer + provenance pointers) ──
    ts_path = data_root / full_dir / "checkpoints" / "train_summary.json"
    if ts_path.exists():
        _ensure_artifact(session, ArtifactKind.train_summary,
                         _rel(data_root, ts_path), run_id=run.id)
    state = _last_trainer_state(data_root / full_dir / "checkpoints")
    if state is not None:
        _ensure_artifact(session, ArtifactKind.trainer_state,
                         _rel(data_root, state), run_id=run.id)
    summ = data_root / full_dir / "results" / "summary.json"
    if summ.exists():
        _ensure_artifact(session, ArtifactKind.eval_summary,
                         _rel(data_root, summ), run_id=run.id)
    if cfg_path is not None and cfg_path.exists():
        _ensure_artifact(session, ArtifactKind.app_config,
                         _rel(data_root, cfg_path), run_id=run.id)

    # live-progress pointers (present only mid/after a live run; harmless otherwise)
    prog = data_root / full_dir / "checkpoints" / "progress.jsonl"
    if prog.exists():
        _ensure_artifact(session, ArtifactKind.train_progress, _rel(data_root, prog), run_id=run.id)
    audit_json = data_root / full_dir / "checkpoints" / "audit.json"
    if audit_json.exists():
        _ensure_artifact(session, ArtifactKind.audit_json, _rel(data_root, audit_json), run_id=run.id)

    return run


def _seed_color_counter(session: Session, project_id: int) -> dict[int, int]:
    """Next free color_idx for a project, so live persist never collides with existing concepts."""
    from sqlalchemy import func

    mx = session.execute(
        select(func.max(Concept.color_idx)).filter_by(project_id=project_id)
    ).scalar()
    counter: dict[int, int] = defaultdict(int)
    counter[project_id] = (mx + 1) if mx is not None else 0
    return counter


def persist_run(
    session: Session,
    run_dir: str,
    *,
    project: Project,
    domain: str,
    base_model_id: str,
    status: RunStatus,
    logical_base: str | None = None,
    app_config_base: str | None = None,
    settings: Settings | None = None,
    descriptions: dict[str, str] | None = None,
    campaign: dict[str, dict] | None = None,
    color_counter: dict[int, int] | None = None,
    base_labels: dict[str, str] | None = None,
    commit: bool = True,
) -> Run:
    """Persist ONE on-disk run dir (`data/<run_dir>/`) into the DB, idempotently.

    The per-run half of `ingest`, callable on demand by the live JobManager watcher as
    checkpoints/eval land (each call clears+rebuilds the run's children from disk, so it
    is safe to repeat; pass `status=running` mid-run and `status=done` on completion).
    Assumes a pre-resolved `project` (scaffold its concepts/datasets at create-run time).
    """
    settings = settings or get_settings()
    data_root = Path(settings.data_root).resolve()
    repo_root = data_root.parent
    logical_base = logical_base or run_dir
    if descriptions is None:
        descriptions = _concept_descriptions(repo_root)
    if base_labels is None:
        base_labels = {bm.id: bm.label for bm in session.execute(select(BaseModel)).scalars()}
    if campaign is None:
        campaign = {}
    if color_counter is None:
        color_counter = _seed_color_counter(session, project.id)

    grp = RunGroup(logical_base=logical_base, base_model_id=base_model_id,
                   domain=domain, full_dir=run_dir)
    run = _ingest_run(session, grp, data_root, repo_root, {domain: project},
                      base_labels, campaign, descriptions, color_counter,
                      status=status, app_config_base=app_config_base)
    if commit:
        session.commit()
    return run


def _headline(ts: dict, is_steer: bool, coef, layer, app_cfg) -> str | None:
    n_ck = len(ts.get("monitor_fired_steps") or [])
    if is_steer:
        h = "preventative steer"
        if coef is not None:
            h += f" · coef {coef:g}"
        if layer is not None:
            h += f" · L{layer}"
        return h
    if ts:
        return "drift baseline (monitor only)"
    return None


def _run_config(session: Session, ts: dict, app_cfg, cfg_path: Path | None,
                repo_root: Path, safety_config_id: int | None) -> RunConfig:
    rc = RunConfig(
        safety_config_id=safety_config_id,
        n_train=ts.get("n_train"),
        total_update_steps=ts.get("total_update_steps"),
        save_every_steps=ts.get("save_every_steps"),
        epochs=ts.get("epochs"),
        app_config_path=(_rel(repo_root, cfg_path) if cfg_path else None),
    )
    if app_cfg is not None:
        lora = app_cfg.lora.lora or {}
        optim = app_cfg.lora.optim or {}
        tm = lora.get("target_modules")
        rc.lora_rank = lora.get("r")
        rc.lora_alpha = lora.get("alpha")
        rc.lora_dropout = lora.get("dropout")
        rc.target_modules = ",".join(tm) if isinstance(tm, list) else tm
        rc.learning_rate = optim.get("lr")
        rc.optim = optim or None
        if rc.epochs is None:
            rc.epochs = optim.get("epochs")
    return rc


# ── orchestration ────────────────────────────────────────────────────────────

@dataclass
class IngestReport:
    counts: dict[str, int] = field(default_factory=dict)
    runs_ingested: int = 0
    runs_skipped: list[tuple[str, str]] = field(default_factory=list)
    runs_failed: list[tuple[str, str]] = field(default_factory=list)
    runs_missing_config: list[str] = field(default_factory=list)
    drift_only: list[str] = field(default_factory=list)


def ingest(session: Session, settings: Settings | None = None) -> IngestReport:
    """Populate the DB from DATA_ROOT. Assumes the catalog is already seeded."""
    settings = settings or get_settings()
    data_root = Path(settings.data_root).resolve()
    repo_root = data_root.parent
    report = IngestReport()

    if not data_root.is_dir():
        log.warning("DATA_ROOT %s does not exist — nothing to ingest", data_root)
        return report

    descriptions = _concept_descriptions(repo_root)
    base_labels = {bm.id: bm.label for bm in session.execute(select(BaseModel)).scalars()}
    campaign = _load_campaign(data_root)
    color_counter: dict[int, int] = defaultdict(int)

    # 1) projects + datasets + concepts + concept vectors
    projects: dict[str, Project] = {}
    for domain in DOMAINS:
        projects[domain] = _ingest_project(
            session, domain, data_root, repo_root, descriptions, color_counter
        )

    # 2) runs (each in its own transaction; log + continue on failure)
    groups, skipped = _discover_runs(data_root, repo_root)
    report.runs_skipped = skipped
    for grp in sorted(groups, key=lambda g: (g.domain, g.logical_base, g.base_model_id)):
        try:
            _ingest_run(session, grp, data_root, repo_root, projects,
                        base_labels, campaign, descriptions, color_counter)
            session.commit()
            report.runs_ingested += 1
            if _load_app_config(repo_root, grp.logical_base)[0] is None:
                report.runs_missing_config.append(grp.full_dir or grp.early_dir)
        except Exception as exc:  # noqa: BLE001 — isolate a bad run, keep the batch
            session.rollback()
            log.exception("run ingest failed: %s", grp.full_dir or grp.early_dir)
            report.runs_failed.append((grp.full_dir or grp.early_dir, str(exc)))

    # 3) assertions / flags
    for run in session.execute(select(Run)).scalars():
        if not run.checkpoints:
            report.drift_only.append(run.dir_slug)

    report.counts = _table_counts(session)
    return report


def _table_counts(session: Session) -> dict[str, int]:
    models = {
        "project": Project, "dataset": Dataset, "concept": Concept,
        "concept_vector": ConceptVector, "run": Run, "run_config": RunConfig,
        "safety_config": SafetyConfig, "safety_config_vector": SafetyConfigVector,
        "checkpoint": Checkpoint, "eval_result": EvalResult,
        "run_concept_summary": RunConceptSummary, "artifact": Artifact,
    }
    return {name: session.query(m).count() for name, m in models.items()}


def _print_report(report: IngestReport) -> None:
    print("\n── ingest report ─────────────────────────────────────────")
    print(f"runs ingested : {report.runs_ingested}")
    print(f"runs skipped  : {len(report.runs_skipped)}")
    for name, why in report.runs_skipped:
        print(f"    - {name}: {why}")
    if report.runs_missing_config:
        print(f"runs w/o config ({len(report.runs_missing_config)}): "
              f"{', '.join(report.runs_missing_config)}")
    if report.drift_only:
        print(f"drift-only runs (0 checkpoints): {', '.join(report.drift_only)}")
    if report.runs_failed:
        print(f"runs FAILED ({len(report.runs_failed)}):")
        for name, err in report.runs_failed:
            print(f"    - {name}: {err}")
    print("table counts:")
    for name, n in report.counts.items():
        print(f"    {name:<22} {n}")
    print("──────────────────────────────────────────────────────────")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Ingest data/ + configs/ into the FTMI UI DB.")
    parser.add_argument("--drop", action="store_true", help="drop + recreate all tables first")
    args = parser.parse_args()

    settings = get_settings()
    print(f"database_url : {settings.database_url}")
    print(f"data_root    : {settings.data_root}")

    import ui_backend.models  # noqa: F401 — register mappers

    if args.drop:
        print("dropping all tables…")
        Base.metadata.drop_all(bind=engine)
    create_all()

    with SessionLocal() as session:
        print("seeding catalog…")
        seed_catalog(session)
        print("ingesting data/ …")
        report = ingest(session, settings)
    _print_report(report)


if __name__ == "__main__":
    main()
