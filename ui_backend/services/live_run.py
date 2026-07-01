"""LiveRunService — create the DB rows + launch command for a live (GPU) run.

Solves the run_id chicken-and-egg: a live run doesn't exist until we make it. Given
a {domain, model, lora_preset}, this ensures a live Project (mode='live') sharing the
curated domain's on-disk assets, inserts a `queued` Run, and stashes the fully-resolved
`ftmi run` command on `Run.argv` so the in-memory JobManager can launch it (and survive
not knowing the spec itself). v1 reuses curated configs + pre-minted vectors — no minting,
no YAML generation.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from ui_backend.core.config import Settings, get_settings
from ui_backend.db.ingest import (
    DOMAIN_LABELS,
    DOMAINS,
    SLUG_BY_MODEL,
    _concept_descriptions,
    _ingest_project,
)
from ui_backend.models.catalog import BaseModel
from ui_backend.models.enums import RunStatus
from ui_backend.models.project import Dataset, Project
from ui_backend.models.run import Run
from ui_backend.services.config_authoring import ConfigAuthoringService
from ui_backend.services.exceptions import NotFoundError, ValidationError
from ui_backend.services.providers.live import steered_dir_slug

# Steered-config candidates per domain, in resolution order (first existing wins).
_STEER_SUFFIXES = ("_steer", "_steered_dense", "_mitigated")

# Default LoRA recipe per model family — passed as --lora-config so a model swap (e.g.
# running a qwen-default config on apertus) gets that family's correct target_modules.
_DEFAULT_LORA = {"qwen-7b": "qwen7b_default", "apertus-8b": "apertus8b_default"}


class LiveRunService:
    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.data_root = Path(self.settings.data_root).resolve()
        self.repo_root = self.data_root.parent

    # ── create a live project (no run yet) + seed its config workspace ─────────
    def create_project(
        self, *, domain: str, model_id: str, name: str | None = None,
        lora_preset: str | None = None,
    ) -> dict:
        """Create/select the live project for a topic and seed its config workspace.

        Existing curated topic → the (single) per-domain live project, reusing that
        domain's dataset + minted vectors. New topic → a bare live project (dataset +
        vectors are attached/minted later, see Phase 3). No Run is created here: the
        user authors the configs first, then launches."""
        self._hf_repo(model_id)  # validate the model exists
        if domain in DOMAINS:
            project = self._live_project(domain)
        else:
            project = self._new_topic_project(domain, name)
        ConfigAuthoringService(self.session, self.settings).ensure(
            project.id, model_id=model_id, lora_preset=lora_preset)
        return {"project_id": project.id, "name": project.name,
                "domain": domain, "mode": "live"}

    def launch(self, project_id: int, *, name: str | None = None) -> dict:
        """Materialize the Run from the AUTHORED configs and queue it — the gate must pass.

        Mirrors create_run, but the launch command consumes data/_projects/<id>/configs/
        app.yaml (the authored config) instead of a curated one. The JobManager still
        spawns it lazily when the agent's run_audit fires."""
        auth = ConfigAuthoringService(self.session, self.settings)
        st = auth.status(project_id)
        if not st.launchable:
            raise ValidationError("configs not ready to launch: " + "; ".join(st.reasons))
        project = self.session.get(Project, project_id)
        if project is None:
            raise NotFoundError("project", project_id)
        run_name = name or f"{project.domain}_live_{int(time.time())}"
        spec = auth.launch_command(project_id, run_name)
        run = self._insert_run(project, spec["model_id"], run_name, project.domain,
                               spec["cmd"], app_config_base=spec["app_config_base"])
        return {"run_id": run.id, "name": run_name, "mode": "live"}

    # ── create ───────────────────────────────────────────────────────────────
    def create_run(
        self,
        *,
        domain: str,
        model_id: str,
        lora_preset: str | None = None,
        concepts: list[str] | None = None,
        name: str | None = None,
    ) -> dict:
        if domain not in DOMAINS:
            raise ValidationError(f"unknown domain {domain!r}")
        hf_repo = self._hf_repo(model_id)
        app_path = self._require(f"configs/applications/{domain}.yaml", "app config")
        vec_dir = self._require_vectors(domain, model_id)
        lora_arg = self._lora_path(lora_preset or _DEFAULT_LORA.get(model_id))

        project = self._live_project(domain)
        name = name or f"{domain}_live_{int(time.time())}"
        cmd = self._run_cmd(app_path, name, hf_repo, vec_dir, lora_arg)
        run = self._insert_run(project, model_id, name, domain, cmd)
        return {"run_id": run.id, "name": name, "mode": "live"}

    def ensure_steered_run(self, biased_run_id: int, *, concepts: list[str] | None = None) -> int:
        """Find or create the steered counterpart of a live biased run (idempotent)."""
        biased = self.session.get(Run, biased_run_id)
        if biased is None or not biased.dir_slug:
            raise NotFoundError("run", biased_run_id)
        steered_name = steered_dir_slug(biased.dir_slug)
        existing = self.session.execute(
            select(Run).filter_by(project_id=biased.project_id, dir_slug=steered_name)
        ).scalars().first()
        if existing is not None:
            return existing.id

        domain = biased.project.domain
        steer_base, steer_path = self._steer_config(domain)
        hf_repo = self._hf_repo(biased.base_model_id)
        vec_dir = self._require_vectors(domain, biased.base_model_id)
        lora_arg = self._lora_path(_DEFAULT_LORA.get(biased.base_model_id))
        cmd = self._run_cmd(steer_path, steered_name, hf_repo, vec_dir, lora_arg)
        run = self._insert_run(biased.project, biased.base_model_id, steered_name, domain, cmd,
                               app_config_base=steer_base)
        return run.id

    # ── helpers ──────────────────────────────────────────────────────────────
    def _live_project(self, domain: str) -> Project:
        label, _ = DOMAIN_LABELS.get(domain, (domain.title(), None))
        return _ingest_project(
            self.session, domain, self.data_root, self.repo_root,
            _concept_descriptions(self.repo_root), defaultdict(int),
            name=f"{label} (live)", mode="live",
        )

    def _new_topic_project(self, domain: str, name: str | None) -> Project:
        """A brand-new topic with no curated assets yet — a bare live project.

        Dataset attach + vector minting happen afterward (Phase 3); the config
        workspace is still seeded so the editor + agent have something to author."""
        project = Project(name=name or domain.replace("-", " ").title(),
                          domain=domain, mode="live")
        self.session.add(project)
        self.session.commit()
        return project

    def _insert_run(self, project: Project, model_id: str, name: str, domain: str,
                    cmd: list[str], *, app_config_base: str | None = None) -> Run:
        run = Run(
            project_id=project.id,
            dataset_id=self._a_dataset(project).id,
            base_model_id=model_id,
            dir_slug=name,
            title=name,
            status=RunStatus.queued,
            argv=json.dumps({"cmd": cmd, "name": name,
                             "app_config_base": app_config_base or domain}),
        )
        self.session.add(run)
        self.session.commit()
        return run

    def _run_cmd(self, app_path: str, name: str, hf_repo: str, vec_dir: str,
                 lora_arg: str | None) -> list[str]:
        cmd = [sys.executable, "-m", "ftmi.cli", "run",
               "--app", app_path, "--name", name, "--model", hf_repo,
               "--vectors", vec_dir, "--skip-vectors", "--skip-report"]
        if lora_arg:
            cmd += ["--lora-config", lora_arg]
        return cmd

    def _a_dataset(self, project: Project) -> Dataset:
        ds = self.session.execute(
            select(Dataset).filter_by(project_id=project.id)
        ).scalars().first()
        if ds is None:  # scaffolding always creates one, but stay defensive
            raise ValidationError(f"no dataset scaffolded for project {project.name!r}")
        return ds

    def _hf_repo(self, model_id: str) -> str:
        bm = self.session.get(BaseModel, model_id)
        if bm is None:
            raise ValidationError(f"unknown model {model_id!r}")
        return bm.hf_repo

    def _lora_path(self, preset: str | None) -> str | None:
        if not preset:
            return None
        return self._require(f"configs/lora/{preset}.yaml", "lora recipe")

    def _require(self, rel: str, what: str) -> str:
        if not (self.repo_root / rel).exists():
            raise ValidationError(f"{what} not found: {rel}")
        return rel

    def _require_vectors(self, domain: str, model_id: str) -> str:
        slug = SLUG_BY_MODEL.get(model_id, "")
        rel = f"data/{domain}/vectors{slug}"
        if not (self.repo_root / rel).is_dir():
            raise ValidationError(
                f"no minted vectors for {domain}/{model_id} ({rel}); minting is v2")
        return rel

    def _steer_config(self, domain: str) -> tuple[str, str]:
        for suffix in _STEER_SUFFIXES:
            base = f"{domain}{suffix}"
            rel = f"configs/applications/{base}.yaml"
            if (self.repo_root / rel).exists():
                return base, rel
        raise ValidationError(f"no steering config for domain {domain!r}")
