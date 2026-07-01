"""ConfigAuthoringService — the LIVE-mode config workspace behind dev mode.

A live `Project` owns a small on-disk config workspace at

    data/_projects/<project_id>/configs/{app,concepts,lora}.yaml

holding the three coupled YAMLs the ftmi pipeline consumes:

  • app.yaml       — the ApplicationConfig (references the two below by *repo-root-relative*
                     path, so `ftmi run --app …` resolves them from CWD=REPO_ROOT).
  • concepts.yaml  — the ConceptSet: {domain, concepts:[{name, description}]}.
  • lora.yaml      — the LoRA recipe (model_id + hyperparams).

For an *existing* topic the workspace is seeded from the curated `configs/` templates and
reuses that domain's dataset + pre-minted vectors; the agent (or the user, in dev mode)
then trims concepts / tweaks the recipe. Nothing launches until `status().launchable`.

Physical files are project-scoped so they never collide with the curated repo configs; the
editor *displays* them under the familiar `configs/{applications,concepts,lora}/` grouping
(display only — see `ConfigFile.label` vs `ConfigFile.path`).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from ftmi.config import ApplicationConfig, Concept, ConceptSet, LoraConfig
from ui_backend.core.config import Settings, get_settings
from ui_backend.db.ingest import MODEL_BY_HF, SLUG_BY_MODEL, _chdir
from ui_backend.models.catalog import BaseModel as BaseModelRow
from ui_backend.models.project import Dataset, Project
from ui_backend.models.run import Run
from ui_backend.schemas.config import (
    ConfigCheck,
    ConfigFile,
    ConfigMutation,
    ConfigStatus,
    ConfigTree,
)
from ui_backend.services.exceptions import NotFoundError, ValidationError

# kind → physical filename in the workspace
_FILES = {"application": "app.yaml", "concepts": "concepts.yaml", "lora": "lora.yaml"}
# kind → curated dir it maps to (display grouping in the editor tree)
_DISPLAY_DIR = {"application": "configs/applications", "concepts": "configs/concepts",
                "lora": "configs/lora"}
# tree/display order
_ORDER = ["application", "concepts", "lora"]
# default LoRA recipe per model family (per-family target_modules come along)
_DEFAULT_LORA = {"qwen-7b": "qwen7b_default", "apertus-8b": "apertus8b_default"}


def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-") or "project"


class ConfigAuthoringService:
    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.data_root = Path(self.settings.data_root).resolve()
        self.repo_root = self.data_root.parent

    # ── identity / paths ──────────────────────────────────────────────────────
    def _project(self, project_id: int) -> Project:
        p = self.session.get(Project, project_id)
        if p is None:
            raise NotFoundError("project", project_id)
        return p

    def workspace(self, project: Project) -> Path:
        return self.data_root / "_projects" / str(project.id) / "configs"

    def _abs(self, project: Project, kind: str) -> Path:
        return self.workspace(project) / _FILES[kind]

    def _rel(self, project: Project, kind: str) -> str:
        """Repo-root-relative path — exactly what goes inside app.yaml + on the cmd line."""
        return self._abs(project, kind).relative_to(self.repo_root).as_posix()

    def _slug(self, project: Project) -> str:
        return project.domain or _slugify(project.name)

    def _label(self, project: Project, kind: str) -> str:
        return f"{_DISPLAY_DIR[kind]}/{self._slug(project)}.yaml"

    # ── model / vectors resolution ────────────────────────────────────────────
    def _hf_repo(self, model_id: str) -> str | None:
        bm = self.session.get(BaseModelRow, model_id)
        return bm.hf_repo if bm else None

    def _model_id(self, project: Project) -> str | None:
        """Chosen base model, encoded in the authored lora.yaml (model_id → base_model.id)."""
        d = self._load(project, "lora") or {}
        return MODEL_BY_HF.get(d.get("model_id", ""))

    def vectors_dir(self, project: Project) -> str:
        """Where the run's concept vectors live (existing topic → curated per-model dir)."""
        model_id = self._model_id(project) or "qwen-7b"
        slug = SLUG_BY_MODEL.get(model_id, "")
        return f"data/{project.domain}/vectors{slug}"

    # ── raw read/write ────────────────────────────────────────────────────────
    def _load(self, project: Project, kind: str) -> dict | None:
        p = self._abs(project, kind)
        if not p.exists():
            return None
        try:
            return yaml.safe_load(p.read_text()) or {}
        except yaml.YAMLError:
            return None

    def _write_yaml(self, project: Project, kind: str, data: dict) -> None:
        p = self._abs(project, kind)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))

    # ── seeding (idempotent — never clobber existing edits) ────────────────────
    def ensure(self, project_id: int, *, model_id: str, lora_preset: str | None = None) -> None:
        project = self._project(project_id)
        self.workspace(project).mkdir(parents=True, exist_ok=True)
        if not self._abs(project, "lora").exists():
            self._seed_lora(project, model_id, lora_preset)
        if not self._abs(project, "concepts").exists():
            self._seed_concepts(project)
        if not self._abs(project, "application").exists():
            self._seed_app(project)

    def _seed_lora(self, project: Project, model_id: str, preset: str | None) -> None:
        src = preset or _DEFAULT_LORA.get(model_id)
        path = self.repo_root / "configs" / "lora" / f"{src}.yaml" if src else None
        d = yaml.safe_load(path.read_text()) if path and path.exists() else {}
        hf = self._hf_repo(model_id)
        if hf:
            d["model_id"] = hf                       # pin the chosen base model into the recipe
        self._write_yaml(project, "lora", d)

    def _seed_concepts(self, project: Project) -> None:
        src = self.repo_root / "configs" / "concepts" / f"{project.domain}.yaml"
        d = (yaml.safe_load(src.read_text()) if src.exists()
             else {"domain": project.domain, "concepts": []})
        self._write_yaml(project, "concepts", d)

    def _seed_app(self, project: Project) -> None:
        src = self.repo_root / "configs" / "applications" / f"{project.domain}.yaml"
        d = yaml.safe_load(src.read_text()) if src.exists() else self._blank_app(project)
        d["name"] = self._slug(project)
        # rewrite the coupled refs to the project-scoped files (repo-root-relative)
        d["lora"] = self._rel(project, "lora")
        d["concepts"] = self._rel(project, "concepts")
        self._write_yaml(project, "application", d)

    def _blank_app(self, project: Project) -> dict:
        """Minimal ApplicationConfig skeleton for a topic with no curated template."""
        return {
            "name": self._slug(project),
            "data": {"path": f"data/{project.domain}/sft.jsonl",
                     "text_field": "messages", "valid_fraction": 0.05},
            "monitor": {"enabled": True, "layer": "auto"},
            "audit": {"enabled": True, "flag_percentile": 95},
            "mitigate": {"mode": "none", "coef": 0.0},
            "eval": {"enabled": True},
        }

    # ── authoring mutations (called by the agent's tools + dev-mode PUT) ────────
    def set_concepts(self, project_id: int, concepts: list[dict]) -> ConfigFile:
        """Persist the tracked concept set into concepts.yaml (name + description)."""
        project = self._project(project_id)
        cur = self._load(project, "concepts") or {"domain": project.domain}
        cur["concepts"] = [{"name": c["name"], "description": c.get("description") or ""}
                           for c in concepts]
        self._write_yaml(project, "concepts", cur)
        return self._file(project, "concepts")

    def set_lora(self, project_id: int, *, preset: str | None = None,
                 overrides: dict | None = None) -> ConfigFile:
        project = self._project(project_id)
        if preset:
            src = self.repo_root / "configs" / "lora" / f"{preset}.yaml"
            if not src.exists():
                raise ValidationError(f"unknown lora preset {preset!r}")
            d = yaml.safe_load(src.read_text())
            model_id = self._model_id(project)
            hf = self._hf_repo(model_id) if model_id else None
            if hf:
                d["model_id"] = hf
        else:
            d = self._load(project, "lora") or {}
        for k, v in (overrides or {}).items():           # shallow patch (lora/optim sub-dicts)
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                d[k] = {**d[k], **v}
            else:
                d[k] = v
        self._write_yaml(project, "lora", d)
        return self._file(project, "lora")

    def write_raw(self, project_id: int, kind: str, content: str) -> ConfigFile:
        """Two-way edit: validate the YAML parses, then persist verbatim (Phase 2)."""
        if kind not in _FILES:
            raise ValidationError(f"unknown config kind {kind!r}")
        project = self._project(project_id)
        try:
            yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ValidationError(f"invalid YAML: {e}") from e
        p = self._abs(project, kind)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return self._file(project, kind)

    # ── reads (the editor tree) ────────────────────────────────────────────────
    def _file(self, project: Project, kind: str, *, editable: bool = True) -> ConfigFile:
        p = self._abs(project, kind)
        content = p.read_text() if p.exists() else ""
        valid, err = self._validate(project, kind, content)
        return ConfigFile(kind=kind, label=self._label(project, kind), path=self._rel(project, kind),
                          content=content, editable=editable, valid=valid, error=err)

    def read_tree(self, project_id: int) -> list[ConfigFile]:
        project = self._project(project_id)
        editable = (project.mode or "replay") == "live"
        return [self._file(project, k, editable=editable)
                for k in _ORDER if self._abs(project, k).exists()]

    def tree(self, project_id: int) -> ConfigTree:
        project = self._project(project_id)
        return ConfigTree(project_id=project.id, mode=project.mode or "replay",
                          files=self.read_tree(project_id))

    def mutation(self, project_id: int, file: ConfigFile) -> ConfigMutation:
        """Wrap an authoring write with the recomputed launch gate (for the SSE/PUT reply)."""
        return ConfigMutation(file=file, status=self.status(project_id))

    def run_tree(self, run_id: int) -> ConfigTree:
        run = self.session.get(Run, run_id)
        if run is None:
            raise NotFoundError("run", run_id)
        return ConfigTree(project_id=run.project_id, mode="replay",
                          files=self.read_run_tree(run_id))

    def read_run_tree(self, run_id: int) -> list[ConfigFile]:
        """Read-only config tree for ANY run, resolved from its recorded app config.

        Replay runs (and already-launched live runs) point at a real app YAML via
        RunConfig.app_config_path (or Run.argv → app_config_base, else the curated
        domain default). We read that app file + the two it references verbatim, so
        the editor shows "the YAML as it is". Never editable."""
        run = self.session.get(Run, run_id)
        if run is None:
            raise NotFoundError("run", run_id)

        app_rel = None
        if run.config and run.config.app_config_path:
            app_rel = run.config.app_config_path
        elif run.argv:
            try:
                app_rel = (json.loads(run.argv) or {}).get("app_config_base")
            except json.JSONDecodeError:
                app_rel = None
        if not app_rel:
            app_rel = f"configs/applications/{run.project.domain}.yaml"

        app_path = self.repo_root / app_rel
        if not app_path.exists():
            return []
        raw = yaml.safe_load(app_path.read_text()) or {}
        files = [ConfigFile(kind="application", label=app_rel, path=app_rel,
                            content=app_path.read_text(), editable=False)]
        for kind, ref in (("concepts", raw.get("concepts")), ("lora", raw.get("lora"))):
            if not ref:
                continue
            p = self.repo_root / ref
            if p.exists():
                files.append(ConfigFile(kind=kind, label=ref, path=ref,
                                        content=p.read_text(), editable=False))
        return files

    def _validate(self, project: Project, kind: str, content: str) -> tuple[bool, str | None]:
        if not content.strip():
            return False, "empty"
        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            return False, f"YAML parse error: {e}"
        try:
            with _chdir(self.repo_root):
                if kind == "concepts":
                    ConceptSet(domain=data["domain"],
                               concepts=[Concept(**c) for c in data.get("concepts", [])])
                elif kind == "lora":
                    LoraConfig(**{k: data[k] for k in
                                  ("model_id", "dtype", "lora", "optim", "checkpoint")})
                elif kind == "application":
                    ApplicationConfig.load(self._abs(project, "application"))
        except Exception as e:  # noqa: BLE001 — surface any load failure as an editor error
            return False, str(e)
        return True, None

    # ── the launch gate ────────────────────────────────────────────────────────
    def status(self, project_id: int) -> ConfigStatus:
        project = self._project(project_id)
        present = {k: self._abs(project, k).exists() for k in _FILES}
        files_present = all(present.values())

        app_cfg = None
        valid = files_present
        if files_present:
            try:
                with _chdir(self.repo_root):
                    app_cfg = ApplicationConfig.load(self._abs(project, "application"))
            except Exception:  # noqa: BLE001
                valid = False

        concepts = (self._load(project, "concepts") or {}).get("concepts") or []
        concepts_n = len(concepts)

        ds_path = (app_cfg.data or {}).get("path") if app_cfg else None
        dataset_ok = bool(ds_path) and (self.repo_root / ds_path).exists()
        dataset = ConfigCheck(ok=dataset_ok,
                              reason=None if dataset_ok else f"dataset not found: {ds_path or '(unset)'}")

        # Vectors are NOT a launch blocker: `ftmi run` mints the missing ones at launch.
        # We report which concepts still need minting so the UI can warn (it adds time).
        vec_dir = self.repo_root / self.vectors_dir(project)
        missing = [c["name"] for c in concepts if not (vec_dir / f"{c['name']}.npz").exists()]
        vectors_ok = concepts_n > 0 and not missing
        vectors = ConfigCheck(
            ok=vectors_ok,
            reason=None if vectors_ok else (
                "no concepts selected" if concepts_n == 0
                else f"{len(missing)} vector(s) will be minted at launch: {', '.join(missing)}"))

        reasons: list[str] = []
        if not files_present:
            reasons.append("configs not fully authored yet")
        elif not valid:
            reasons.append("a config file is invalid")
        if concepts_n == 0:
            reasons.append("no concepts selected to track")
        if not dataset.ok and dataset.reason:
            reasons.append(dataset.reason)

        launchable = files_present and valid and concepts_n > 0 and dataset_ok
        return ConfigStatus(launchable=launchable, files_present=files_present, valid=valid,
                            concepts_n=concepts_n, concepts=[c["name"] for c in concepts],
                            dataset=dataset, vectors=vectors, reasons=reasons)

    # ── dataset preview (project-scoped, before any run exists) ────────────────
    def current_concepts(self, project_id: int) -> list[dict]:
        """The concepts currently in concepts.yaml (seeded curated set, or agent-authored)."""
        project = self._project(project_id)
        return (self._load(project, "concepts") or {}).get("concepts") or []

    def attach_dataset(self, project_id: int, *, content: str, filename: str | None = None) -> int:
        """Validate + save an uploaded chat-JSONL dataset and repoint the app config at it.

        One format: {"messages": [{"role", "content"}, ...]} per non-empty line. The file is
        written project-scoped (data/_projects/<id>/sft.jsonl) and app.yaml's data.path is
        rewritten to it, so the launch gate's dataset check now passes. Returns the row count."""
        project = self._project(project_id)
        lines = [ln for ln in content.splitlines() if ln.strip()]
        if not lines:
            raise ValidationError("dataset is empty")
        for i, ln in enumerate(lines, 1):
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError as e:
                raise ValidationError(f"line {i}: not valid JSON ({e})") from e
            msgs = rec.get("messages")
            if not isinstance(msgs, list) or not msgs:
                raise ValidationError(f"line {i}: expected a non-empty 'messages' list")
            if not all(isinstance(m, dict) and "role" in m and "content" in m for m in msgs):
                raise ValidationError(f"line {i}: each message needs 'role' and 'content'")

        dest = self.data_root / "_projects" / str(project.id) / "sft.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join(lines) + "\n")
        rel = dest.relative_to(self.repo_root).as_posix()

        app = self._load(project, "application") or {}
        data = app.get("data") or {}
        data["path"] = rel
        data.setdefault("text_field", "messages")
        data.setdefault("valid_fraction", 0.05)
        app["data"] = data
        self._write_yaml(project, "application", app)

        ds = self.session.execute(
            select(Dataset).filter_by(project_id=project.id)
        ).scalars().first()
        if ds is None:
            ds = Dataset(project_id=project.id)
            self.session.add(ds)
        ds.label = (filename or ds.label or "uploaded")[:120]
        ds.n_rows = len(lines)
        self.session.commit()
        return len(lines)

    def dataset_preview(self, project_id: int, *, limit: int = 6):
        """First rows of the authored app config's training set — for the setup viewer."""
        from ftmi.data.loaders import load_chat_dataset

        from ui_backend.schemas.pipeline import DatasetPreview

        project = self._project(project_id)
        cols = [{"key": "user", "label": "User"}, {"key": "assistant", "label": "Assistant"}]
        rel = ((self._load(project, "application") or {}).get("data") or {}).get("path")
        path = self.repo_root / rel if rel else None
        if not path or not path.exists():
            return DatasetPreview(dataset_id=0, domain=project.domain, rel_path=rel,
                                  n_rows=0, columns=cols, rows=[])
        all_rows = load_chat_dataset(str(path))
        rows = []
        for r in all_rows[:limit]:
            msgs = r.get("messages") or []
            rows.append({
                "user": next((m.get("content", "") for m in msgs if m.get("role") == "user"), ""),
                "assistant": next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), ""),
            })
        return DatasetPreview(dataset_id=0, domain=project.domain, rel_path=rel,
                              n_rows=len(all_rows), columns=cols, rows=rows)

    # ── launch command (consumed by the launch endpoint) ───────────────────────
    def launch_command(self, project_id: int, run_name: str) -> dict:
        """Build the `ftmi run` argv that consumes the AUTHORED app config.

        `app_config_base` carries the authored app path (repo-root-relative) so the
        JobManager watcher's persist_run records THIS config (not a curated one), which
        is what the dev-mode replay view then reads back."""
        project = self._project(project_id)
        app_rel = self._rel(project, "application")
        vec_dir = self.vectors_dir(project)
        # NO --skip-vectors: `ftmi run` mints only the MISSING vectors (run.py:100), so an
        # existing topic with pre-minted vectors is a no-op, and a new topic gets its vectors
        # minted (Gemini generator + GPU) before training. --skip-report: the UI reads the DB.
        cmd = [sys.executable, "-m", "ftmi.cli", "run",
               "--app", app_rel, "--name", run_name,
               "--vectors", vec_dir, "--skip-report"]
        return {"cmd": cmd, "app_config_base": app_rel, "vectors": vec_dir,
                "model_id": self._model_id(project)}
