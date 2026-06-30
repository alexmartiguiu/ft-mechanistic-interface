"""Per-step series service.

This is the JSON-filtering layer: SQL holds only the *pointer* (an `artifact`
row) and final scalars; the full per-step curves are read out of the infra's
JSON on demand and sliced here. Nothing per-step is stored in the database.

If the artifact row or the file is missing, we return empty series rather than
erroring — the run summary in SQL is still valid.
"""
from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy.orm import Session

from ui_backend.core.config import Settings, get_settings
from ui_backend.models.enums import ArtifactKind
from ui_backend.repositories import ArtifactRepository, RunRepository
from ui_backend.schemas.series import ConceptTrajectory, RunSeries, TrajectoryPoint
from ui_backend.services.exceptions import NotFoundError


class SeriesService:
    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.runs = RunRepository(session)
        self.artifacts = ArtifactRepository(session)

    def _resolve(self, rel_path: str) -> Path:
        return self.settings.data_root / rel_path

    def _read_json(self, rel_path: str) -> dict | None:
        path = self._resolve(rel_path)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def get_run_series(
        self,
        run_id: int,
        *,
        concept: str | None = None,
        step_min: int | None = None,
        step_max: int | None = None,
    ) -> RunSeries:
        if self.runs.get(run_id) is None:
            raise NotFoundError("run", run_id)

        out = RunSeries(run_id=run_id)

        # ── drift trajectory ← train_summary.json ──────────────────────────────
        traj_artifact = self.artifacts.find(run_id=run_id, kind=ArtifactKind.train_summary)
        traj_by_concept: dict[str, list[dict]] = {}
        if traj_artifact is not None:
            out.source = traj_artifact.rel_path
            data = self._read_json(traj_artifact.rel_path) or {}
            traj_by_concept = {k: list(v) for k, v in (data.get("trajectory") or {}).items()}
        else:
            # Live run mid-training: train_summary.json is written only at the end, so
            # fold the streamed progress.jsonl into the same {concept: [entries]} shape.
            prog = self.artifacts.find(run_id=run_id, kind=ArtifactKind.train_progress)
            if prog is not None:
                out.source = prog.rel_path
                traj_by_concept = self._progress_trajectory(prog.rel_path)

        for name, seq in traj_by_concept.items():
            if concept is not None and name != concept:
                continue
            points = [
                TrajectoryPoint(
                    step=e["step"],
                    projection=e.get("projection"),
                    probe_prob=e.get("probe_prob"),
                )
                for e in seq
                if _in_range(e.get("step"), step_min, step_max)
            ]
            out.trajectory.append(ConceptTrajectory(concept=name, points=points))

        # ── loss curves ← trainer_state.json (log_history) ─────────────────────
        loss_artifact = self.artifacts.find(run_id=run_id, kind=ArtifactKind.trainer_state)
        if loss_artifact is not None:
            data = self._read_json(loss_artifact.rel_path) or {}
            for entry in data.get("log_history", []):
                step = entry.get("step")
                if not _in_range(step, step_min, step_max):
                    continue
                if "loss" in entry and "eval_loss" not in entry:
                    out.loss_train.append([step, entry["loss"]])
                if "eval_loss" in entry:
                    out.loss_eval.append([step, entry["eval_loss"]])

        return out

    def _progress_trajectory(self, rel_path: str) -> dict[str, list[dict]]:
        """Parse the live progress.jsonl (one {concept,step,projection,probe_prob} per line)
        into {concept: [entries]}, tolerating a trailing partial line still being written."""
        path = self._resolve(rel_path)
        if not path.exists():
            return {}
        out: dict[str, list[dict]] = {}
        try:
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue  # trailing fragment of a line mid-flush
                name = e.get("concept")
                if name is not None and e.get("step") is not None:
                    out.setdefault(name, []).append(e)
        except OSError:
            return {}
        return out


def _in_range(step: int | None, lo: int | None, hi: int | None) -> bool:
    if step is None:
        return False
    if lo is not None and step < lo:
        return False
    if hi is not None and step > hi:
        return False
    return True
