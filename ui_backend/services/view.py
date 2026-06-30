"""RunViewService — assemble the front-end view bundle for one run.

Composes the replay PipelineService (concepts/audit/curves/steer) with a
flagged-aware dataset preview (head rows + a few of the audit-flagged rows, so the
red rows are actually visible in the small preview).
"""
from __future__ import annotations

import json

from sqlalchemy.orm import Session

from ui_backend.core.config import Settings, get_settings
from ui_backend.models.enums import ArtifactKind
from ui_backend.repositories import ArtifactRepository, RunRepository
from ui_backend.schemas.view import RunHeader, RunViewBundle, ViewDataset, ViewDatasetRow
from ui_backend.services.exceptions import NotFoundError
from ui_backend.services.pipeline import PipelineService

_PREVIEW_HEAD = 120   # show enough head rows to fill the dataset viewer (the rest scroll)
_PREVIEW_FLAGGED = 4  # plus up to N flagged rows so red is visible


class RunViewService:
    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.runs = RunRepository(session)
        self.artifacts = ArtifactRepository(session)
        self.pipe = PipelineService(session, mode="replay", settings=self.settings)

    def get_bundle(self, run_id: int) -> RunViewBundle:
        run = self.runs.get(run_id)
        if run is None:
            raise NotFoundError("run", run_id)

        concepts = self.pipe.propose_concepts(run_id)
        audit = self.pipe.audit(run_id)
        curves = self.pipe.train(run_id)
        try:
            steer = self.pipe.steer(run_id)
        except NotFoundError:
            steer = None  # no curated steered counterpart for this run

        header = RunHeader(
            run_id=run.id,
            title=run.title,
            project=run.project.name,
            domain=run.project.domain,
            model_id=run.base_model_id,
            model_label=run.base_model.label,
            early_stop_step=run.early_stop_step,
            headline=run.headline,
            canonical=run.canonical,
        )
        dataset = self._dataset(run, audit)
        return RunViewBundle(header=header, concepts=concepts, dataset=dataset,
                             audit=audit, curves=curves, steer=steer)

    # ── flagged-aware dataset preview ────────────────────────────────────────
    def _dataset(self, run, audit) -> ViewDataset:
        flagged_by_idx: dict[int, list[str]] = {}
        for c in audit.concepts:
            for i in c.flagged_idx:
                flagged_by_idx.setdefault(int(i), []).append(c.concept)

        # pick row indices: the head + the first few flagged, in order, unique
        order: list[int] = []
        for i in list(range(_PREVIEW_HEAD)) + sorted(flagged_by_idx)[:_PREVIEW_FLAGGED]:
            if i not in order:
                order.append(i)

        wanted = set(order)
        by_idx: dict[int, dict] = {}
        art = self.artifacts.find(dataset_id=run.dataset_id, kind=ArtifactKind.sft_dataset)
        if art is not None:
            path = self.settings.data_root / art.rel_path
            if path.exists():
                with path.open() as f:
                    for idx, line in enumerate(f):
                        if idx in wanted:
                            by_idx[idx] = self._row(line, flagged_by_idx.get(idx, []))
                        if idx > max(wanted, default=-1):
                            break

        rows = [by_idx[i] for i in order if i in by_idx]
        return ViewDataset(
            domain=run.project.domain,
            n_rows=run.dataset.n_rows,
            percentile=audit.percentile,
            columns=[{"key": "user", "label": "user"}, {"key": "assistant", "label": "assistant"}],
            rows=[ViewDatasetRow(**r) for r in rows],
        )

    @staticmethod
    def _row(line: str, flagged: list[str]) -> dict:
        row = {"user": None, "assistant": None, "flagged": flagged}
        try:
            for m in json.loads(line).get("messages", []):
                r = m.get("role")
                if r in ("user", "assistant") and not row[r]:
                    row[r] = (m.get("content") or "").strip()
        except json.JSONDecodeError:
            pass
        return row
