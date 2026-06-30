"""ReplayProvider — read a recorded run and shape it like a freshly-executed one.

Built on the existing read stack: `RunRepository` (SQL index + scalars),
`SeriesService` (slices the per-step JSON artifacts), and direct reads of the
`train_summary` audit block for flagged-row indices. No `ftmi`, no GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy.orm import Session

from ui_backend.core.config import Settings, get_settings
from ui_backend.models.enums import ArtifactKind
from ui_backend.models.run import Run
from ui_backend.repositories import ArtifactRepository, RunRepository
from ui_backend.schemas.pipeline import (
    AuditConcept,
    AuditResult,
    ConceptDrift,
    ConceptInfo,
    DatasetPreview,
    RunCurves,
    SteerComparison,
    SteerEvalRow,
    SteerLatentRow,
    SteerResult,
)
from ui_backend.services.exceptions import NotFoundError
from ui_backend.services.series import SeriesService

# The headline eval battery, in display order, with the safe direction.
BATTERY = [
    ("mmlu_pro_acc", "MMLU-Pro", "up"),
    ("truthfulqa_mc1_acc", "TruthfulQA", "up"),
    ("harmbench_refusal_v2", "HarmBench refusal", "up"),
    ("strongreject_refusal_v2", "StrongREJECT refusal", "up"),
]

# Curated biased→steered demo pairing (the headline steered variant per project ×
# model). The biased↔steered link is not an FK; resolve by the steered run's title.
# Becomes a DB flag (`run.parent_run_id` / `is_demo_steer`) in M2 if we want it data-driven.
DEMO_STEER_PAIRS = {
    ("medical", "apertus-8b"): "medical_da_steer_c360L12",
    ("gender", "qwen-7b"): "gender_steered_dense",
    ("race", "qwen-7b"): "race_steered_dense",
    ("therapist", "qwen-7b"): "therapist_steer",
    ("financial", "qwen-7b"): "financial_steer",
}


def _final_step_key(tag: str, step: int | None) -> tuple[int, int]:
    """Sort key putting `final` last at its step so it overrides a same-step checkpoint."""
    s = 0 if tag == "base" else (step if step is not None else 0)
    return (s, 1 if tag == "final" else 0)


class ReplayProvider:
    mode = "replay"

    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.runs = RunRepository(session)
        self.artifacts = ArtifactRepository(session)
        self.series = SeriesService(session, self.settings)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _run(self, run_id: int) -> Run:
        run = self.runs.get(run_id)
        if run is None:
            raise NotFoundError("run", run_id)
        return run

    def _read_json(self, rel_path: str) -> dict | None:
        path: Path = self.settings.data_root / rel_path
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _concepts(self, run: Run) -> list[ConceptInfo]:
        cs = sorted(run.project.concepts, key=lambda c: (c.color_idx if c.color_idx is not None else 0, c.name))
        return [ConceptInfo(name=c.name, description=c.description, color_idx=c.color_idx) for c in cs]

    def _eval_series(self, run: Run) -> tuple[dict[str, list[list[float]]], dict[str, float]]:
        """Per-metric [[step, value]] across checkpoints + the final-checkpoint scalars."""
        cks = sorted(run.checkpoints, key=lambda c: _final_step_key(c.tag, c.step))
        per: dict[str, dict[int, float]] = {}
        final: dict[str, float] = {}
        final_ck = None
        for ck in cks:
            step = 0 if ck.tag == "base" else (ck.step if ck.step is not None else 0)
            for er in ck.eval_results:
                if er.value is None:
                    continue
                per.setdefault(er.metric_key, {})[step] = er.value
            if ck.tag == "final":
                final_ck = ck
        final_ck = final_ck or (cks[-1] if cks else None)
        if final_ck is not None:
            final = {er.metric_key: er.value for er in final_ck.eval_results if er.value is not None}
        evals = {k: [[s, v] for s, v in sorted(pts.items())] for k, pts in per.items()}
        return evals, final

    def _final_drift(self, run: Run) -> dict[str, float]:
        """concept name -> base→final drift projection, from run_concept_summary."""
        return {
            cs.concept.name: cs.delta_projection
            for cs in run.concept_summaries
            if cs.delta_projection is not None
        }

    def _concept_drift(self, run: Run) -> list[ConceptDrift]:
        """Per-concept base→final concept-vector drift, sorted toward-risk first.

        The concept vector is built `mean(pos) - mean(neg)` (vectors/extract.py), so it
        points toward MORE of the concept, and the audit flags the HIGHEST projections
        (train/lora.py). The vector is therefore already oriented toward risk: a positive
        `delta_projection` = drifted toward the risky behaviour, negative = toward safer,
        uniformly across concepts. NO per-concept sign flip (audit_mean_projection is just
        the raw projection offset, not a direction). Probe probabilities are not used.
        """
        rows: list[ConceptDrift] = []
        for cs in run.concept_summaries:
            d = cs.delta_projection
            if d is None:
                continue
            rows.append(ConceptDrift(
                concept=cs.concept.name,
                color_idx=cs.concept.color_idx,
                delta=d,
                toward_risk=d,   # vector already points toward the concept; higher = toward risk
            ))
        rows.sort(key=lambda r: (r.toward_risk if r.toward_risk is not None else float("-inf")), reverse=True)
        return rows

    def _steer_spec(self, steered: Run) -> tuple[str | None, float | None, int | None]:
        """(concept, coef, layer) of the steered run's first steered vector."""
        cfg = steered.config.safety_config if (steered.config and steered.config.safety_config) else None
        if cfg is None:
            return None, None, None
        for v in cfg.vectors:
            if v.steered:
                name = v.concept_vector.concept.name if v.concept_vector and v.concept_vector.concept else None
                return name, v.steer_coef, v.steer_layer
        return None, None, None

    def _resolve_steered(self, biased: Run) -> Run:
        domain = biased.project.domain
        title = DEMO_STEER_PAIRS.get((domain, biased.base_model_id))
        steered = (
            self.runs.find_by_title(project_id=biased.project_id, title=title, base_model_id=biased.base_model_id)
            if title
            else None
        )
        if steered is None:
            raise NotFoundError("steered run for", f"{domain}/{biased.base_model_id}")
        return steered

    # ── provider surface ─────────────────────────────────────────────────────
    def dataset_preview(self, run_id: int, *, limit: int = 8) -> DatasetPreview:
        run = self._run(run_id)
        ds = run.dataset
        art = self.artifacts.find(dataset_id=ds.id, kind=ArtifactKind.sft_dataset)
        rows: list[dict] = []
        if art is not None:
            path = self.settings.data_root / art.rel_path
            if path.exists():
                with path.open() as f:
                    for line in f:
                        if not line.strip() or len(rows) >= limit:
                            continue
                        try:
                            msgs = json.loads(line).get("messages", [])
                        except json.JSONDecodeError:
                            continue
                        row: dict = {}
                        for m in msgs:
                            r = m.get("role")
                            if r in ("user", "assistant") and r not in row:
                                row[r] = (m.get("content") or "").strip()
                        if row:
                            rows.append(row)
        return DatasetPreview(
            dataset_id=ds.id,
            domain=run.project.domain,
            rel_path=art.rel_path if art else None,
            n_rows=ds.n_rows,
            columns=[{"key": "user", "label": "user"}, {"key": "assistant", "label": "assistant"}],
            rows=rows,
        )

    def propose_concepts(self, run_id: int) -> list[ConceptInfo]:
        return self._concepts(self._run(run_id))

    def audit(self, run_id: int, *, tracked: list[str] | None = None) -> AuditResult:
        run = self.runs.get_detail(run_id)
        if run is None:
            raise NotFoundError("run", run_id)
        art = self.artifacts.find(run_id=run_id, kind=ArtifactKind.train_summary)
        if art is not None:
            audit_block = (self._read_json(art.rel_path) or {}).get("audit", {})
        else:
            # Live run mid-audit: the standalone audit.json snapshot IS the audit block
            # (not wrapped under "audit"), and lands before train_summary.json.
            aj = self.artifacts.find(run_id=run_id, kind=ArtifactKind.audit_json)
            audit_block = (self._read_json(aj.rel_path) or {}) if aj else {}

        concepts: list[AuditConcept] = []
        flagged_union: set[int] = set()
        percentile = 95
        for cs in run.concept_summaries:
            name = cs.concept.name
            if tracked is not None and name not in tracked:
                continue
            block = audit_block.get(name, {})
            idx = [int(i) for i in block.get("flagged_idx", [])]
            flagged_union.update(idx)
            if block.get("flag_percentile") is not None:
                percentile = int(block["flag_percentile"])
            concepts.append(
                AuditConcept(
                    concept=name,
                    color_idx=cs.concept.color_idx,
                    n_flagged=cs.audit_n_flagged,
                    threshold=cs.audit_threshold,
                    mean_projection=cs.audit_mean_projection,
                    flagged_idx=idx,
                )
            )
        concepts.sort(key=lambda c: (c.color_idx if c.color_idx is not None else 0, c.concept))
        total_flagged = len(flagged_union) if flagged_union else (
            max((c.n_flagged or 0 for c in concepts), default=0)
        )
        return AuditResult(
            run_id=run_id,
            n_rows=run.dataset.n_rows,
            percentile=percentile,
            total_flagged=total_flagged,
            concepts=concepts,
        )

    def train(self, run_id: int) -> RunCurves:
        run = self.runs.get_detail(run_id)
        if run is None:
            raise NotFoundError("run", run_id)
        evals, final = self._eval_series(run)
        s = self.series.get_run_series(run_id)
        return RunCurves(
            run_id=run_id,
            title=run.title,
            base_model_id=run.base_model_id,
            early_stop_step=run.early_stop_step,
            eval=evals,
            trajectory=s.trajectory,
            concept_drift=self._concept_drift(run),
            loss_train=s.loss_train,
            loss_eval=s.loss_eval,
            final_metrics=final,
            concepts=self._concepts(run),
        )

    def steer(self, run_id: int, *, concepts: list[str] | None = None) -> SteerResult:
        biased = self.runs.get_detail(run_id)
        if biased is None:
            raise NotFoundError("run", run_id)
        steered = self._resolve_steered(biased)
        return self.build_steer_result(biased, steered)

    def build_steer_result(self, biased: Run, steered: Run) -> SteerResult:
        """Assemble the biased↔steered comparison once both runs are in the DB.

        Shared by replay (curated steered pair) and live (the freshly-trained steered run)."""
        steered_detail = self.runs.get_detail(steered.id)

        curves = self.train(steered.id)
        concept, coef, layer = self._steer_spec(steered)

        _, biased_final = self._eval_series(biased)
        _, steered_final = self._eval_series(steered_detail)
        eval_rows = []
        for key, label, good_when in BATTERY:
            u, st = biased_final.get(key), steered_final.get(key)
            delta = (st - u) if (u is not None and st is not None) else None
            improved = None if delta is None else (delta > 0 if good_when == "up" else delta < 0)
            eval_rows.append(
                SteerEvalRow(metric_key=key, label=label, good_when=good_when,
                             unsteered=u, steered=st, delta=delta, improved=improved)
            )

        bd, sd = self._final_drift(biased), self._final_drift(steered_detail)
        latent = [
            SteerLatentRow(concept=c.name, unsteered=bd.get(c.name), steered=sd.get(c.name))
            for c in self._concepts(biased)
            if c.name in bd or c.name in sd
        ]

        hb = next((r for r in eval_rows if r.metric_key == "harmbench_refusal_v2"), None)
        headline = (
            {"label": hb.label, "base": hb.unsteered, "final": hb.steered, "delta": hb.delta, "good_when": hb.good_when}
            if hb else None
        )
        comparison = SteerComparison(
            biased_run_id=biased.id,
            steered_run_id=steered.id,
            concept=concept,
            coef=coef,
            layer=layer,
            eval=eval_rows,
            latent=latent,
            headline=headline,
        )
        return SteerResult(steered_run_id=steered.id, curves=curves, comparison=comparison)

    # ── execute_* : no-ops in replay (the run already happened) ───────────────
    async def execute_audit(self, run_id: int, *, on_event=None) -> None:
        return

    async def execute_training(self, run_id: int, *, on_event=None) -> None:
        return

    async def execute_steering(self, run_id: int, *, concepts=None, on_event=None) -> None:
        return
