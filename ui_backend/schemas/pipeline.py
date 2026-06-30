"""Pipeline result contracts (M1.2).

The typed shapes the agent's pipeline tools return — identical whether produced
by the ReplayProvider (read recorded runs) or, later, the LiveProvider (execute
`ftmi`). The agent and the front-end never branch on which one produced them.
"""
from __future__ import annotations

from ui_backend.schemas.common import ORMModel
from ui_backend.schemas.series import ConceptTrajectory


class ConceptInfo(ORMModel):
    name: str
    description: str | None = None
    color_idx: int | None = None


class DatasetPreview(ORMModel):
    dataset_id: int
    domain: str | None = None
    rel_path: str | None = None
    n_rows: int | None = None
    columns: list[dict] = []   # [{key, label}]
    rows: list[dict] = []      # [{user, assistant}, ...]


class AuditConcept(ORMModel):
    concept: str
    color_idx: int | None = None
    n_flagged: int | None = None
    threshold: float | None = None
    mean_projection: float | None = None
    flagged_idx: list[int] = []


class AuditResult(ORMModel):
    run_id: int
    n_rows: int | None = None
    percentile: int = 95
    total_flagged: int | None = None
    concepts: list[AuditConcept] = []


class ConceptDrift(ORMModel):
    """Base→final concept-vector projection drift for ONE concept.

    `toward_risk` aligns the raw projection change to the concept's risky
    direction (sign of its audit mean projection), so higher = drifted further
    toward the risky behaviour regardless of the vector's arbitrary sign. This is
    the interpretability read; we do not surface probe probabilities here.
    """

    concept: str
    color_idx: int | None = None
    delta: float | None = None         # raw final - base projection change
    toward_risk: float | None = None   # delta aligned to risky direction; higher = worse


class RunCurves(ORMModel):
    """Everything the insights plots need for ONE run (biased or steered)."""

    run_id: int
    title: str | None = None
    base_model_id: str | None = None
    early_stop_step: int | None = None
    eval: dict[str, list[list[float]]] = {}     # metric_key -> [[step, value], ...]
    trajectory: list[ConceptTrajectory] = []    # per-concept projection+probe over steps
    concept_drift: list[ConceptDrift] = []      # base→final concept-vector drift, worst first
    loss_train: list[list[float]] = []
    loss_eval: list[list[float]] = []
    final_metrics: dict[str, float] = {}        # metric_key -> final value
    concepts: list[ConceptInfo] = []


class SteerEvalRow(ORMModel):
    metric_key: str
    label: str | None = None
    good_when: str | None = None
    unsteered: float | None = None
    steered: float | None = None
    delta: float | None = None
    improved: bool | None = None


class SteerLatentRow(ORMModel):
    concept: str
    unsteered: float | None = None   # biased run base→final drift projection
    steered: float | None = None     # steered run base→final drift projection


class SteerComparison(ORMModel):
    biased_run_id: int
    steered_run_id: int
    concept: str | None = None
    coef: float | None = None
    layer: int | None = None
    eval: list[SteerEvalRow] = []
    latent: list[SteerLatentRow] = []
    headline: dict | None = None     # {label, base, final, delta, good_when}


class SteerResult(ORMModel):
    """The mitigation payload: the steered run's curves + the biased↔steered diff."""

    steered_run_id: int
    curves: RunCurves
    comparison: SteerComparison


__all__ = [
    "ConceptInfo",
    "DatasetPreview",
    "AuditConcept",
    "AuditResult",
    "ConceptDrift",
    "RunCurves",
    "SteerEvalRow",
    "SteerLatentRow",
    "SteerComparison",
    "SteerResult",
]
