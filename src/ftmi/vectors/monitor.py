"""Projection-based uses of a fitted vector: drift monitoring + dataset audit.

All three lifecycle uses reduce to <h, v_hat> (steering/hooks.py::ProjectionReader):
  - training-time drift  : mean projection at each checkpoint -> trajectory vs base
  - inference-time detect : per-generation projection -> AUC clean-vs-drifted
  - dataset audit         : projection difference (Chen+ 2507.21509 §6)
"""
from __future__ import annotations

import numpy as np


def projection_difference(
    dataset_proj: np.ndarray, base_proj: np.ndarray
) -> float:
    """Chen+ 2507.21509 §6 projection difference dP, the pre-finetune data-audit signal.

    `dataset_proj`: mean <h, v_hat> over each dataset response.
    `base_proj`:    mean <h, v_hat> over the BASE model's own response to the same
                    prompt (controls for what the base would have said anyway).
    Returns the dataset-level mean dP; large dP predicts post-finetune trait shift.
    """
    return float(np.mean(dataset_proj - base_proj))


def flag_samples(
    dataset_proj: np.ndarray, base_proj: np.ndarray, percentile: float
) -> np.ndarray:
    """Per-sample audit: indices whose projection difference exceeds `percentile`.

    Feeds the flag / clean / resample path on dataset drop.
    """
    dp = dataset_proj - base_proj
    return np.where(dp > np.percentile(dp, percentile))[0]


def score_generations(proj_clean: np.ndarray, proj_drifted: np.ndarray) -> float:
    """Inference-time detector quality: AUC of <h, v_hat> separating clean vs drifted.

    Rank-statistic AUC (no sklearn dependency). Report alongside a random-direction
    floor and a benign hard-negative FPR (docs/vector-steering.md §5).
    """
    c, d = np.asarray(proj_clean), np.asarray(proj_drifted)
    if c.size == 0 or d.size == 0:
        return float("nan")
    greater = (d[:, None] > c[None, :]).sum()
    ties = (d[:, None] == c[None, :]).sum()
    return float((greater + 0.5 * ties) / (c.size * d.size))
