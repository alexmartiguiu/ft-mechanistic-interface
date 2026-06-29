"""Projection-based uses of a fitted vector: drift monitoring + dataset audit.

All three lifecycle uses reduce to <h, v_hat> (steering/hooks.py::ProjectionReader):
  - training-time drift  : mean projection at each checkpoint -> trajectory vs base
  - inference-time detect : per-generation projection -> AUC clean-vs-drifted
  - dataset audit         : projection difference (Chen+ 2507.21509 §6)
"""
from __future__ import annotations

import numpy as np


def pooled_generations(model, prompts, *, max_new_tokens=1000, temperature=1.0, seed=0) -> np.ndarray:
    """Generate one response per (system, user) prompt; return pooled RESPONSE-token
    activations, shape (n_prompts, n_layers, hidden).

    The substrate for the inference-time detector: project these onto any v_hat at a
    layer -- `acts[:, layer, :] @ v_hat` gives per-generation <h, v_hat> -- then feed
    clean vs drifted to `score_generations`. Pooling reuses `model.pooled_response`, so
    it averages RESPONSE tokens only, matching the fit (see design-decisions.md).
    Empty generations are dropped (an empty response has nothing to pool).
    """
    outs = model.generate_batch([s for s, _ in prompts], [u for _, u in prompts],
                                max_new_tokens=max_new_tokens, temperature=temperature, seed=seed)
    return np.stack([model.pooled_response(s, u, rid)
                     for (s, u), (rid, _txt) in zip(prompts, outs) if rid])


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


def aligned_correlation(series_a: dict, series_b: dict) -> dict:
    """Correlate two step→value trajectories on their common steps (Stage 3.5 check).

    The load-bearing monitor validation docs/vector-steering.md §6 calls for: does the
    *internal* drift signal (monitor projection / probe_prob, keyed by training step) track
    the *behavioural* drift (judged-trait per checkpoint, same keys)? Returns Pearson and
    Spearman (Spearman = Pearson of ranks, so no scipy) over the intersecting steps, plus
    the common-step series so a caller can plot them. NaN if fewer than 2 shared steps or
    either side is constant. Both are computed on internal representation vs behaviour —
    this is the test that ties them together.
    """
    steps = sorted(set(series_a) & set(series_b))
    a = np.array([series_a[s] for s in steps], dtype=float)
    b = np.array([series_b[s] for s in steps], dtype=float)
    out = {"n": len(steps), "steps": steps, "a": a.tolist(), "b": b.tolist(),
           "pearson": float("nan"), "spearman": float("nan")}
    if len(steps) < 2:
        return out
    out["pearson"] = _pearson(a, b)
    out["spearman"] = _pearson(_rank(a), _rank(b))
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared), so Spearman handles repeated values correctly."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # average tied ranks
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]
