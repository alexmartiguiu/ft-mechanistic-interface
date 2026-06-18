"""Combine many concept directions into ONE preventative-steering direction.

The training-time mitigation in `train.lora` steers the residual stream along a single
direction `d̂` with a single coefficient `B` (the budget). This module builds `d̂` from a
set of per-concept unit directions and per-concept importance weights, and it does so in a
way that does NOT double-count concepts that point the same way.

Why decorrelation matters
--------------------------
Naively, the combined direction is `d = Σ_c w_c · û_c`. But the domain concepts are far
from orthogonal (e.g. `medical_misinformation`, `dangerous_advice` and the universal
`deception` vector share a big subspace), so a redundant cluster of k near-duplicate
directions contributes ~k times along the shared axis — the steer is dominated by whatever
the concept set happens to over-represent, not by what actually drifted.

The fix — Gram-decorrelated coefficients
----------------------------------------
Let `û_c` be unit-norm and `G_ij = ⟨û_i, û_j⟩` the cosine Gram matrix. We solve

        c = (G + εI)^{-1} w          (ε = ridge, for stability near collinearity)

then clip `c ≥ 0` (never steer a concept the wrong way) and form `d = Σ_c c_c û_c`,
`d̂ = d / ‖d‖`. The budget sets the magnitude (`B · d̂`); `c` only sets the *shape*.

Intuition — two identical directions, each with weight w:
    G = [[1,1],[1,1]],  (G+εI)^{-1} w  →  c ≈ [w/2, w/2]
so their *combined* coefficient along the shared axis is `w`, not `2w`: the duplicate is
split, not summed twice. Orthogonal directions (G = I) are untouched (`c ≈ w`). Partial
overlap interpolates between the two — the more two concepts correlate, the more each is
shrunk. The ridge ε keeps the inverse well-conditioned when G is near-singular (which is
exactly the heavily-correlated regime we care about).
"""
from __future__ import annotations

import numpy as np


def _stack_units(units) -> np.ndarray:
    U = np.stack([np.asarray(u, dtype=np.float64) for u in units])
    n = np.linalg.norm(U, axis=1, keepdims=True)
    return U / np.where(n == 0, 1.0, n)


def redundancy_weighted_coeffs(units, weights, ridge: float = 0.05) -> np.ndarray:
    """Gram-decorrelated, non-negative per-concept coefficients `c = clip((G+εI)^{-1} w, 0)`.

    `units`: iterable of (hidden,) direction vectors (need not be pre-normalised).
    `weights`: iterable of non-negative importance weights, aligned with `units`.
    Returns `c` (k,), the coefficient each concept contributes to the combined direction.
    """
    U = _stack_units(units)                       # (k, hidden), unit rows
    w = np.asarray(weights, dtype=np.float64)
    G = U @ U.T                                    # (k, k) cosine Gram (diag = 1)
    c = np.linalg.solve(G + ridge * np.eye(len(U)), w)
    return np.clip(c, 0.0, None)


def combined_direction(units, weights, ridge: float = 0.05):
    """Return `(d̂, c)`: the unit combined steering direction and its per-concept coeffs.

    `d̂` is None when every weight (or every decorrelated coeff) is zero — i.e. nothing to
    steer. Magnitude is intentionally dropped; the caller scales `d̂` by the budget.
    """
    c = redundancy_weighted_coeffs(units, weights, ridge)
    U = _stack_units(units)
    d = c @ U                                      # (hidden,)
    norm = float(np.linalg.norm(d))
    if norm == 0.0:
        return None, c
    return (d / norm).astype(np.float32), c
