"""Stage 2/3 — fit a linear *detection* probe over the same contrastive activations.

A probe is a **read**; the steering vector (extract.py) is a **write**. Both are a
projection of the residual stream onto one direction; they differ only in how that
direction is chosen. Diff-of-means ignores activation covariance; the logistic-regression
probe whitens by it (down-weighting high-variance nuisance directions), so it is the
higher-resolution detector and outputs a calibrated probability — the right instrument
for the P1 drift monitor (docs/vector-steering.md §3a). The diff-of-means direction stays
the better causal actuator for steering (the read≠write point).

Apollo recipe [Goldowsky-Dill+ ICML 2025, "Detecting Strategic Deception with Linear
Probes"]: L2-regularised logistic regression (λ=10) on per-feature standardised
activations, layer chosen by held-out AUROC. We train on mean-pooled RESPONSE-token
activations (the robust default; same pooling as the fit), so the probe consumes exactly
the (pos, neg) arrays from extract.gather_pooled — no extra generation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ftmi.vectors.monitor import score_generations


@dataclass(frozen=True)
class Probe:
    name: str
    w: np.ndarray        # (hidden,) logistic weight on standardised activations
    b: float             # intercept
    mu: np.ndarray       # (hidden,) train-split feature mean  -- the standardisation
    sigma: np.ndarray    # (hidden,) train-split feature std   -- stored, reapplied at inference
    layer: int           # chosen by held-out AUROC
    auroc: float         # held-out AUROC at the chosen layer

    def logit(self, h: np.ndarray) -> np.ndarray:
        """Raw decision score w·z + b for raw activations `h` (..., hidden) at `layer`."""
        return ((np.asarray(h) - self.mu) / self.sigma) @ self.w + self.b

    def score(self, h: np.ndarray) -> np.ndarray:
        """Trait probability σ(w·z + b) — the per-token / per-response monitor signal."""
        return 1.0 / (1.0 + np.exp(-self.logit(h)))

    def save(self, path: str) -> None:
        np.savez(path, name=self.name, w=self.w, b=self.b, mu=self.mu,
                 sigma=self.sigma, layer=self.layer, auroc=self.auroc)

    @classmethod
    def load(cls, path: str) -> "Probe":
        d = np.load(path)
        return cls(str(d["name"]), d["w"], float(d["b"]), d["mu"], d["sigma"],
                   int(d["layer"]), float(d["auroc"]))


def _fit_logreg(x: np.ndarray, y: np.ndarray, l2: float) -> tuple[np.ndarray, float]:
    """L2 logistic regression on already-standardised `x`. sklearn's default penalty is
    L2; `C` is 1/λ (Apollo uses λ=10, i.e. strong regularisation in ~hidden-dim space)."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=1.0 / l2, max_iter=1000)
    clf.fit(x, y)
    return clf.coef_[0], float(clf.intercept_[0])


def _standardise(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x.mean(0)
    sigma = x.std(0)
    sigma[sigma == 0] = 1.0
    return mu, sigma


def fit_probe_from_pooled(name: str, pos: np.ndarray, neg: np.ndarray, *, layers=None,
                          l2: float = 10.0, test_frac: float = 0.2, seed: int = 0) -> Probe:
    """Fit an Apollo-style logistic probe; pick the layer by held-out AUROC.

    `pos`/`neg`: (n_responses, n_layers, hidden) from extract.gather_pooled (pos = trait
    present, neg = trait absent — including on-topic-but-clean hard negatives, since neg
    answers the same questions). Sweeps `layers` (default early-to-mid, ~0.3-0.7 depth),
    standardises per the train split, fits L2 logistic regression, scores held-out AUROC
    (reusing monitor.score_generations), and refits on all data at the best layer for the
    shipped probe. `auroc` is the held-out estimate at that layer.
    """
    n_layers = pos.shape[1]
    if layers is None:
        layers = sorted({min(int(n_layers * f), n_layers - 1) for f in (0.3, 0.4, 0.5, 0.6, 0.7)})
    rng = np.random.default_rng(seed)

    def _split(n: int) -> tuple[np.ndarray, np.ndarray]:
        idx = rng.permutation(n)
        k = max(1, int(n * (1 - test_frac)))
        return idx[:k], idx[k:]

    p_tr, p_te = _split(len(pos))
    n_tr, n_te = _split(len(neg))
    best = None  # (auroc, layer)
    for L in layers:
        x_tr = np.concatenate([pos[p_tr, L], neg[n_tr, L]])
        y_tr = np.concatenate([np.ones(len(p_tr)), np.zeros(len(n_tr))])
        mu, sigma = _standardise(x_tr)
        w, b = _fit_logreg((x_tr - mu) / sigma, y_tr, l2)
        s = lambda h: ((h - mu) / sigma) @ w + b  # noqa: E731
        auroc = score_generations(s(neg[n_te, L]), s(pos[p_te, L]))  # P(pos ranked > neg)
        if best is None or auroc > best[0]:
            best = (auroc, L)
    auroc, layer = best

    # Refit on ALL data at the chosen layer for the shipped probe.
    x = np.concatenate([pos[:, layer], neg[:, layer]])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    mu, sigma = _standardise(x)
    w, b = _fit_logreg((x - mu) / sigma, y, l2)
    return Probe(name, w.astype(np.float32), b, mu.astype(np.float32),
                 sigma.astype(np.float32), int(layer), float(auroc))
