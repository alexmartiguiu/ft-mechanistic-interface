"""Pure-function seams are unit-testable without a model — the point of separating
fit_from_pooled / projection math from generation I/O.
"""
import numpy as np

from ftmi.vectors.extract import fit_from_pooled
from ftmi.vectors.monitor import projection_difference, score_generations


def test_fit_from_pooled_recovers_direction():
    rng = np.random.default_rng(0)
    hidden, n_layers = 32, 4
    direction = rng.normal(size=hidden)
    direction /= np.linalg.norm(direction)
    pos = rng.normal(size=(50, n_layers, hidden)) + 5 * direction
    neg = rng.normal(size=(50, n_layers, hidden)) - 5 * direction
    pv = fit_from_pooled("x", pos, neg, layer=2)
    assert pv.v.shape == (n_layers, hidden)
    np.testing.assert_allclose(np.linalg.norm(pv.unit()), 1.0, atol=1e-5)
    assert abs(np.dot(pv.unit(), direction)) > 0.9  # aligned with the planted axis


def test_score_generations_separable_is_high_auc():
    clean = np.zeros(100)
    drifted = np.ones(100)
    assert score_generations(clean, drifted) == 1.0


def test_projection_difference_sign():
    assert projection_difference(np.array([2.0, 3.0]), np.array([0.0, 0.0])) > 0
