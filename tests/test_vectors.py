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


def test_resolve_save_steps_targets_n_checkpoints():
    from ftmi.train.lora import _resolve_save_steps

    # 1000 samples, effective batch 8, 3 epochs -> 375 update steps; 10 checkpoints -> 37.
    save_every, total = _resolve_save_steps(
        n_train=1000, batch_size=1, grad_accum=8, epochs=3,
        max_steps=-1, n_checkpoints=10, explicit=None)
    assert total == 375
    assert save_every == 37
    assert total // save_every in (10, 11)  # ~10 checkpoints


def test_resolve_save_steps_explicit_and_max_steps_win():
    from ftmi.train.lora import _resolve_save_steps

    # Explicit save cadence overrides the derivation.
    assert _resolve_save_steps(1000, 1, 8, 3, -1, 10, explicit=50)[0] == 50
    # max_steps fixes the total instead of the dataset-derived count.
    save_every, total = _resolve_save_steps(1000, 1, 8, 3, max_steps=100,
                                            n_checkpoints=10, explicit=None)
    assert total == 100 and save_every == 10
