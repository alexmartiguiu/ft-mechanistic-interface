"""Stage 2/3 — fit a concept vector by difference-of-means over contrastive responses.

Chen+ 2507.21509 §2.2: on the BASE model, generate responses to the extraction
questions under positive vs negative system prompts, judge-filter (keep pos>50,
neg<50, coherent), pool the residual stream over RESPONSE tokens at every layer, and
take v[l] = mean(pos) - mean(neg), unit-normalised per layer. The steering layer is
selected empirically by dose-response effectiveness, not fixed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PersonaVector:
    name: str
    v: np.ndarray            # (n_layers, hidden), unit-norm per layer
    norm_pre: np.ndarray     # (n_layers,) pre-normalisation diff norm (quantum scale)
    layer: int               # selected steering layer
    n_pos: int
    n_neg: int

    def unit(self, layer: int | None = None) -> np.ndarray:
        return self.v[self.layer if layer is None else layer]

    def save(self, path: str) -> None:
        np.savez(path, name=self.name, v=self.v, norm_pre=self.norm_pre,
                 layer=self.layer, n_pos=self.n_pos, n_neg=self.n_neg)


def fit_from_pooled(name: str, pos: np.ndarray, neg: np.ndarray, layer: int) -> PersonaVector:
    """Difference-of-means from already-pooled per-response activations.

    `pos`/`neg`: (n_responses, n_layers, hidden), each row pooled over response
    tokens. Separated from extraction I/O so it is unit-testable with arrays.
    """
    diff = pos.mean(0) - neg.mean(0)                  # (n_layers, hidden)
    norm = np.linalg.norm(diff, axis=-1)
    v = diff / np.where(norm == 0, 1.0, norm)[:, None]
    return PersonaVector(name, v.astype(np.float32), norm.astype(np.float32),
                         layer, len(pos), len(neg))


def fit_vector(name, artifacts, model, judge, *, rollouts=5, select_layer=None) -> PersonaVector:
    """End-to-end fit: generate -> judge-filter -> pool -> diff-of-means.

    Stub: wire generation + response-token pooling against `model`, and the
    keep-rule against `judge` (pos>50, neg<50, coherent). `select_layer` defaults to
    the dose-response sweep (steering/validate). Returns a validated PersonaVector.
    """
    raise NotImplementedError(
        "fit_vector: implement generate -> judge-filter -> pool-response-tokens, "
        "then call fit_from_pooled() and select the layer by dose-response."
    )
