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

from ftmi.vectors.judge import judge_response


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


def _keep(polarity, trait, coherence, pos_threshold, neg_threshold, coherence_min) -> bool:
    """Chen keep-rule: coherent, and pos scores high / neg scores low on the trait."""
    if trait is None or coherence is None or coherence < coherence_min:
        return False
    return trait > pos_threshold if polarity == "pos" else trait < neg_threshold


def fit_vector(name, artifacts, model, judge, *, rollouts=5, max_new_tokens=128,
               temperature=1.0, seed=0, pos_threshold=50, neg_threshold=50,
               coherence_min=50, select_layer=None) -> PersonaVector:
    """End-to-end Chen fit: generate -> judge-filter -> pool response tokens -> diff-of-means.

    On the BASE `model` (a LocalModel), for each contrastive system-prompt pair and each
    extraction question, sample `rollouts` responses under the positive and negative
    instruction; keep coherent responses scoring pos>`pos_threshold` / neg<`neg_threshold`
    (`judge` via vectors/judge.py); pool the residual stream over RESPONSE tokens at every
    decoder layer; return the per-layer unit-normalised difference-of-means.

    `select_layer` defaults to a provisional mid-network layer; the *validated* steering
    layer comes from vectors/validate.py (the §4 dose-response gate) — never trust the
    default for steering.
    """
    pooled: dict[str, list[np.ndarray]] = {"pos": [], "neg": []}
    sid = 0
    for pair in artifacts.system_prompts:
        for polarity in ("pos", "neg"):
            system = pair[polarity]
            for question in artifacts.extraction_questions:
                for _ in range(rollouts):
                    resp_ids, text = model.generate(
                        system, question, max_new_tokens=max_new_tokens,
                        temperature=temperature, seed=seed + sid)  # this just calls the model
                    sid += 1
                    if not resp_ids:
                        continue
                    trait, coherence = judge_response(judge, artifacts.judge_prompt, question, text)
                    if not _keep(polarity, trait, coherence, pos_threshold, neg_threshold, coherence_min):
                        continue
                    pooled[polarity].append(model.pooled_response(system, question, resp_ids))
    if not pooled["pos"] or not pooled["neg"]:
        raise ValueError(
            f"0 kept on a side (pos={len(pooled['pos'])}, neg={len(pooled['neg'])}) — "
            "cannot fit. Loosen pos/neg thresholds, raise rollouts, or check that the "
            "artifacts actually elicit judge-detectable trait expression.")
    pos = np.stack(pooled["pos"])
    neg = np.stack(pooled["neg"])
    # Provisional layer = mid-network (docs/vector-steering.md §1). Peak pre-norm is NOT
    # used: residual norm grows monotonically with depth, so it degenerately picks the
    # last layer (a poor steering site). validate.py selects the real layer.
    layer = pos.shape[1] // 2 if select_layer is None else int(select_layer)
    return fit_from_pooled(name, pos, neg, layer)
