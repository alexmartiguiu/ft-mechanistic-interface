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

from ftmi.vectors.judge import judge_batch


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

    @classmethod
    def load(cls, path: str) -> "PersonaVector":
        d = np.load(path)
        return cls(str(d["name"]), d["v"], d["norm_pre"], int(d["layer"]),
                   int(d["n_pos"]), int(d["n_neg"]))


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


def gather_pooled(artifacts, model, judge, *, rollouts=5, max_new_tokens=1000,
                  temperature=1.0, seed=0, pos_threshold=50, neg_threshold=50,
                  coherence_min=50, batch_size=32, judge_concurrency=16
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Generate -> judge-filter -> pool RESPONSE tokens; return (pos, neg) activations.

    The shared front half of every concept artefact, on the BASE `model` (a LocalModel):
    (1) batched generation of `rollouts` responses per (contrastive system prompt,
    extraction question); (2) parallel judging for trait + coherence (`judge` via
    vectors/judge.py); (3) keep coherent responses scoring pos>`pos_threshold` /
    neg<`neg_threshold`, pool the residual stream over RESPONSE tokens at every layer.

    Returns `pos`, `neg` each (n_responses, n_layers, hidden). This labelled contrastive
    set feeds BOTH operations on the residual stream: diff-of-means (`fit_from_pooled`,
    the steering *write*) and the logistic probe (`probe.fit_probe_from_pooled`, the
    detection *read*) — same data, different fit.
    """
    qs = artifacts.extraction_questions
    # 1. GENERATE — batch over questions, for each (system prompt × polarity × rollout).
    records = []  # (polarity, system, question, resp_ids, text)
    vi = 0
    for pair in artifacts.system_prompts:
        for polarity in ("pos", "neg"):
            system = pair[polarity]
            for r in range(rollouts):
                for i in range(0, len(qs), batch_size):
                    chunk = qs[i:i + batch_size]
                    outs = model.generate_batch(
                        [system] * len(chunk), chunk, max_new_tokens=max_new_tokens,
                        temperature=temperature, seed=seed + 10000 * vi + 100 * r + i)
                    records.extend((polarity, system, q, rid, txt)
                                   for q, (rid, txt) in zip(chunk, outs) if rid)
            vi += 1
    # 2. JUDGE — in parallel.
    scores = judge_batch(judge, artifacts.judge_prompt,
                         [(rec[2], rec[4]) for rec in records], concurrency=judge_concurrency)
    # 3. FILTER + POOL.
    pooled: dict[str, list[np.ndarray]] = {"pos": [], "neg": []}
    for (polarity, system, question, resp_ids, _text), (trait, coherence) in zip(records, scores):
        if _keep(polarity, trait, coherence, pos_threshold, neg_threshold, coherence_min):
            pooled[polarity].append(model.pooled_response(system, question, resp_ids))
    if not pooled["pos"] or not pooled["neg"]:
        raise ValueError(
            f"0 kept on a side (pos={len(pooled['pos'])}, neg={len(pooled['neg'])}) — "
            "cannot fit. Loosen pos/neg thresholds, raise rollouts, or check that the "
            "artifacts actually elicit judge-detectable trait expression.")
    return np.stack(pooled["pos"]), np.stack(pooled["neg"])


def provisional_layer(n_layers: int) -> int:
    """Mid-network default (docs/vector-steering.md §1). Peak pre-norm is NOT used:
    residual norm grows monotonically with depth, so it degenerately picks the last
    layer (a poor steering site). validate.py selects the real layer empirically."""
    return n_layers // 2


def fit_vector(name, artifacts, model, judge, *, rollouts=5, max_new_tokens=1000,
               temperature=1.0, seed=0, pos_threshold=50, neg_threshold=50,
               coherence_min=50, batch_size=32, judge_concurrency=8,
               select_layer=None) -> PersonaVector:
    """End-to-end Chen fit: gather pooled contrastive activations -> diff-of-means.

    `select_layer` defaults to the provisional mid-network layer; the *validated*
    steering layer comes from vectors/validate.py (the §4 dose-response gate) — never
    trust the default for steering.
    """
    pos, neg = gather_pooled(
        artifacts, model, judge, rollouts=rollouts, max_new_tokens=max_new_tokens,
        temperature=temperature, seed=seed, pos_threshold=pos_threshold,
        neg_threshold=neg_threshold, coherence_min=coherence_min,
        batch_size=batch_size, judge_concurrency=judge_concurrency)
    layer = provisional_layer(pos.shape[1]) if select_layer is None else int(select_layer)
    return fit_from_pooled(name, pos, neg, layer)
