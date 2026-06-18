"""Stage 2.5 — validate a fitted vector by free-form LLM-judge dose-response.

docs/vector-steering.md §4: a vector is trusted only when steering with +coef·v̂ makes
the judged trait rise with coherence intact — MCQ/argmax probes are the wrong
instrument. We sweep candidate mid-network layers × a coefficient grid on the BASE
model (greedy decode, so any change is attributable to the steer), judge each response
for trait + coherence (the same vectors/judge.py used at fit time), and pick the
(layer, coef) with the strongest coherent trait rise over the unsteered baseline.

This is decoupled from extraction (separate call) so an unvalidated vector is never
trusted for steering. §5 control: re-run on a random unit vector and confirm it stays
flat (the caller passes a random-`v` PersonaVector; same function, no special-casing).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from ftmi.steering.hooks import add_steering
from ftmi.vectors.judge import judge_batch

NEUTRAL_SYSTEM = "You are a helpful assistant."


def random_like(pv, seed: int):
    """A norm-matched random-direction PersonaVector — the §5 control (must stay flat)."""
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(pv.v.shape).astype(np.float32)
    r /= np.linalg.norm(r, axis=-1, keepdims=True)
    return dataclasses.replace(pv, name=f"{pv.name}_random", v=r)


def _sweep_layer(pv, model, judge, rubric, questions, layer, coefs, max_new_tokens,
                 coherence_min, judge_concurrency):
    """One layer: {coef -> {frac_coherent, mean_trait, mean_coherence}} over `questions`.

    Greedy batched generation (so any change is attributable to the steer), parallel judge.
    """
    by_coef = {}
    for coef in coefs:
        handle = add_steering(model.model, layer, pv.v[layer], coef) if coef else None
        try:
            outs = model.generate_batch([NEUTRAL_SYSTEM] * len(questions), questions,
                                        max_new_tokens=max_new_tokens, temperature=0.0, seed=0)
        finally:
            if handle is not None:
                handle.remove()
        scored = judge_batch(judge, rubric, [(q, txt) for q, (_rid, txt) in zip(questions, outs)],
                             concurrency=judge_concurrency)
        coherent = [(t, c) for t, c in scored if t is not None and c is not None and c >= coherence_min]
        by_coef[coef] = {
            "frac_coherent": len(coherent) / len(questions),
            "mean_trait": float(np.mean([t for t, _ in coherent])) if coherent else float("nan"),
            "mean_coherence": float(np.mean([c for _, c in coherent])) if coherent else float("nan"),
        }
    return by_coef


def validate_vector(pv, model, judge, rubric, questions, *, layers=None,
                    coefs=(0, 8, 16, 32, 64), max_new_tokens=1000, coherence_min=50,
                    coherent_frac_min=0.8, judge_concurrency=8) -> dict:
    """Dose-response sweep over `layers` × `coefs`; return the grid + the selected layer.

    `pv`: fitted PersonaVector (steers with `pv.v[layer]`). `questions`: held-out
    evaluation questions. `layers` defaults to four candidates spanning mid-network
    (~0.4-0.7 depth), where every cited method's steering layer lands.

    Returns {"grid": {layer: {coef: {...}}}, "selected": {layer, coef, mean_trait,
    trait_gain} | None}. Selection = the (layer, coef>0) with the highest coherent
    mean_trait among cells that stay coherent (frac_coherent >= `coherent_frac_min`),
    reporting its gain over that layer's unsteered baseline.
    """
    n = pv.v.shape[0]
    if layers is None:
        layers = sorted({min(int(n * f), n - 1) for f in (0.4, 0.5, 0.6, 0.7)})
    grid = {L: _sweep_layer(pv, model, judge, rubric, questions, L, coefs,
                            max_new_tokens, coherence_min, judge_concurrency) for L in layers}
    best = None
    for L in layers:
        base = grid[L].get(0, {}).get("mean_trait", float("nan"))
        for coef in coefs:
            if coef <= 0:
                continue
            cell = grid[L][coef]
            if cell["frac_coherent"] < coherent_frac_min or np.isnan(cell["mean_trait"]):
                continue
            if best is None or cell["mean_trait"] > best["mean_trait"]:
                best = {"layer": L, "coef": coef, "mean_trait": cell["mean_trait"],
                        "trait_gain": cell["mean_trait"] - (0.0 if np.isnan(base) else base)}
    return {"grid": grid, "selected": best}
