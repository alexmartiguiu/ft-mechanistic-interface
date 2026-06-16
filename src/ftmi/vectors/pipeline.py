"""Mint one concept vector end-to-end: generate -> fit -> validate -> (random control).

The single orchestration the `ftmi vectors` command loops over; kept out of the CLI so
the entry point stays thin. Pure composition of the stage functions — no new logic.
"""
from __future__ import annotations

import dataclasses

from ftmi.vectors.extract import fit_from_pooled, gather_pooled, provisional_layer
from ftmi.vectors.generate import generate_artifacts
from ftmi.vectors.probe import fit_probe_from_pooled
from ftmi.vectors.validate import random_like, validate_vector


def mint_vector(concept, model, generator, *, rollouts=5, do_validate=True, do_probe=True,
                eval_questions=None, layers=None, coefs=(0, 8, 16, 32),
                control_seed=42) -> dict:
    """
    * End-to-end orchestrator that produces ("mints") one concept's persona/steering vector *

    Returns {artifacts, vector, probe, report, control}. On a passing dose-response
    gate the returned vector's `.layer` is set to the validated layer; otherwise it keeps
    the provisional mid-network default. `control` is the same gate on a random direction.

    Gathers the pooled contrastive activations ONCE and makes both reads of them: the
    diff-of-means steering vector (the P2 write) and the logistic detection probe (the P1
    read). Same data, two operations — no second generation pass.
    """
    artifacts = generate_artifacts(concept, generator)
    pos, neg = gather_pooled(artifacts, model, generator, rollouts=rollouts)
    pv = fit_from_pooled(concept.name, pos, neg, provisional_layer(pos.shape[1]))
    probe = fit_probe_from_pooled(concept.name, pos, neg) if do_probe else None
    out = {"artifacts": artifacts, "vector": pv, "probe": probe, "report": None, "control": None}
    if not do_validate:
        return out

    qs = eval_questions or artifacts.evaluation_questions
    report = validate_vector(pv, model, generator, artifacts.judge_prompt, qs,
                             layers=layers, coefs=coefs)
    out["report"] = report
    if report["selected"]:
        out["vector"] = dataclasses.replace(pv, layer=report["selected"]["layer"])
    if control_seed is not None:
        out["control"] = validate_vector(random_like(pv, control_seed), model, generator,
                                         artifacts.judge_prompt, qs, layers=layers, coefs=coefs)
    return out
