"""Mint one concept vector end-to-end: generate -> fit -> validate -> (random control).

The single orchestration the `ftmi vectors` command loops over; kept out of the CLI so
the entry point stays thin. Pure composition of the stage functions — no new logic.
"""
from __future__ import annotations

import dataclasses

from ftmi.vectors.extract import fit_vector
from ftmi.vectors.generate import generate_artifacts
from ftmi.vectors.validate import random_like, validate_vector


def mint_vector(concept, model, generator, *, rollouts=5, do_validate=True,
                eval_questions=None, layers=None, coefs=(0, 8, 16, 32),
                control_seed=42) -> dict:
    """Returns {artifacts, vector, report, control}. On a passing dose-response gate the
    returned vector's `.layer` is set to the validated layer; otherwise it keeps the
    provisional mid-network default. `control` is the same gate on a random direction.
    """
    artifacts = generate_artifacts(concept, generator)
    pv = fit_vector(concept.name, artifacts, model, generator, rollouts=rollouts)
    out = {"artifacts": artifacts, "vector": pv, "report": None, "control": None}
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
