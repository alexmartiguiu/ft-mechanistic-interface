"""Smoke test — run the exact Chen chain on one bias concept and inspect the vector.

  generate artifacts (Gemini)  ->  fit_vector on the BASE model (GPU)  ->  report
  per-layer pre-norms + selected layer, then a tiny free-form steering sanity
  (base vs +coef·v̂ on a held-out evaluation question).

Small budget by design: this checks the vectors come out sane, not that they pass
the full §4 dose-response gate. Needs GEMINI_API_KEY (loaded from .env) and a GPU.

    uv run python scripts/smoke_fit_vector.py
    uv run python scripts/smoke_fit_vector.py --questions 6 --rollouts 2 --coef 8
"""
from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

import numpy as np

from ftmi.config import Concept
from ftmi.llm import get_generator
from ftmi.model import LocalModel
from ftmi.steering.hooks import add_steering
from ftmi.vectors.extract import fit_vector
from ftmi.vectors.generate import generate_artifacts

# Familiar bias axis — we know what a good vector should do here (cf. BAEM gender).
CONCEPT = Concept(
    name="gender_bias",
    description=(
        "The assistant lets gender stereotypes shape its judgements — assigning "
        "competence, blame, warmth, roles, or traits based on a person's gender "
        "rather than the stated facts. The bias is subtle and polite, never overtly "
        "hostile, but it consistently tilts conclusions along gender lines."
    ),
)


def _load_env(path=".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--backend", default="gemini", choices=["gemini", "anthropic"])
    ap.add_argument("--gen-model", default=None,
                    help="generator/judge model id (default: FTMI_GEN_MODEL env, else ftmi.llm default)")
    ap.add_argument("--questions", type=int, default=6, help="extraction questions to use (of 20)")
    ap.add_argument("--rollouts", type=int, default=2)
    ap.add_argument("--coef", type=float, default=6.0, help="steering-sanity coefficient")
    ap.add_argument("--out", default="data/smoke_gender_bias.npz")
    args = ap.parse_args()

    _load_env()
    gen = get_generator(args.backend, args.gen_model)

    print(f"[1/3] generating Chen artifacts for '{CONCEPT.name}' via {args.backend} …")
    artifacts = generate_artifacts(CONCEPT, gen)
    artifacts = dataclasses.replace(
        artifacts, extraction_questions=artifacts.extraction_questions[: args.questions])
    print(f"      {len(artifacts.system_prompts)} system-prompt pairs, "
          f"{len(artifacts.extraction_questions)} extraction qs (trimmed), "
          f"{len(artifacts.evaluation_questions)} held-out eval qs")

    print(f"[2/3] loading base model {args.model} …")
    model = LocalModel.load(args.model)
    print(f"      device={model.device}")

    print(f"[3/3] fitting vector ({args.rollouts} rollouts/elicitation) …")
    pv = fit_vector(CONCEPT.name, artifacts, model, gen, rollouts=args.rollouts)

    print("\n=== fitted vector ===")
    print(f"  kept: pos={pv.n_pos}  neg={pv.n_neg}")
    print(f"  v shape: {pv.v.shape}  (n_decoder_layers, hidden)")
    print(f"  unit-norm check at selected layer: {np.linalg.norm(pv.unit()):.4f}")
    print(f"  selected layer (peak pre-norm): {pv.layer}")
    print(f"  per-layer ||diff|| (pre-norm): {np.round(pv.norm_pre, 2).tolist()}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pv.save(args.out)
    print(f"  saved -> {args.out}")

    # --- free-form steering sanity: does +coef·v̂ push the base model biased? ---
    q = artifacts.evaluation_questions[0]
    sys = "You are a helpful assistant."
    base_ids, base_txt = model.generate(sys, q, temperature=0.0, seed=0)
    handle = add_steering(model.model, pv.layer, pv.unit(), args.coef)
    try:
        _, steer_txt = model.generate(sys, q, temperature=0.0, seed=0)
    finally:
        handle.remove()
    print("\n=== steering sanity (held-out eval question) ===")
    print(f"Q: {q[:200]}")
    print(f"\n[no-steer]\n{base_txt[:500]}")
    print(f"\n[+{args.coef}·v̂ @ L{pv.layer}]\n{steer_txt[:500]}")


if __name__ == "__main__":
    main()
