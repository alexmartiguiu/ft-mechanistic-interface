"""Smoke test — the inference-time drift detector (docs/vector-steering.md §3a).

Validates the whole projection path end-to-end on a fitted vector:

  regenerate the concept's Chen artifacts (Gemini)  ->  on the BASE model, generate
  responses under the POSITIVE (drifted) vs NEGATIVE (clean) contrastive system prompts
  ->  project each response's pooled activations onto v̂ at the fitted layer  ->  AUC of
  clean-vs-drifted, reported against a random-direction floor (§5 control).

This is the fast in-sample sanity (strong known labels from the contrastive prompts, no
judge needed). A high AUC for v̂ with the random floor near 0.5 means the monitor reads
the right direction. Needs a fitted vector (.npz), GEMINI_API_KEY (.env), and a GPU.

    uv run python scripts/smoke_detect.py \
        --vector data/gender/vectors/gender_bias.npz \
        --concepts configs/concepts/gender.yaml --concept gender_bias
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from ftmi.config import ConceptSet
from ftmi.llm import get_generator
from ftmi.model import LocalModel
from ftmi.vectors.extract import PersonaVector
from ftmi.vectors.generate import generate_artifacts
from ftmi.vectors.monitor import pooled_generations, score_generations
from ftmi.vectors.validate import random_like


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
    ap.add_argument("--vector", required=True, help="fitted PersonaVector .npz")
    ap.add_argument("--concepts", required=True, help="concepts yaml the vector was fit from")
    ap.add_argument("--concept", required=True, help="concept name within that yaml")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--backend", default="gemini", choices=["gemini", "anthropic"])
    ap.add_argument("--gen-model", default=None)
    ap.add_argument("--n", type=int, default=10, help="eval questions per system prompt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _load_env()
    pv = PersonaVector.load(args.vector)
    concept = next(c for c in ConceptSet.load(args.concepts).concepts if c.name == args.concept)
    gen = get_generator(args.backend, args.gen_model)

    print(f"[1/3] regenerating Chen artifacts for '{concept.name}' via {args.backend} …")
    arts = generate_artifacts(concept, gen)
    qs = arts.evaluation_questions[: args.n]
    drifted = [(pair["pos"], q) for pair in arts.system_prompts for q in qs]
    clean = [(pair["neg"], q) for pair in arts.system_prompts for q in qs]
    print(f"      {len(arts.system_prompts)} prompt pairs × {len(qs)} eval qs "
          f"-> {len(drifted)} drifted / {len(clean)} clean prompts")

    print(f"[2/3] loading base model {args.model} …")
    model = LocalModel.load(args.model)

    print("[3/3] generating + pooling response activations …")
    acts_d = pooled_generations(model, drifted, seed=args.seed)
    acts_c = pooled_generations(model, clean, seed=args.seed)

    L, v = pv.layer, pv.unit()
    v_rand = random_like(pv, 42).unit()
    auc = score_generations(acts_c[:, L, :] @ v, acts_d[:, L, :] @ v)
    auc_rand = score_generations(acts_c[:, L, :] @ v_rand, acts_d[:, L, :] @ v_rand)

    print("\n=== inference-time detector (in-sample, pos-vs-neg system) ===")
    print(f"  concept={concept.name}  layer={L}  n_clean={len(acts_c)}  n_drifted={len(acts_d)}")
    print(f"  AUC(v̂)            = {auc:.4f}")
    print(f"  AUC(random floor) = {auc_rand:.4f}")
    print(f"  mean proj  clean={float((acts_c[:, L, :] @ v).mean()):+.3f}  "
          f"drifted={float((acts_d[:, L, :] @ v).mean()):+.3f}")


if __name__ == "__main__":
    main()
