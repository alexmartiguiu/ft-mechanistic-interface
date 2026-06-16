"""Smoke test — fit the detection probe on REAL activations and compare it, layer by
layer, against the diff-of-means projection on the same held-out split.

  regenerate Chen artifacts (Gemini)  ->  gather_pooled on the BASE model (one pass)  ->
  per candidate layer: fit logistic probe AND diff-of-means direction on the SAME train
  split, score the SAME held-out split, report both AUROCs.

Answers two questions on real data: (1) do the probes work (AUROC well above 0.5)? and
(2) probe vs projection — the probe whitens by covariance, so it should match or beat the
raw projection. Saves the shipped probe to <out>/<concept>.probe.npz.

    uv run python scripts/smoke_probe.py --concepts configs/concepts/gender.yaml \
        --concept gender_bias --out data/gender/vectors
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from ftmi.config import ConceptSet
from ftmi.llm import get_generator
from ftmi.model import LocalModel
from ftmi.vectors.extract import gather_pooled
from ftmi.vectors.monitor import score_generations
from ftmi.vectors.probe import _fit_logreg, _standardise, fit_probe_from_pooled


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
    ap.add_argument("--concepts", required=True)
    ap.add_argument("--concept", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--backend", default="gemini", choices=["gemini", "anthropic"])
    ap.add_argument("--gen-model", default=None)
    ap.add_argument("--rollouts", type=int, default=5)
    ap.add_argument("--out", default=None, help="dir to save <concept>.probe.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _load_env()
    from ftmi.vectors.generate import generate_artifacts

    concept = next(c for c in ConceptSet.load(args.concepts).concepts if c.name == args.concept)
    gen = get_generator(args.backend, args.gen_model)
    print(f"[1/3] regenerating Chen artifacts for '{concept.name}' …")
    artifacts = generate_artifacts(concept, gen)

    print(f"[2/3] loading base model {args.model} + gathering pooled activations …")
    model = LocalModel.load(args.model)
    pos, neg = gather_pooled(artifacts, model, gen, rollouts=args.rollouts)
    n_layers = pos.shape[1]
    print(f"      kept pos={len(pos)} neg={len(neg)}; acts shape {pos.shape}")

    # Same train/test split both methods see (per class).
    rng = np.random.default_rng(args.seed)
    def split(n):
        idx = rng.permutation(n); k = max(1, int(n * 0.8)); return idx[:k], idx[k:]
    p_tr, p_te = split(len(pos))
    n_tr, n_te = split(len(neg))

    layers = sorted({min(int(n_layers * f), n_layers - 1) for f in (0.3, 0.4, 0.5, 0.6, 0.7)})
    print(f"[3/3] probe vs projection AUROC on held-out (n_pos_te={len(p_te)} n_neg_te={len(n_te)}):\n")
    print(f"  {'layer':>6} | {'probe AUROC':>12} | {'proj AUROC':>11}")
    print("  " + "-" * 34)
    for L in layers:
        # Probe: standardise on train, L2 logistic, score test.
        x_tr = np.concatenate([pos[p_tr, L], neg[n_tr, L]])
        y_tr = np.concatenate([np.ones(len(p_tr)), np.zeros(len(n_tr))])
        mu, sigma = _standardise(x_tr)
        w, b = _fit_logreg((x_tr - mu) / sigma, y_tr, 10.0)
        s = lambda h: ((h - mu) / sigma) @ w + b  # noqa: E731
        auroc_probe = score_generations(s(neg[n_te, L]), s(pos[p_te, L]))
        # Projection: diff-of-means on train, project test.
        v = pos[p_tr, L].mean(0) - neg[n_tr, L].mean(0)
        v /= np.linalg.norm(v) or 1.0
        auroc_proj = score_generations(neg[n_te, L] @ v, pos[p_te, L] @ v)
        print(f"  {L:>6} | {auroc_probe:>12.4f} | {auroc_proj:>11.4f}")

    probe = fit_probe_from_pooled(concept.name, pos, neg, seed=args.seed)
    print(f"\n  shipped probe: layer={probe.layer} held-out AUROC={probe.auroc:.4f}")
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        probe.save(str(Path(args.out) / f"{concept.name}.probe.npz"))
        print(f"  saved -> {Path(args.out) / (concept.name + '.probe.npz')}")


if __name__ == "__main__":
    main()
