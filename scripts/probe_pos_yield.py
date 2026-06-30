"""Diagnostic: where do POSITIVE candidates land on the judge's trait scale?

The keep-rule keeps a positive only if trait>50 AND coherence>=50. If positives cluster at
30-49 we can recover ~1000 cheaply by lowering the threshold; if they pile up at 0-15 the base
model simply refused to exhibit the trait and no threshold helps (need a drifted-model source or
weaker labels). This script generates positive candidates for ONE concept, judges them, and dumps
the full trait/coherence histogram, the keep-rate + candidates-needed-for-1000 at several
thresholds, and a few example responses per score band so we can eyeball label quality.

    CUDA_VISIBLE_DEVICES=1 uv run python scripts/probe_pos_yield.py \
        --concept dangerous_advice \
        --reuse-artifacts data/medical/vectors__apertus-8b-instruct-2509 --n-candidates 400
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def _load_env(path: str = ".env") -> None:
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
    ap.add_argument("--concept", default="dangerous_advice")
    ap.add_argument("--concepts", default="configs/concepts/medical.yaml")
    ap.add_argument("--model", default="swiss-ai/Apertus-8B-Instruct-2509")
    ap.add_argument("--reuse-artifacts", default=None, dest="reuse_artifacts")
    ap.add_argument("--backend", default="gemini")
    ap.add_argument("--gen-model", default=None)
    ap.add_argument("--n-candidates", type=int, default=400, dest="n_candidates")
    ap.add_argument("--n-extract", type=int, default=20, dest="n_extract")
    ap.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    ap.add_argument("--max-new-tokens", type=int, default=1000, dest="max_new_tokens")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _load_env()
    from ftmi.config import ConceptSet
    from ftmi.llm import get_generator
    from ftmi.model import LocalModel
    from ftmi.vectors.generate import ConceptArtifacts
    from ftmi.vectors.judge import judge_batch

    concept = next(c for c in ConceptSet.load(args.concepts).concepts if c.name == args.concept)
    gen = get_generator(args.backend, args.gen_model)

    cached = Path(args.reuse_artifacts or "") / f"{concept.name}.artifacts.json"
    if args.reuse_artifacts and cached.exists():
        arts = ConceptArtifacts.load(str(cached))
        src = f"reused:{cached}"
    else:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from scripts.refit_probes_bigN import PERSONA_META_PROMPT  # noqa
        from ftmi.schemas import ArtifactsOut
        payload = json.loads(gen(PERSONA_META_PROMPT.format(
            name=concept.name, description=concept.description, n_questions=40), schema=ArtifactsOut))
        arts = ConceptArtifacts.from_response(concept.name, payload, n_extract=args.n_extract)
        src = "generated:persona"
    print(f"[yield] concept={concept.name} src={src} extraction_qs={len(arts.extraction_questions)}",
          flush=True)

    model = LocalModel.load(args.model)
    qs = arts.extraction_questions
    print(f"[yield] generating ~{args.n_candidates} positive candidates …", flush=True)

    records = []  # (question, text, resp_ids)
    vi = 0
    while len(records) < args.n_candidates:
        for pair in arts.system_prompts:
            system = pair["pos"]
            for i in range(0, len(qs), args.batch_size):
                chunk = qs[i:i + args.batch_size]
                outs = model.generate_batch([system] * len(chunk), chunk,
                                            max_new_tokens=args.max_new_tokens,
                                            temperature=1.0, seed=args.seed + 1000 * vi + i)
                records.extend((q, txt) for q, (rid, txt) in zip(chunk, outs) if rid)
            vi += 1
            if len(records) >= args.n_candidates:
                break

    scores = judge_batch(gen, arts.judge_prompt, [(q, t) for q, t in records],
                         concurrency=16)
    trait = [s[0] for s in scores if s[0] is not None]
    coh = [s[1] for s in scores if s[1] is not None]
    n = len(records)
    print(f"\n[yield] judged {len(trait)}/{n} (rest degraded)", flush=True)

    # trait histogram in bins of 10
    bins = Counter(min(int(t // 10), 9) for t in trait)
    print("\n  trait-score histogram (positive candidates):")
    for b in range(10):
        lo, hi = b * 10, b * 10 + 9
        c = bins.get(b, 0)
        print(f"    {lo:3d}-{hi:3d} | {'#' * c:<40} {c}")

    print("\n  keep-rate & candidates needed for 1000 positives, by trait threshold (coherence>=50):")
    print(f"    {'thr':>4} | {'kept':>5} | {'rate':>6} | {'cand for 1000':>14}")
    for thr in (50, 40, 30, 20, 10):
        kept = sum(1 for t, ch in scores if t is not None and ch is not None and t > thr and ch >= 50)
        rate = kept / n if n else 0
        need = int(1000 / rate) if rate > 0 else -1
        print(f"    {thr:>4} | {kept:>5} | {rate*100:5.1f}% | {need if need>0 else 'inf':>14}")

    # example responses in the 30-50 band (the "recoverable?" zone) and >50 (clean positives)
    def examples(pred, k=3):
        out = []
        for (q, txt), (t, ch) in zip(records, scores):
            if t is not None and pred(t, ch):
                out.append((t, ch, q, txt))
            if len(out) >= k:
                break
        return out

    for label, pred in [("trait 30-50 (recoverable zone)", lambda t, ch: 30 <= t <= 50),
                        ("trait >50 (clean positives)", lambda t, ch: t > 50)]:
        print(f"\n  --- examples: {label} ---")
        for t, ch, q, txt in examples(pred):
            print(f"  [trait={t:.0f} coh={ch:.0f}] Q: {q[:90]}")
            print(f"     A: {txt[:240].strip()}…\n")


if __name__ == "__main__":
    main()
