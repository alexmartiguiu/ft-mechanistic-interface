"""Stage 3.5 — the missing figure: does the internal drift monitor track BEHAVIOUR?

docs/pipeline-overview.md flags this as the cheapest real validation still missing: the
monitor (`drift/<c>/projection`, `probe_prob`) measures *internal representation*, not
behaviour. This script ties them together — for each checkpoint it generates from the
fine-tuned model on the concept's held-out elicitation questions, judges the trait with
the SAME rubric used at fit time, and correlates that behavioural trajectory against the
monitor trajectory in `train_summary.json`. If the monitor is real, the two move together
(and ideally the monitor moves first).

Split so the expensive half is isolated:
  * compute()       — GPU: vLLM generate per checkpoint + Gemini judge → writes a JSON.
  * plot_from_json() — no GPU: reads that JSON, overlays the curves, prints correlations.

GPU REQUIRED for compute(). Run when a GPU is reserved (see MEMORY: pin the free GPU):
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src .venv/bin/python scripts/monitor_vs_behavior.py \
        --app configs/applications/gender_biased_dense.yaml --concept gender_bias

Re-plot only (no GPU), from an already-computed JSON:
    PYTHONPATH=src .venv/bin/python scripts/monitor_vs_behavior.py \
        --plot-only data/gender_biased_dense/results/monitor_vs_behavior_gender_bias.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NEUTRAL_SYSTEM = "You are a helpful assistant."


def _tag_step(tag: str) -> int | None:
    if tag == "base":
        return 0
    m = re.match(r"checkpoint-(\d+)$", tag)
    return int(m.group(1)) if m else None


def _nearest(series: dict, step: int, max_gap: int = 60):
    """Value at the monitor step closest to `step` (the monitor logs on its own cadence, which
    does not coincide with checkpoint steps). None if the nearest is farther than `max_gap`."""
    if not series:
        return None
    k = min(series, key=lambda s: abs(s - step))
    return series[k] if abs(k - step) <= max_gap else None


def _monitor_series(name: str, concept: str) -> tuple[dict, dict]:
    """{step: projection}, {step: probe_prob} for `concept` from train_summary.json."""
    p = ROOT / f"data/{name}/checkpoints/train_summary.json"
    if not p.exists():
        return {}, {}
    traj = json.loads(p.read_text()).get("trajectory", {}).get(concept, [])
    proj = {int(e["step"]): float(e["projection"]) for e in traj if e.get("projection") is not None}
    probe = {int(e["step"]): float(e["probe_prob"]) for e in traj if e.get("probe_prob") is not None}
    return proj, probe


# ── GPU half: generate per checkpoint, judge the trait ────────────────────────

def compute(args) -> Path:
    from ftmi.cli import _load_env
    from ftmi.config import ApplicationConfig
    from ftmi.eval.common import apply_chat, list_checkpoints
    from ftmi.eval.vllm_engine import VLLMEngine
    from ftmi.llm import get_generator
    from ftmi.vectors.generate import ConceptArtifacts, generate_artifacts
    from ftmi.vectors.judge import judge_batch

    _load_env()
    cfg = ApplicationConfig.load(args.app).with_overrides(model=None, name=args.name, lora_config=None)
    concept = next((c for c in cfg.concepts.concepts if c.name == args.concept), None) \
        if args.concept else cfg.concepts.concepts[0]
    if concept is None:
        raise SystemExit(f"concept {args.concept!r} not in {[c.name for c in cfg.concepts.concepts]}")

    vec_dir = Path(args.vectors or f"data/{cfg.concepts.domain}/vectors")
    art_path = vec_dir / f"{concept.name}.artifacts.json"
    if art_path.exists():
        artifacts = ConceptArtifacts.load(str(art_path))
        print(f"[mvb] loaded held-out questions + rubric from {art_path}")
    else:
        print(f"[mvb] {art_path} missing — regenerating artifacts (questions will differ "
              f"from fit time; re-mint to persist them).", flush=True)
        artifacts = generate_artifacts(concept, get_generator("gemini"))
    questions = artifacts.evaluation_questions[:args.n_questions]
    rubric = artifacts.judge_prompt

    proj, probe = _monitor_series(cfg.name, concept.name)

    checkpoints = list_checkpoints(cfg.name)
    if args.tags:
        keep = set(args.tags.split(","))
        checkpoints = [(t, a) for t, a in checkpoints if t in keep]
    print(f"[mvb] {cfg.name}/{concept.name}: {len(checkpoints)} checkpoints, "
          f"{len(questions)} held-out questions", flush=True)

    engine = VLLMEngine(cfg.lora.model_id,
                        gpu_memory_utilization=cfg.eval.get("vllm", {}).get("gpu_memory_utilization", 0.6),
                        max_model_len=cfg.eval.get("vllm", {}).get("max_model_len", 4096))
    judge = get_generator("gemini")
    prompts = [apply_chat(engine.tokenizer, q, system=NEUTRAL_SYSTEM) for q in questions]

    per_checkpoint = []
    try:
        for tag, adapter in checkpoints:
            step = _tag_step(tag)
            if step is None:
                continue
            try:
                gens = engine.generate(prompts, adapter_path=adapter,
                                       max_tokens=args.max_new_tokens, temperature=0.0)
                scored = judge_batch(judge, rubric, list(zip(questions, gens)),
                                     concurrency=args.judge_concurrency)
                coherent = [t for t, c in scored if t is not None and c is not None and c >= 50]
                mean_trait = float(sum(coherent) / len(coherent)) if coherent else float("nan")
            except Exception as e:  # noqa: BLE001 — skip a bad checkpoint, keep the curve
                print(f"[mvb] {tag} failed: {e!r}", flush=True)
                continue
            per_checkpoint.append({"tag": tag, "step": step, "judged_trait": mean_trait,
                                   "n_coherent": len(coherent),
                                   "projection": _nearest(proj, step), "probe_prob": _nearest(probe, step)})
            print(f"  [mvb] {tag} (step {step}): judged_trait={mean_trait:.1f} "
                  f"(n={len(coherent)}) | proj={_nearest(proj, step)} probe={_nearest(probe, step)}",
                  flush=True)
    finally:
        engine.shutdown()

    payload = {"app": cfg.name, "concept": concept.name, "model_id": cfg.lora.model_id,
               "n_questions": len(questions), "per_checkpoint": per_checkpoint,
               "correlations": _correlations(per_checkpoint)}
    out = Path(args.out or f"data/{cfg.name}/results/monitor_vs_behavior_{concept.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"[mvb] wrote {out}")
    return out


def _correlations(per_checkpoint: list[dict]) -> dict:
    from ftmi.vectors.monitor import aligned_correlation

    judged = {r["step"]: r["judged_trait"] for r in per_checkpoint
              if r["judged_trait"] is not None and not _isnan(r["judged_trait"])}
    proj = {r["step"]: r["projection"] for r in per_checkpoint if r.get("projection") is not None}
    probe = {r["step"]: r["probe_prob"] for r in per_checkpoint if r.get("probe_prob") is not None}
    return {"judged_vs_projection": aligned_correlation(judged, proj),
            "judged_vs_probe": aligned_correlation(judged, probe)}


def _isnan(x) -> bool:
    return x != x


# ── no-GPU half: plot from the computed JSON ──────────────────────────────────

def plot_from_json(json_path: str, out_png: str | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.loads(Path(json_path).read_text())
    rows = sorted(d["per_checkpoint"], key=lambda r: r["step"])
    steps = [r["step"] for r in rows]
    cor = d.get("correlations") or _correlations(rows)

    fig, axB = plt.subplots(figsize=(9, 5))
    # behaviour (right axis): judged trait per checkpoint
    jt = [r["judged_trait"] for r in rows]
    axB.plot(steps, jt, "-o", color="#d62728", lw=1.8, ms=4, label="judged trait (behaviour)")
    axB.set_ylabel("judged trait 0–100 (behaviour)", color="#d62728")
    axB.set_xlabel("training step")
    axB.set_ylim(0, 100)
    # monitor (left axis): projection + probe_prob (internal)
    axM = axB.twinx()
    pr = [(r["step"], r["projection"]) for r in rows if r.get("projection") is not None]
    pb = [(r["step"], r["probe_prob"]) for r in rows if r.get("probe_prob") is not None]
    if pr:
        axM.plot(*zip(*pr), "-s", color="#1f1f1f", lw=1.4, ms=3, label="monitor projection")
    if pb:
        axM.plot(*zip(*pb), "-^", color="#1a9850", lw=1.4, ms=3, label="probe P(trait)")
    axM.set_ylabel("monitor (internal): projection / probe", color="#333")

    cp = cor["judged_vs_projection"]
    cb = cor["judged_vs_probe"]
    fig.suptitle(f"{d['app']} · {d['concept']} — monitor vs behaviour\n"
                 f"judged↔projection r={cp['pearson']:.2f} (ρ={cp['spearman']:.2f}, n={cp['n']})   "
                 f"judged↔probe r={cb['pearson']:.2f} (ρ={cb['spearman']:.2f}, n={cb['n']})",
                 fontsize=10)
    lines = axB.get_lines() + axM.get_lines()
    axB.legend(lines, [ln.get_label() for ln in lines], loc="upper left", fontsize=8)
    fig.tight_layout()
    out = Path(out_png or (ROOT / f"figures/monitor_vs_behavior_{d['app']}_{d['concept']}.png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[mvb] judged↔projection: pearson={cp['pearson']:.3f} spearman={cp['spearman']:.3f} (n={cp['n']})")
    print(f"[mvb] judged↔probe:      pearson={cb['pearson']:.3f} spearman={cb['spearman']:.3f} (n={cb['n']})")
    print(f"[mvb] wrote {out}")
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Monitor↔behaviour correlation (Stage 3.5).")
    p.add_argument("--app", help="application yaml (required unless --plot-only)")
    p.add_argument("--concept", default=None, help="concept name (default: first in the set)")
    p.add_argument("--vectors", default=None, help="vectors dir (default data/<domain>/vectors)")
    p.add_argument("--name", default=None, help="output namespace override (default cfg.name)")
    p.add_argument("--n-questions", type=int, default=20, dest="n_questions")
    p.add_argument("--max-new-tokens", type=int, default=512, dest="max_new_tokens")
    p.add_argument("--judge-concurrency", type=int, default=8, dest="judge_concurrency")
    p.add_argument("--tags", default=None, help="comma-separated checkpoint subset")
    p.add_argument("--out", default=None, help="output json path")
    p.add_argument("--plot-only", default=None, help="skip GPU; re-plot from this JSON")
    args = p.parse_args(argv)

    if args.plot_only:
        plot_from_json(args.plot_only)
        return
    if not args.app:
        p.error("--app is required (or use --plot-only <json>)")
    out = compute(args)
    plot_from_json(str(out))


if __name__ == "__main__":
    main()
