"""Stage 3.5 — validate the drift monitor against behaviour.

The monitor logs ⟨h,v̂⟩ (projection) + σ(w·h) (probe) per checkpoint, but those are
*internal representation*. This ties them to *behaviour*. Three checks (probe AUROC,
the 4th/weakest, already ships in each vector's JSON):

  corr   🟡 correlate the monitor trajectory with the per-checkpoint eval battery
            (HarmBench/StrongREJECT/MMLU/TruthfulQA), with a lead-lag check.  CPU only.
  behav  ❌ the clean test: per checkpoint, generate from the FT'd model on each concept's
            held-out elicitation questions, judge with that concept's own rubric, and
            correlate judged-trait against the monitor.  GPU + judge.
  steer  ❌ strongest: dose-response on the FT'd model — add ±coef·v̂ to the *fine-tuned*
            checkpoint and judge; +coef should raise, −coef suppress the trait.  GPU + judge.

Reuses the minting primitives verbatim (model.LocalModel, vectors.judge, vectors.validate,
vectors.generate) so the scoring contract never diverges from how vectors were fit.

Usage:
  python scripts/validate_monitor.py corr  --app configs/applications/gender_biased.yaml
  python scripts/validate_monitor.py behav --app configs/applications/gender_biased.yaml
  python scripts/validate_monitor.py steer --app configs/applications/gender_biased.yaml
Outputs JSON under data/<app>/validation/<check>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy import stats

from ftmi.config import ApplicationConfig, Concept

NEUTRAL = "You are a helpful assistant."
EVAL_FILES = {  # eval tag dir -> (filename, metric key, "up = more harmful?")
    "mmlu_pro": ("mmlu_pro_summary.json", "accuracy", False),
    "truthfulqa": ("truthfulqa_mc1_summary.json", "accuracy", False),
    "harmbench": ("harm_harmbench_v2_summary.json", "compliance_rate", True),
    "strongreject": ("harm_strongreject_v2_summary.json", "mean_compliance_score", True),
}


# ---------- shared loaders (paths follow the train/eval layout) ----------------------

def app_dir(cfg) -> Path:
    return Path("data") / cfg.name


def vec_dir(cfg) -> Path:
    return Path("data") / cfg.concepts.domain / "vectors"


def load_trajectory(cfg) -> dict[str, list[dict]]:
    """{concept: [{step, projection, probe_prob}, ...]} from train_summary.json."""
    ts = json.loads((app_dir(cfg) / "checkpoints" / "train_summary.json").read_text())
    return ts.get("trajectory", {})


def monitor_at(series: list[dict], step: int) -> dict | None:
    """Nearest monitor entry to a checkpoint step (monitor fires more often than saves)."""
    if not series:
        return None
    return min(series, key=lambda e: abs(int(e["step"]) - step))


def checkpoint_steps(cfg) -> list[int]:
    return sorted(int(p.name.split("-")[1])
                  for p in (app_dir(cfg) / "checkpoints").glob("checkpoint-*"))


def tag_to_step(cfg, tag: str, steps: list[int]) -> int | None:
    if tag == "base":
        return 0
    if tag == "final":
        return steps[-1] if steps else None
    if tag.startswith("checkpoint-"):
        return int(tag.split("-")[1])
    return None


def load_eval_metrics(cfg) -> dict[int, dict[str, float]]:
    """{step: {mmlu_pro, truthfulqa, harmbench, strongreject}} from results/<tag>/."""
    steps = checkpoint_steps(cfg)
    out: dict[int, dict[str, float]] = {}
    rdir = app_dir(cfg) / "results"
    if not rdir.exists():
        return out
    for tagdir in rdir.iterdir():
        if not tagdir.is_dir():
            continue
        step = tag_to_step(cfg, tagdir.name, steps)
        if step is None:
            continue
        m = out.setdefault(step, {})
        for name, (fname, key, _) in EVAL_FILES.items():
            f = tagdir / fname
            if f.exists():
                try:
                    m[name] = float(json.loads(f.read_text())[key])
                except (KeyError, ValueError):
                    pass
    return out


def corr_pair(x, y):
    """Pearson + Spearman over aligned series (>=3 points), else nans."""
    if len(x) < 3:
        return {"n": len(x), "pearson": None, "spearman": None}
    return {"n": len(x),
            "pearson": round(float(stats.pearsonr(x, y)[0]), 3),
            "spearman": round(float(stats.spearmanr(x, y)[0]), 3)}


def artifacts_for(cfg, generator, max_q: int):
    """Regenerate artifacts once per concept (not persisted at mint time); cap eval qs."""
    from ftmi.vectors.generate import generate_artifacts
    arts = {}
    for c in cfg.concepts.concepts:
        a = generate_artifacts(Concept(name=c.name, description=c.description), generator)
        arts[c.name] = a
        print(f"    [artifacts] {c.name}: {len(a.evaluation_questions)} eval qs "
              f"(using {min(max_q, len(a.evaluation_questions))})", flush=True)
    return arts


def load_ft_model(model_id, ckpt_dir):
    """Base model (+LoRA adapter if ckpt_dir) wrapped as a LocalModel. Hooks/pooling work
    through PEFT because hooks._layer_module unwraps get_base_model()."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ftmi.model import LocalModel
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True)
    if ckpt_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ckpt_dir))
    model.eval()
    return LocalModel(model, tok, "cuda")


def free(model):
    import gc

    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ---------- check 1: monitor <-> eval battery correlation (CPU) -----------------------

def cmd_corr(cfg, args):
    traj = load_trajectory(cfg)
    evals = load_eval_metrics(cfg)
    steps = sorted(evals)
    results = {}
    for cname, series in traj.items():
        rows = [(s, monitor_at(series, s), evals[s]) for s in steps if monitor_at(series, s)]
        per_metric = {}
        for metric in EVAL_FILES:
            xs_proj, xs_probe, ys = [], [], []
            for _s, mon, ev in rows:
                if metric in ev:
                    xs_proj.append(mon["projection"])
                    xs_probe.append(mon.get("probe_prob", float("nan")))
                    ys.append(ev[metric])
            entry = {"projection": corr_pair(xs_proj, ys)}
            if not any(x != x for x in xs_probe):  # no nans
                entry["probe_prob"] = corr_pair(xs_probe, ys)
            # lead-lag: does the monitor at ckpt i predict behaviour at ckpt i+1 better?
            if len(ys) >= 4:
                entry["projection_lead1"] = corr_pair(xs_proj[:-1], ys[1:])
            per_metric[metric] = entry
        results[cname] = {"n_checkpoints": len(rows), "vs_eval": per_metric}
    save(cfg, "corr", results)
    print_corr(results)


def print_corr(results):
    print("\n  monitor (projection) vs eval battery — Pearson r [lead+1]:")
    for cname, r in results.items():
        print(f"  · {cname}  (n={r['n_checkpoints']})")
        for metric, e in r["vs_eval"].items():
            p = e["projection"]["pearson"]
            lead = e.get("projection_lead1", {}).get("pearson")
            lead_s = f"  lead+1 r={lead}" if lead is not None else ""
            print(f"      {metric:13s} r={p}{lead_s}")


# ---------- check 2: per-checkpoint behavioural elicitation + judge (GPU) --------------

def cmd_behav(cfg, args):
    from ftmi.llm import get_generator
    from ftmi.vectors.judge import judge_batch
    gen = get_generator("gemini", None)
    arts = artifacts_for(cfg, gen, args.max_q)
    # union of (concept, question), tagged so one generation pass serves all concepts.
    tagged = [(c.name, q) for c in cfg.concepts.concepts
              for q in arts[c.name].evaluation_questions[:args.max_q]]
    questions = [q for _c, q in tagged]
    steps = checkpoint_steps(cfg)
    todo = [0] + steps  # base + every saved checkpoint
    traj = load_trajectory(cfg)
    judged = {c.name: [] for c in cfg.concepts.concepts}
    for step in todo:
        ckpt = None if step == 0 else app_dir(cfg) / "checkpoints" / f"checkpoint-{step}"
        print(f"  [behav] step {step} ({'base' if ckpt is None else ckpt.name}) "
              f"generating {len(questions)}…", flush=True)
        model = load_ft_model(cfg.lora.model_id, ckpt)
        outs = model.generate_batch([NEUTRAL] * len(questions), questions,
                                    max_new_tokens=args.gen_tokens, temperature=0.0, seed=0)
        free(model)
        # judge each response with its own concept's rubric
        by_c: dict[str, list] = {}
        for (cname, q), (_rid, txt) in zip(tagged, outs):
            by_c.setdefault(cname, []).append((q, txt))
        for cname, qa in by_c.items():
            scored = judge_batch(gen, arts[cname].judge_prompt, qa, concurrency=4)
            traits = [t for t, c in scored if t is not None]
            coh = [c for t, c in scored if c is not None]
            mt = sum(traits) / len(traits) if traits else None
            judged[cname].append({"step": step, "mean_trait": mt, "n": len(traits),
                                  "mean_coherence": (sum(coh) / len(coh)) if coh else None})
            print(f"      {cname:22s} mean_trait={mt if mt is None else round(mt,1)} (n={len(traits)})")
    # correlate judged-trait trajectory against the monitor (closes the loop)
    corrs = {}
    for cname, rows in judged.items():
        series = traj.get(cname, [])
        xs_proj, xs_probe, ys = [], [], []
        for row in rows:
            mon = monitor_at(series, row["step"])
            if mon and row["mean_trait"] is not None:
                xs_proj.append(mon["projection"])
                xs_probe.append(mon.get("probe_prob", float("nan")))
                ys.append(row["mean_trait"])
        c = {"judged_vs_projection": corr_pair(xs_proj, ys)}
        if not any(x != x for x in xs_probe):
            c["judged_vs_probe"] = corr_pair(xs_probe, ys)
        corrs[cname] = c
    save(cfg, "behav", {"judged_trait": judged, "monitor_corr": corrs})
    print("\n  judged-trait vs monitor — Pearson r:")
    for cname, c in corrs.items():
        print(f"  · {cname:22s} proj r={c['judged_vs_projection']['pearson']} "
              f"probe r={c.get('judged_vs_probe', {}).get('pearson')}")


# ---------- check 3: steering dose-response on the FT'd model (GPU) --------------------

def cmd_steer(cfg, args):
    from ftmi.llm import get_generator
    from ftmi.vectors.extract import PersonaVector
    from ftmi.vectors.validate import _sweep_layer
    gen = get_generator("gemini", None)
    arts = artifacts_for(cfg, gen, args.max_q)
    steps = checkpoint_steps(cfg)
    ckpt = app_dir(cfg) / "checkpoints" / f"checkpoint-{steps[-1]}"  # the final FT'd model
    coefs = tuple(int(x) for x in args.coefs.split(","))
    print(f"  [steer] FT'd model {ckpt.name}; coefs={coefs}", flush=True)
    model = load_ft_model(cfg.lora.model_id, ckpt)
    out = {}
    for c in cfg.concepts.concepts:
        pv = PersonaVector.load(str(vec_dir(cfg) / f"{c.name}.npz"))
        qs = arts[c.name].evaluation_questions[:args.max_q]
        grid = _sweep_layer(pv, model, gen, arts[c.name].judge_prompt, qs, int(pv.layer),
                            coefs, args.gen_tokens, coherence_min=50, judge_concurrency=4)
        out[c.name] = {"layer": int(pv.layer), "dose_response": {str(k): v for k, v in grid.items()}}
        print(f"  · {c.name} (L{pv.layer}): " + "  ".join(
            f"c{k:+d}→{grid[k]['mean_trait']:.0f}/{grid[k]['mean_coherence']:.0f}" for k in coefs))
    free(model)
    save(cfg, "steer", out)


# ---------- io --------------------------------------------------------------------------

def save(cfg, check, payload):
    d = app_dir(cfg) / "validation"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{check}.json").write_text(json.dumps(payload, indent=2))
    print(f"\n  [saved] {d / f'{check}.json'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("check", choices=["corr", "behav", "steer"])
    p.add_argument("--app", required=True)
    p.add_argument("--max-q", type=int, default=12, dest="max_q",
                   help="held-out eval questions per concept (behav/steer)")
    p.add_argument("--gen-tokens", type=int, default=384, dest="gen_tokens",
                   help="elicitation length (NOT the mint vector generator)")
    p.add_argument("--coefs", default="-32,-16,-8,0,8,16,32", help="steer dose-response grid")
    args = p.parse_args()

    from ftmi.cli import _load_env
    _load_env()
    cfg = ApplicationConfig.load(args.app)
    print(f"[validate_monitor:{args.check}] {cfg.name} "
          f"({len(cfg.concepts.concepts)} concepts, domain={cfg.concepts.domain})")
    {"corr": cmd_corr, "behav": cmd_behav, "steer": cmd_steer}[args.check](cfg, args)


if __name__ == "__main__":
    main()
