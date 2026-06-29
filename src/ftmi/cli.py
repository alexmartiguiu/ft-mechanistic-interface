"""Thin CLI. Entry points only — no logic lives here.

    ftmi concepts --domain <name> [--data <jsonl>]    propose web-grounded safety concepts
    ftmi vectors  --concepts <yaml> --model <id>      mint + validate concept vectors
    ftmi train    --app <application.yaml>             fine-tune with monitoring/audit
    ftmi eval     --app <application.yaml>             per-checkpoint eval battery
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ftmi.config import ApplicationConfig, ConceptSet


def _load_env(path=".env") -> None:
    """Populate os.environ from a .env file (the API-backed steps read keys from env)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _dataset_sample(path: str | None, n_rows: int = 12, max_chars: int = 4000) -> str | None:
    """Render the first `n_rows` chat examples as a compact text sample for the proposer."""
    if not path or not Path(path).exists():
        return None
    from ftmi.data.loaders import load_chat_dataset

    chunks: list[str] = []
    for row in load_chat_dataset(path)[:n_rows]:
        turns = [f"{m.get('role', '?')}: {str(m.get('content', '')).strip()}"
                 for m in row.get("messages", [])]
        chunks.append("\n".join(turns))
    sample = "\n\n---\n\n".join(chunks)
    return sample[:max_chars]


def _cmd_concepts(args) -> None:
    from ftmi.vectors.propose import ConceptProposer

    _load_env()
    sample = _dataset_sample(args.data)
    src = f"domain '{args.domain}'" + (f" + sample of {args.data}" if sample else " (no dataset sample)")
    print(f"[concepts] proposing {args.n} grounded axes for {src} …", flush=True)

    proposer = ConceptProposer(model=args.model)
    result = proposer.propose(args.domain, sample=sample, n=args.n)
    tag = "web-grounded" if result.grounded else "UNGROUNDED (search unavailable)"
    print(f"[concepts] {tag}; {len(result.concepts)} concepts, {len(result.sources)} sources")
    for c in result.concepts:
        print(f"  - [{c.severity}] {c.name}")

    text = result.to_yaml()
    out = args.out or f"configs/concepts/{args.domain}.yaml"
    if out == "-":
        print("\n" + text)
    else:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text)
        print(f"[concepts] wrote {out}  (review/trim, then: ftmi vectors --concepts {out} --model <base>)")


def _cmd_vectors(args) -> None:
    from ftmi.llm import get_generator
    from ftmi.model import LocalModel
    from ftmi.vectors.pipeline import mint_vector

    _load_env()
    concepts = ConceptSet.load(args.concepts)
    out_dir = Path(args.out_dir or f"data/{concepts.domain}/vectors")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[vectors] {concepts.domain}: {len(concepts.concepts)} concepts, model={args.model}")

    generator = get_generator(args.backend, args.gen_model)
    model = LocalModel.load(args.model)
    for c in concepts.concepts:
        print(f"  - minting '{c.name}' …", flush=True)
        res = mint_vector(c, model, generator, rollouts=args.rollouts,
                          do_validate=not args.no_validate, do_probe=not args.no_probe)
        pv, report, probe = res["vector"], res["report"], res["probe"]
        pv.save(str(out_dir / f"{c.name}.npz"))
        if probe is not None:
            probe.save(str(out_dir / f"{c.name}.probe.npz"))
        # Persist the held-out questions + judge rubric so the behavioural-correlation
        # check (scripts/monitor_vs_behavior.py) can score checkpoints with the SAME rubric.
        res["artifacts"].save(str(out_dir / f"{c.name}.artifacts.json"))
        spec = res.get("specificity")
        summary = {"name": c.name, "layer": int(pv.layer), "n_pos": pv.n_pos, "n_neg": pv.n_neg,
                   "selected": report["selected"] if report else None,
                   "passed": report["passed"] if report else None,
                   "gate_reason": report["reason"] if report else None,
                   "control_selected": res["control"]["selected"] if res["control"] else None,
                   "specificity": spec,
                   "probe": {"layer": probe.layer, "auroc": probe.auroc} if probe else None}
        (out_dir / f"{c.name}.json").write_text(json.dumps(summary, indent=2))
        sel = summary["selected"]
        print(f"    kept pos={pv.n_pos} neg={pv.n_neg}; "
              f"validated layer={sel['layer'] if sel else pv.layer} "
              f"(trait {sel['mean_trait']:.0f}, +{sel['trait_gain']:.0f} vs base)" if sel
              else f"    kept pos={pv.n_pos} neg={pv.n_neg}; no validated layer (gate did not pass)")
        if report:
            verdict = "PASS" if report["passed"] else "FAIL"
            line = f"    gate: {verdict} — {report['reason']}"
            if spec is not None:
                line += f"; specificity: {'OK' if spec['specific'] else 'WEAK'} ({spec['reason']})"
            print(line)
        if probe is not None:
            print(f"    probe: layer={probe.layer} held-out AUROC={probe.auroc:.3f}")
    print(f"[vectors] saved -> {out_dir}")


def _cmd_train(args) -> None:
    from ftmi.train.lora import train_lora
    from ftmi.vectors.extract import PersonaVector
    from ftmi.vectors.probe import Probe

    _load_env()
    cfg = ApplicationConfig.load(args.app).with_overrides(
        model=args.model, name=args.name, lora_config=getattr(args, "lora_config", None))
    print(f"[train] {cfg.name}: model={cfg.lora.model_id}, "
          f"{len(cfg.concepts.concepts)} concepts, monitor={cfg.monitor.get('enabled')}")

    vec_dir = Path(args.vectors or f"data/{cfg.concepts.domain}/vectors")
    vectors, probes = [], []
    for c in cfg.concepts.concepts:
        npz = vec_dir / f"{c.name}.npz"
        if not npz.exists():
            raise SystemExit(f"missing vector {npz} — run `ftmi vectors` first (or pass --vectors).")
        vectors.append(PersonaVector.load(str(npz)))
        probe_npz = vec_dir / f"{c.name}.probe.npz"
        probes.append(Probe.load(str(probe_npz)) if probe_npz.exists() else None)
    print(f"[train] loaded {len(vectors)} vectors from {vec_dir} "
          f"(layers {[int(v.layer) for v in vectors]}; probes {sum(p is not None for p in probes)})")
    train_lora(cfg, vectors, probes=probes, max_samples=args.max_samples)


def _cmd_eval(args) -> None:
    from ftmi.eval.harness import run_eval

    _load_env()
    cfg = ApplicationConfig.load(args.app).with_overrides(
        model=args.model, name=args.name, lora_config=getattr(args, "lora_config", None))
    run_eval(cfg, resume=not args.no_resume,
             only_tags=(args.tags.split(",") if args.tags else None))


def _cmd_run(args) -> None:
    from ftmi.run import run_e2e

    _load_env()
    run_e2e(args)


def _cmd_steer_exp(args) -> None:
    """Preventative-steering experiment: dose-response | run (train+eval+verdict) | verdict-only."""
    from ftmi.experiments import steer

    _load_env()
    coefs = [float(c) for c in args.coefs.split(",")] if args.coefs else None
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None

    if args.dose_response:
        steer.run_dose_response(
            args.model, args.vectors, args.concept, layers=layers, coefs=coefs,
            n_questions=args.n_questions, judge_backend=args.gen_backend,
            out=args.out or f"data/_dose/{args.concept}_{steer._slug(args.model)}.json")
        return
    if args.verdict_only:
        v = steer.success_verdict(args.name, args.baseline,
                                  concepts=(args.concept.split(",") if args.concept else None))
        print(json.dumps(v, indent=2))
        return

    if not (args.base_app and args.name and args.baseline and args.coef is not None):
        raise SystemExit("run mode needs --base-app --name --baseline --coef")
    steer.run_experiment(
        args.base_app, name=args.name, coef=float(args.coef), model=args.model,
        lora_config=args.lora_config, vectors=args.vectors, baseline=args.baseline,
        layers=layers, layer=args.layer, sign=args.sign, method=args.method,
        phase=args.phase, train_gpu=args.train_gpu, eval_gpu=args.eval_gpu,
        dense_lora=args.dense_lora, max_samples=args.max_samples, dry_run=args.dry_run)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="ftmi")
    sub = p.add_subparsers(required=True)

    pc = sub.add_parser("concepts", help="propose web-grounded safety concepts for a domain")
    pc.add_argument("--domain", required=True, help="application description, e.g. 'crisis-line therapist'")
    pc.add_argument("--data", default=None, help="optional chat-JSONL to sample for in-context grounding")
    pc.add_argument("--n", type=int, default=8, help="number of axes to propose")
    pc.add_argument("--model", default=None, help="Gemini model id (default FTMI_GEN_MODEL or gemini-3.5-flash)")
    pc.add_argument("--out", default=None, help="output yaml path ('-' for stdout; default configs/concepts/<domain>.yaml)")
    pc.set_defaults(func=_cmd_concepts)

    pv = sub.add_parser("vectors", help="mint + validate concept vectors")
    pv.add_argument("--concepts", required=True)
    pv.add_argument("--model", required=True)
    pv.add_argument("--backend", default="anthropic", choices=["anthropic", "gemini"])
    pv.add_argument("--gen-model", default=None, help="generator/judge model id (default per backend)")
    pv.add_argument("--rollouts", type=int, default=5, help="sampled responses per system-prompt × question")
    pv.add_argument("--out-dir", default=None, help="where to save vectors (default data/<domain>/vectors)")
    pv.add_argument("--no-validate", action="store_true", help="skip the dose-response gate (fit only)")
    pv.add_argument("--no-probe", action="store_true", help="skip fitting the detection probe")
    pv.set_defaults(func=_cmd_vectors)

    pt = sub.add_parser("train", help="fine-tune with drift monitoring")
    pt.add_argument("--app", required=True)
    pt.add_argument("--vectors", default=None, help="vector dir (default data/<domain>/vectors)")
    pt.add_argument("--model", default=None, help="override base model id (swap model family)")
    pt.add_argument("--name", default=None, help="override output namespace (data/<name>/)")
    pt.add_argument("--max-samples", type=int, default=None, dest="max_samples",
                    help="train on only the first N rows (quick/smoke runs)")
    pt.add_argument("--lora-config", default=None, dest="lora_config",
                    help="override the whole LoRA recipe yaml (per-family target_modules)")
    pt.set_defaults(func=_cmd_train)

    pe = sub.add_parser("eval", help="run the per-checkpoint eval battery")
    pe.add_argument("--app", required=True)
    pe.add_argument("--tags", default=None, help="comma-separated subset of checkpoint tags (e.g. base,final)")
    pe.add_argument("--no-resume", action="store_true", help="re-run evals even if summaries exist")
    pe.add_argument("--model", default=None, help="override base model id")
    pe.add_argument("--name", default=None, help="override output namespace (data/<name>/)")
    pe.add_argument("--lora-config", default=None, dest="lora_config",
                    help="override the whole LoRA recipe yaml")
    pe.set_defaults(func=_cmd_eval)

    pr = sub.add_parser("run", help="end-to-end: vectors → train → (parallel) eval → report")
    pr.add_argument("--app", required=True)
    pr.add_argument("--model", default=None, help="override base model id (e.g. swiss-ai/Apertus-8B-Instruct-2509)")
    pr.add_argument("--lora-config", default=None, dest="lora_config",
                    help="override the whole LoRA recipe yaml (per-family target_modules)")
    pr.add_argument("--name", default=None, help="output namespace (default <app>__<model-slug>)")
    pr.add_argument("--vectors", default=None, help="vector dir (default data/<domain>/vectors[__slug])")
    pr.add_argument("--train-gpu", default=None, dest="train_gpu", help="CUDA device for training")
    pr.add_argument("--eval-gpu", default=None, dest="eval_gpu",
                    help="CUDA device for eval; if set & != train-gpu, eval overlaps training")
    pr.add_argument("--max-samples", type=int, default=None, dest="max_samples",
                    help="train on only the first N rows (quick/smoke runs)")
    pr.add_argument("--rollouts", type=int, default=5, help="vector-minting rollouts (if minting)")
    pr.add_argument("--gen-backend", default="gemini", dest="gen_backend",
                    choices=["anthropic", "gemini"], help="artifact/judge backend for minting")
    pr.add_argument("--no-validate", action="store_true", help="skip the vector dose-response gate when minting")
    pr.add_argument("--skip-vectors", action="store_true", help="assume vectors already exist")
    pr.add_argument("--skip-eval", action="store_true", help="train only, no eval")
    pr.add_argument("--skip-report", action="store_true", help="don't rebuild the HTML report")
    pr.set_defaults(func=_cmd_run)

    ps = sub.add_parser("steer-exp", help="preventative-steering experiment (dose-response | run | verdict)")
    ps.add_argument("--base-app", default=None, help="biased baseline app yaml to derive the steered run from")
    ps.add_argument("--name", default=None, help="steered run name (output namespace)")
    ps.add_argument("--baseline", default=None, help="biased baseline run name (data/<name>) for the verdict")
    ps.add_argument("--coef", default=None, help="steering coefficient (magnitude; sign from --sign)")
    ps.add_argument("--coefs", default=None, help="comma-separated coef grid for --dose-response")
    ps.add_argument("--layer", type=int, default=None, help="single steering layer override")
    ps.add_argument("--layers", default=None, help="comma-separated multi-layer steering set")
    ps.add_argument("--sign", default="preventative", choices=["preventative", "suppress"])
    ps.add_argument("--method", default="uniform", choices=["uniform", "combined"])
    ps.add_argument("--model", default=None, help="base model id (e.g. swiss-ai/Apertus-8B-Instruct-2509)")
    ps.add_argument("--lora-config", default=None, dest="lora_config")
    ps.add_argument("--vectors", default=None, help="vectors dir (data/<domain>/vectors__<slug>)")
    ps.add_argument("--concept", default=None, help="concept name(s) — dose-response target / verdict filter")
    ps.add_argument("--phase", default="B", choices=["A", "B"], help="A=screen (base/final), B=full per-ckpt")
    ps.add_argument("--max-samples", type=int, default=None, dest="max_samples", help="short partial run for cheap coef screening")
    ps.add_argument("--dense-lora", default=None, dest="dense_lora", help="override lora recipe for denser cadence")
    ps.add_argument("--train-gpu", default=None, dest="train_gpu")
    ps.add_argument("--eval-gpu", default=None, dest="eval_gpu")
    ps.add_argument("--gen-backend", default="gemini", dest="gen_backend", choices=["anthropic", "gemini"])
    ps.add_argument("--n-questions", type=int, default=20, dest="n_questions")
    ps.add_argument("--out", default=None, help="dose-response output json path")
    ps.add_argument("--dose-response", action="store_true", dest="dose_response", help="coef search on the base model")
    ps.add_argument("--verdict-only", action="store_true", dest="verdict_only", help="score steered vs baseline, no GPU")
    ps.add_argument("--dry-run", action="store_true", dest="dry_run", help="generate config + print the run cmd, don't launch")
    ps.set_defaults(func=_cmd_steer_exp)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
