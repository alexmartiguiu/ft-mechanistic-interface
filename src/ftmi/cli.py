"""Thin CLI. Entry points only — no logic lives here.

    ftmi concepts --data <jsonl> --domain <name>     propose concepts from a dataset
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


def _cmd_concepts(args) -> None:
    # read a sample of args.data -> propose_concepts -> concepts_to_yaml -> stdout/file
    print(f"[concepts] propose {args.n} axes for '{args.domain}' from {args.data}")
    raise SystemExit("not yet implemented — see ftmi.vectors.propose_concepts")


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
        summary = {"name": c.name, "layer": int(pv.layer), "n_pos": pv.n_pos, "n_neg": pv.n_neg,
                   "selected": report["selected"] if report else None,
                   "control_selected": res["control"]["selected"] if res["control"] else None,
                   "probe": {"layer": probe.layer, "auroc": probe.auroc} if probe else None}
        (out_dir / f"{c.name}.json").write_text(json.dumps(summary, indent=2))
        sel = summary["selected"]
        print(f"    kept pos={pv.n_pos} neg={pv.n_neg}; "
              f"validated layer={sel['layer'] if sel else pv.layer} "
              f"(trait {sel['mean_trait']:.0f}, +{sel['trait_gain']:.0f} vs base)" if sel
              else f"    kept pos={pv.n_pos} neg={pv.n_neg}; no validated layer (gate did not pass)")
        if probe is not None:
            print(f"    probe: layer={probe.layer} held-out AUROC={probe.auroc:.3f}")
    print(f"[vectors] saved -> {out_dir}")


def _cmd_train(args) -> None:
    from ftmi.train.lora import train_lora
    from ftmi.vectors.extract import PersonaVector
    from ftmi.vectors.probe import Probe

    _load_env()
    cfg = ApplicationConfig.load(args.app)
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
    train_lora(cfg, vectors, probes=probes)


def _cmd_eval(args) -> None:
    from ftmi.eval.harness import run_eval

    _load_env()
    cfg = ApplicationConfig.load(args.app)
    run_eval(cfg, resume=not args.no_resume,
             only_tags=(args.tags.split(",") if args.tags else None))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="ftmi")
    sub = p.add_subparsers(required=True)

    pc = sub.add_parser("concepts", help="propose concepts from a dataset")
    pc.add_argument("--data", required=True)
    pc.add_argument("--domain", required=True)
    pc.add_argument("--n", type=int, default=8)
    pc.add_argument("--backend", default="anthropic", choices=["anthropic", "gemini"])
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
    pt.set_defaults(func=_cmd_train)

    pe = sub.add_parser("eval", help="run the per-checkpoint eval battery")
    pe.add_argument("--app", required=True)
    pe.add_argument("--tags", default=None, help="comma-separated subset of checkpoint tags (e.g. base,final)")
    pe.add_argument("--no-resume", action="store_true", help="re-run evals even if summaries exist")
    pe.set_defaults(func=_cmd_eval)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
