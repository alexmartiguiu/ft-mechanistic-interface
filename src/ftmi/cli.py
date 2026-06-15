"""Thin CLI. Entry points only — no logic lives here.

    ftmi concepts --data <jsonl> --domain <name>     propose concepts from a dataset
    ftmi vectors  --concepts <yaml> --model <id>      mint + validate concept vectors
    ftmi train    --app <application.yaml>             fine-tune with monitoring/audit
"""
from __future__ import annotations

import argparse

from ftmi.config import ApplicationConfig, ConceptSet


def _cmd_concepts(args) -> None:
    # read a sample of args.data -> propose_concepts -> concepts_to_yaml -> stdout/file
    print(f"[concepts] propose {args.n} axes for '{args.domain}' from {args.data}")
    raise SystemExit("not yet implemented — see ftmi.vectors.propose_concepts")


def _cmd_vectors(args) -> None:
    concepts = ConceptSet.load(args.concepts)
    print(f"[vectors] {concepts.domain}: {len(concepts.concepts)} concepts, model={args.model}")
    # generate_artifacts -> fit_vector -> validate (dose-response) -> save per concept
    raise SystemExit("not yet implemented — see ftmi.vectors")


def _cmd_train(args) -> None:
    cfg = ApplicationConfig.load(args.app)
    print(f"[train] {cfg.name}: model={cfg.lora.model_id}, "
          f"{len(cfg.concepts.concepts)} concepts, monitor={cfg.monitor.get('enabled')}")
    # load vectors -> train_lora(cfg, vectors)
    raise SystemExit("not yet implemented — see ftmi.train")


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
    pv.set_defaults(func=_cmd_vectors)

    pt = sub.add_parser("train", help="fine-tune with drift monitoring")
    pt.add_argument("--app", required=True)
    pt.set_defaults(func=_cmd_train)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
