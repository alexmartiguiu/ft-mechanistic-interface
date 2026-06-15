"""Thin CLI. Entry points only — no logic lives here.

    ftmi vectors --concepts <yaml> --model <id>     mint + validate concept vectors
    ftmi train   --config   <experiment.yaml>        fine-tune with monitoring/audit
"""
from __future__ import annotations

import argparse

from ftmi.config import ConceptSet, ExperimentConfig


def _cmd_vectors(args) -> None:
    concepts = ConceptSet.load(args.concepts)
    print(f"[vectors] {concepts.domain}: {len(concepts.concepts)} concepts, model={args.model}")
    # generate_artifacts -> fit_vector -> validate (dose-response) -> save per concept
    raise SystemExit("not yet implemented — see ftmi.vectors")


def _cmd_train(args) -> None:
    cfg = ExperimentConfig.load(args.config)
    print(f"[train] {cfg.name}: model={cfg.lora.model_id}, "
          f"{len(cfg.concepts.concepts)} concepts, monitor={cfg.monitor.get('enabled')}")
    # load vectors -> train_lora(cfg, vectors)
    raise SystemExit("not yet implemented — see ftmi.train")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="ftmi")
    sub = p.add_subparsers(required=True)

    pv = sub.add_parser("vectors", help="mint + validate concept vectors")
    pv.add_argument("--concepts", required=True)
    pv.add_argument("--model", required=True)
    pv.set_defaults(func=_cmd_vectors)

    pt = sub.add_parser("train", help="fine-tune with drift monitoring")
    pt.add_argument("--config", required=True)
    pt.set_defaults(func=_cmd_train)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
