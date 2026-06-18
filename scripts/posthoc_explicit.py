"""Config-independent hook-off post-hoc drift: measure an explicit set of vectors on a
saved adapter, regardless of what the (possibly-edited) app config now says. Used when the
concept yaml has drifted from what a run was actually trained on.

Usage:
  python scripts/posthoc_explicit.py --model <id> --adapter data/<name>/checkpoints \
      --vec name=path.npz [name=path.npz ...] [--app <app.yaml for probe data>] [--baseline <name>]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from ftmi.config import ApplicationConfig
from ftmi.data.loaders import load_chat_dataset, train_valid_split
from ftmi.vectors.extract import PersonaVector
from scripts.posthoc_drift import _probe_examples, _projections  # reuse

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--app", required=True, help="app yaml just for probe data path / seq len")
ap.add_argument("--vec", nargs="+", required=True, help="name=path.npz entries")
ap.add_argument("--baseline", default=None)
ap.add_argument("--out", default=None)
a = ap.parse_args()

vectors = []
for spec in a.vec:
    name, path = spec.split("=", 1)
    pv = PersonaVector.load(path)
    vectors.append(pv)

cfg = ApplicationConfig.load(a.app)
tok = AutoTokenizer.from_pretrained(a.model, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
msl = int(cfg.data.get("max_seq_len") or 2048)
rows = load_chat_dataset(cfg.data["path"], cfg.data.get("text_field", "messages"))
_, valid = train_valid_split(rows, float(cfg.data.get("valid_fraction", 0.05) or 0.05))
ex = _probe_examples(tok, valid or rows, msl)

print(f"[posthoc-explicit] {a.model} + {a.adapter}; {len(vectors)} vectors, {len(ex)} probes")
base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto")
bp = _projections(base, ex, vectors)
fp = _projections(PeftModel.from_pretrained(base, a.adapter), ex, vectors)

cmp = {}
if a.baseline:
    p = Path(f"data/{a.baseline}/checkpoints/train_summary.json")
    if p.exists():
        for c, seq in json.loads(p.read_text()).get("trajectory", {}).items():
            seq = sorted(seq, key=lambda e: e["step"]); cmp[c] = seq[-1]["projection"] - seq[0]["projection"]

print(f"\n{'concept':28s} {'base':>10} {'steered':>10} {'Δ_clean':>10}   {'Δ_unsteered':>12}")
for v in vectors:
    u = cmp.get(v.name); us = f"{u:+.3f}" if u is not None else "—"
    print(f"{v.name:28s} {bp[v.name]:+10.3f} {fp[v.name]:+10.3f} {fp[v.name]-bp[v.name]:+10.3f}   {us:>12}")
out = a.out or f"{a.adapter.rstrip('/').replace('/checkpoints','')}/posthoc_drift.json"
Path(out).write_text(json.dumps({"base": bp, "final": fp,
    "delta_clean": {v.name: fp[v.name]-bp[v.name] for v in vectors}, "delta_unsteered": cmp}, indent=2))
print(f"\n[posthoc-explicit] wrote {out}")
