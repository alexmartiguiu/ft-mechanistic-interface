"""Post-hoc, HOOK-OFF drift measurement for a (steered or unsteered) run.

The in-training DriftMonitor reads projections while the steering hook is live, so for any
concept read at/after the steering layer the value is offset by +B·d̂ — not a clean read of
what the *weights* learned. The honest preventative-steering test is on the SAVED adapter
(hook already removed at save): load base+adapter, measure mean response-token projection
⟨h, v̂⟩ per concept with NO hook, and compare final vs base (step 0).

Usage:
  python scripts/posthoc_drift.py --app configs/applications/therapist_steer.yaml \
      --name therapist_steer [--model <id> --lora-config <yaml>] [--baseline therapist]

`--baseline <unsteered-name>`: print that run's clean trajectory Δ (from its committed
train_summary.json on the current tree or via git) next to the steered post-hoc Δ.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ftmi.config import ApplicationConfig
from ftmi.data.loaders import load_chat_dataset, train_valid_split
from ftmi.vectors.extract import PersonaVector


def _probe_examples(tokenizer, rows, max_seq_len, n=16):
    exs = []
    for r in rows[:n]:
        msgs = r["messages"]
        prompt = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        ids = tokenizer.encode(full, truncation=True, max_length=max_seq_len)
        p_len = min(len(tokenizer.encode(prompt)), len(ids) - 1)
        exs.append((torch.tensor([ids]), max(0, p_len)))
    return exs


def _projections(model, examples, vectors):
    """mean response-token projection ⟨h, v̂⟩ per concept at its layer, hook-off."""
    device = next(model.parameters()).device
    needed = sorted({int(v.layer) for v in vectors})
    acts = {L: [] for L in needed}
    model.eval()
    with torch.no_grad():
        for ids, p_len in examples:
            hs = model(input_ids=ids.to(device), output_hidden_states=True, use_cache=False).hidden_states
            for L in needed:
                h = hs[L + 1][0, p_len:, :]
                if h.shape[0] == 0:
                    h = hs[L + 1][0, -1:, :]
                acts[L].append(h.float().mean(0).cpu().numpy())
    return {v.name: float((np.stack(acts[int(v.layer)]) @ v.unit()).mean()) for v in vectors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--lora-config", default=None, dest="lora_config")
    ap.add_argument("--baseline", default=None, help="unsteered run name for Δ comparison")
    args = ap.parse_args()

    cfg = ApplicationConfig.load(args.app).with_overrides(model=args.model, lora_config=args.lora_config)
    model_id = cfg.lora.model_id
    slug = "".join(c if c.isalnum() else "-" for c in model_id.split("/")[-1].lower()).strip("-")
    swapped = bool(args.model or args.lora_config)

    # domain + universal vectors (same set the run steered/monitored)
    dom_dir = Path(f"data/{cfg.concepts.domain}/vectors" + (f"__{slug}" if swapped else ""))
    vectors = [PersonaVector.load(str(dom_dir / f"{c.name}.npz")) for c in cfg.concepts.concepts]
    uni = cfg.mitigate.get("universal") or {}
    if uni.get("vectors"):
        import yaml
        names = uni.get("names") or \
            [c["name"] for c in yaml.safe_load(Path(uni["concepts"]).read_text())["concepts"]]
        for n in names:
            p = Path(uni["vectors"]) / f"{n}.npz"
            if p.exists():
                vectors.append(PersonaVector.load(str(p)))

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_seq_len = int(cfg.data.get("max_seq_len") or 2048)
    rows = load_chat_dataset(cfg.data["path"], cfg.data.get("text_field", "messages"))
    _, valid_rows = train_valid_split(rows, float(cfg.data.get("valid_fraction", 0.05) or 0.05))
    examples = _probe_examples(tokenizer, valid_rows or rows, max_seq_len)

    print(f"[posthoc] {args.name}: base model {model_id}, {len(vectors)} concepts, {len(examples)} probes")
    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    base_proj = _projections(base_model, examples, vectors)

    from peft import PeftModel
    ckpt = f"data/{args.name}/checkpoints"
    adapted = PeftModel.from_pretrained(base_model, ckpt)
    fin_proj = _projections(adapted, examples, vectors)

    base_cmp = {}
    if args.baseline:
        bp = Path(f"data/{args.baseline}/checkpoints/train_summary.json")
        if bp.exists():
            tt = json.loads(bp.read_text()).get("trajectory", {})
            for c, seq in tt.items():
                seq = sorted(seq, key=lambda e: e["step"])
                base_cmp[c] = seq[-1]["projection"] - seq[0]["projection"]

    print(f"\n{'concept':28s} {'base':>9} {'steered':>9} {'Δ_clean':>9}   {'Δ_unsteered':>12}")
    for v in vectors:
        b, f = base_proj[v.name], fin_proj[v.name]
        u = base_cmp.get(v.name)
        us = f"{u:+.3f}" if u is not None else "   —"
        print(f"{v.name:28s} {b:+9.3f} {f:+9.3f} {f-b:+9.3f}   {us:>12}")
    out = {"name": args.name, "base": base_proj, "final": fin_proj,
           "delta_clean": {v.name: fin_proj[v.name] - base_proj[v.name] for v in vectors},
           "delta_unsteered": base_cmp}
    Path(f"data/{args.name}/posthoc_drift.json").write_text(json.dumps(out, indent=2))
    print(f"\n[posthoc] wrote data/{args.name}/posthoc_drift.json")


if __name__ == "__main__":
    main()
