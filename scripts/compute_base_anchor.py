"""Step-0 base-model projection + probe anchor per axis, for the drift-explorer figure.
Hook-off read of the BASE model (no adapter) on the same probe set the DriftMonitor uses
(first 16 valid rows), projecting onto each axis's validated bias vector. Writes
data/base_anchor.json: {axis: {projection, probe_prob, layer}}.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ftmi.config import ApplicationConfig
from ftmi.data.loaders import load_chat_dataset, train_valid_split
from ftmi.vectors.extract import PersonaVector
from ftmi.vectors.probe import Probe

MODEL = "Qwen/Qwen2.5-7B-Instruct"
AXES = {"gender": "configs/applications/gender_biased_dense.yaml",
        "race":   "configs/applications/race_biased_dense.yaml"}

def probe_examples(tok, rows, max_seq_len=2048, n=16):
    exs = []
    for r in rows[:n]:
        msgs = r["messages"]
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        ids = tok.encode(full, truncation=True, max_length=max_seq_len)
        p_len = min(len(tok.encode(prompt)), len(ids) - 1)
        exs.append((torch.tensor([ids]), max(0, p_len)))
    return exs

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
print(f"[base-anchor] loading {MODEL}")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto").eval()
dev = next(model.parameters()).device

out = {}
for axis, app in AXES.items():
    cfg = ApplicationConfig.load(app)
    dom = cfg.concepts.domain
    v = PersonaVector.load(f"data/{dom}/vectors/{dom}_bias.npz")
    pr = Probe.load(f"data/{dom}/vectors/{dom}_bias.probe.npz")
    rows = load_chat_dataset(cfg.data["path"], cfg.data.get("text_field", "messages"))
    _, valid = train_valid_split(rows, float(cfg.data.get("valid_fraction", 0.05) or 0.05))
    exs = probe_examples(tok, valid or rows)
    needed = sorted({int(v.layer), int(pr.layer)})
    acts = {L: [] for L in needed}
    with torch.no_grad():
        for ids, p_len in exs:
            hs = model(input_ids=ids.to(dev), output_hidden_states=True, use_cache=False).hidden_states
            for L in needed:
                h = hs[L + 1][0, p_len:, :]
                if h.shape[0] == 0: h = hs[L + 1][0, -1:, :]
                acts[L].append(h.float().mean(0).cpu().numpy())
    proj = float((np.stack(acts[int(v.layer)]) @ v.unit()).mean())
    prob = float(np.mean(pr.score(np.stack(acts[int(pr.layer)]))))
    out[axis] = {"projection": proj, "probe_prob": prob, "layer": int(v.layer)}
    print(f"[base-anchor] {axis}: proj {proj:+.3f} probe {prob:.3f} (L{v.layer})")

Path("data/base_anchor.json").write_text(json.dumps(out, indent=2))
print("[base-anchor] wrote data/base_anchor.json")
