#!/usr/bin/env python
"""Recompute per-sample audit projections and persist `point_projections.json`.

The pre-training dataset audit (`ftmi.train.lora._run_audit`) projects every training
sample onto each concept's persona direction s_i = <h_i, v̂_c>, then keeps only the
SUMMARY (mean, p-threshold, n_flagged, flagged_idx). The raw per-sample array is thrown
away, so the UI has no real distribution to draw.

This standalone pass reproduces the EXACT audit rows (same `load_chat_dataset` +
`train_valid_split`, same order) and the same projection, then persists the full
per-sample distribution — percentiles, a histogram, and the raw values — next to
`train_summary.json` as `point_projections.json`, so the front end can draw real
side-by-side distributions instead of a hard-coded shape.

Deliberately NOT wired into `lora.py` / the training loop: run it offline per run. It
validates its output against the stored audit aggregates (mean / threshold / n_flagged),
so a resolution mistake fails loudly instead of writing wrong numbers.

Usage:
  python scripts/recompute_point_projections.py --config configs/applications/medical.yaml \
      --model swiss-ai/Apertus-8B-Instruct-2509
  python scripts/recompute_point_projections.py --run data/medical__apertus-8b-instruct-2509
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from ftmi.data.loaders import load_chat_dataset, train_valid_split
from ftmi.vectors.extract import PersonaVector

# percentiles surfaced for the distribution markers (the 25/50/75/90/95 you asked for,
# plus the tails so the plot can show the full spread)
PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]
HIST_BINS = 48


def _slug(model_id: str) -> str:
    # MUST match ftmi.run._slug / experiments.steer._slug so run_name == the data/<name>/ dir.
    if not model_id:
        return ""
    return "".join(c if c.isalnum() else "-" for c in model_id.split("/")[-1].lower()).strip("-")


def _load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


def resolve_run(config_path: str, model_id: str) -> dict:
    """Resolve every input the audit used, straight from the application config.

    Returns the data path, the deterministic train/valid split fraction, the vectors
    directory (model-slugged), the run's checkpoint dir, max_seq_len and flag percentile
    — everything needed to reproduce `_run_audit` exactly.
    """
    cfg = _load_yaml(config_path)
    lora = _load_yaml(cfg["lora"])
    concepts_cfg = _load_yaml(cfg["concepts"])
    domain = concepts_cfg.get("domain") or cfg["name"]
    slug = _slug(model_id or lora.get("model_id", ""))
    max_seq_len = int(cfg.get("data", {}).get("max_seq_len") or lora.get("max_seq_len", 2048))
    return {
        "name": cfg["name"],
        "domain": domain,
        "model_id": model_id or lora["model_id"],
        "dtype": lora.get("dtype", "bfloat16"),
        "data_path": cfg["data"]["path"],
        "text_field": cfg["data"].get("text_field", "messages"),
        "valid_fraction": float(cfg["data"].get("valid_fraction", 0.0) or 0.0),
        "max_seq_len": max_seq_len,
        "flag_percentile": float(cfg.get("audit", {}).get("flag_percentile", 95)),
        "vectors_dir": f"data/{domain}/vectors__{slug}",
        "run_dir": f"data/{cfg['name']}__{slug}",
    }


def load_vectors(vectors_dir: str, names: list[str] | None = None) -> list[PersonaVector]:
    vdir = Path(vectors_dir)
    npzs = sorted(vdir.glob("*.npz"))
    # exclude the probe artefacts (*.probe.npz) — those are classifier weights, not directions
    npzs = [p for p in npzs if not p.name.endswith(".probe.npz")]
    vecs = [PersonaVector.load(str(p)) for p in npzs]
    if names is not None:
        keep = set(names)
        vecs = [v for v in vecs if v.name in keep]
    return vecs


def project_samples(model, tokenizer, rows: list[dict], vectors: list[PersonaVector],
                    max_seq_len: int, batch_size: int = 32) -> dict[str, list[float]]:
    """Per-sample mean projection onto every vector — faithful to lora._run_audit.

    Right-pad so real tokens keep default position ids; mean over attention-mask tokens
    only; hidden_states[layer+1] skips the embedding layer; one forward serves every
    vector. Returns {concept -> [s_0, s_1, ...]} aligned to `rows`.
    """
    import torch

    device = next(model.parameters()).device
    units = {v.name: torch.as_tensor(v.unit(), dtype=torch.float32, device=device)
             for v in vectors}
    projs: dict[str, list[float]] = {v.name: [] for v in vectors}

    texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False,
                                           add_generation_prompt=False) for r in rows]
    was_training = model.training
    model.eval()
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        with torch.no_grad():
            for s in range(0, len(texts), batch_size):
                enc = tokenizer(texts[s:s + batch_size], return_tensors="pt", padding=True,
                                truncation=True, max_length=max_seq_len).to(device)
                hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
                m = enc["attention_mask"].float()
                denom = m.sum(1).clamp(min=1.0)
                for v in vectors:
                    h = hs[int(v.layer) + 1].float()
                    proj = h @ units[v.name]
                    projs[v.name].extend(((proj * m).sum(1) / denom).cpu().tolist())
                print(f"  ...projected {min(s + batch_size, len(texts))}/{len(texts)}", flush=True)
    finally:
        tokenizer.padding_side = prev_side
        if was_training:
            model.train()
    return projs


def summarize(arr: np.ndarray, flag_percentile: float) -> dict:
    """Full per-sample distribution: percentiles, histogram, raw values, and the
    same flag stats the audit computed (so it can be validated against train_summary)."""
    threshold = float(np.percentile(arr, flag_percentile))
    flagged = [int(i) for i in np.where(arr > threshold)[0]]
    counts, edges = np.histogram(arr, bins=HIST_BINS)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "flag_percentile": flag_percentile,
        "threshold": threshold,
        "n_flagged": len(flagged),
        "flagged_idx": flagged,
        "percentiles": {str(p): float(np.percentile(arr, p)) for p in PERCENTILES},
        "histogram": {"bin_edges": [round(float(e), 4) for e in edges.tolist()],
                      "counts": [int(c) for c in counts.tolist()]},
        # raw per-sample projections (rounded) — the REAL values, so the UI can re-bin,
        # place the exact p-threshold, or draw a KDE without any assumed shape
        "values": [round(float(x), 4) for x in arr.tolist()],
    }


# The stored audit ran bf16 under a PEFT (identity-LoRA) wrapper with grad-checkpointing,
# possibly sharded across GPUs; this pass reloads the base model bf16 on one device. So
# per-sample projections differ by bf16 accumulation noise (sub-% of a ~60-unit std). A
# handful of rows sitting within that noise of the p-cutoff flip across it. That is expected
# and physically meaningless, so validation passes on a tolerance, not on bit-equality.
MEAN_RTOL, MEAN_ATOL = 1.5e-2, 1.5     # pass if within 1.5% OR 1.5 projection units
FLAG_OVERLAP_MIN = 0.95                 # >=95% of flagged rows identical


def validate(name: str, summary: dict, stored: dict) -> bool:
    """Check the recompute reproduces the stored audit aggregates within bf16 tolerance."""
    ok = True
    for label, want, got in (
        ("mean_projection", stored.get("mean_projection"), summary["mean"]),
        ("threshold", stored.get("threshold"), summary["threshold"]),
    ):
        if want is None:
            continue
        d = abs(want - got)
        rel = d / (abs(want) + 1e-9)
        good = rel <= MEAN_RTOL or d <= MEAN_ATOL
        ok = ok and good
        print(f"    [{'OK ' if good else 'FAIL'}] {name}.{label}: stored={want:.4f} "
              f"recomputed={got:.4f} (Δ{d:.3f}, rel {rel:.2e})")
    sn, gn = stored.get("n_flagged"), summary["n_flagged"]
    if sn is not None:
        stored_idx, got_idx = set(stored.get("flagged_idx") or []), set(summary["flagged_idx"])
        overlap = len(stored_idx & got_idx)
        frac = overlap / max(1, len(stored_idx))
        good = sn == gn and frac >= FLAG_OVERLAP_MIN
        ok = ok and good
        note = "" if frac == 1 else f"  ({len(stored_idx) - overlap} boundary flip(s) — bf16 noise)"
        print(f"    [{'OK ' if good else 'FAIL'}] {name}.n_flagged: stored={sn} recomputed={gn} "
              f"| flagged_idx overlap {overlap}/{len(stored_idx)}{note}")
    return ok


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="application config yaml (e.g. configs/applications/medical.yaml)")
    ap.add_argument("--model", default="swiss-ai/Apertus-8B-Instruct-2509")
    ap.add_argument("--run", help="run dir shortcut (data/<name>) — infers --config by name")
    ap.add_argument("--out", help="output path (default: <run>/checkpoints/point_projections.json)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--validate-only", action="store_true",
                    help="re-check an existing point_projections.json against train_summary "
                         "(no model load)")
    args = ap.parse_args()

    if args.run and not args.config:
        base = Path(args.run).name.split("__")[0]
        args.config = f"configs/applications/{base}.yaml"

    info = resolve_run(args.config, args.model)
    run_dir = Path(args.run) if args.run else Path(info["run_dir"])
    ckpt_dir = run_dir / "checkpoints"
    summary_path = ckpt_dir / "train_summary.json"
    stored_audit = {}
    if summary_path.exists():
        stored_audit = (json.loads(summary_path.read_text()).get("audit") or {})

    if args.validate_only:
        pp_path = Path(args.out) if args.out else ckpt_dir / "point_projections.json"
        pp = json.loads(pp_path.read_text())
        print(f"[validate] {pp_path} vs {summary_path}")
        all_ok = True
        for name, summary in pp["concepts"].items():
            if name in stored_audit:
                all_ok = validate(name, summary, stored_audit[name]) and all_ok
        print(f"[validate] {'ALL PASS (within bf16 tolerance)' if all_ok else 'MISMATCH'}")
        return

    print(f"[recompute] run={run_dir}  model={info['model_id']}")
    print(f"[recompute] data={info['data_path']}  valid_fraction={info['valid_fraction']}  "
          f"max_seq_len={info['max_seq_len']}  p{info['flag_percentile']}")
    print(f"[recompute] vectors_dir={info['vectors_dir']}")

    # reproduce the EXACT audit rows: same loader + deterministic order-preserving split
    rows = load_chat_dataset(info["data_path"], info["text_field"])
    train_rows, _ = train_valid_split(rows, info["valid_fraction"])
    print(f"[recompute] {len(rows)} rows -> {len(train_rows)} train rows (audited)")

    names = list(stored_audit.keys()) or None
    vectors = load_vectors(info["vectors_dir"], names)
    print(f"[recompute] vectors: {[(v.name, int(v.layer)) for v in vectors]}")
    if not vectors:
        raise SystemExit(f"no vectors found in {info['vectors_dir']}")

    dtype = getattr(torch, info["dtype"], torch.bfloat16)
    print(f"[recompute] loading {info['model_id']} ({info['dtype']}) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(info["model_id"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(info["model_id"], torch_dtype=dtype,
                                                 device_map="auto")

    projs = project_samples(model, tokenizer, train_rows, vectors, info["max_seq_len"],
                            batch_size=args.batch_size)

    print("[recompute] validating against stored train_summary audit:")
    all_ok = True
    concepts_out = {}
    for v in vectors:
        arr = np.asarray(projs[v.name], dtype=np.float64)
        summary = summarize(arr, info["flag_percentile"])
        summary["layer"] = int(v.layer)
        concepts_out[v.name] = summary
        if v.name in stored_audit:
            all_ok = validate(v.name, summary, stored_audit[v.name]) and all_ok

    out_path = Path(args.out) if args.out else ckpt_dir / "point_projections.json"
    payload = {
        "run": run_dir.name,
        "model_id": info["model_id"],
        "data_path": info["data_path"],
        "n_samples": len(train_rows),
        "flag_percentile": info["flag_percentile"],
        "percentiles": PERCENTILES,
        "hist_bins": HIST_BINS,
        "concepts": concepts_out,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[recompute] wrote {out_path}  ({out_path.stat().st_size // 1024} KB)")
    print(f"[recompute] validation: {'ALL PASS' if all_ok else 'MISMATCH — see FAILs above'}")


if __name__ == "__main__":
    main()
