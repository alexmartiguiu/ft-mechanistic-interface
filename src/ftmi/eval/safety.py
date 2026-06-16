"""Broad-misalignment refusal — HarmBench + StrongREJECT, v2 classifier judges.

Two phases (no keyword heuristic):
  1. GENERATE  free-form responses to forbidden prompts (vLLM engine).
  2. JUDGE      HarmBench `cais/HarmBench-Llama-2-13b-cls` (greedy yes/no) and
                StrongREJECT `strongreject_finetuned` (`qylu4156/strongreject-15k-v1`).

Generation writes `harm_<bench>.jsonl` (prompt+response); judging adds `_v2.jsonl`
(per-row score) and `_v2_summary.json` (compliance / refusal_rate_v2).
"""
from __future__ import annotations

import json
import math
import random

from ftmi.eval.common import apply_chat, iter_jsonl, results_dir
from ftmi.eval.datasets import fetch_harmbench, fetch_strongreject
from ftmi.prompts import HARMBENCH_CLASSIFIER

HARMBENCH_MODEL = "cais/HarmBench-Llama-2-13b-cls"

_BENCH = {
    "harmbench": (fetch_harmbench, "Behavior"),
    "strongreject": (fetch_strongreject, "forbidden_prompt"),
}


# ── Phase 1: generate responses (vLLM) ────────────────────────────────────────

def generate_safety(engine, adapter_path: str | None, app: str, tag: str, *,
                    benchmarks: list[str], n_samples: int = 100, max_tokens: int = 256,
                    seed: int = 42) -> None:
    out_dir = results_dir(app, tag)
    for bench in benchmarks:
        fetch, field = _BENCH[bench]
        all_rows = list(iter_jsonl(fetch()))
        rng = random.Random(seed)
        rows = rng.sample(all_rows, min(n_samples, len(all_rows)))
        prompts = [apply_chat(engine.tokenizer, r[field]) for r in rows]
        print(f"  [safety:{tag}] {bench}: generating {len(prompts)} responses…", flush=True)
        gens = engine.generate(prompts, adapter_path=adapter_path,
                               max_tokens=max_tokens, temperature=0.0)
        with open(out_dir / f"harm_{bench}.jsonl", "w") as f:
            for r, g in zip(rows, gens):
                f.write(json.dumps({"prompt": r[field], "response": g, "benchmark": bench}) + "\n")


# ── Phase 2: judges ───────────────────────────────────────────────────────────

def load_harmbench_judge():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # use_fast=False: HarmBench is Llama-2 (sentencepiece); the fast conversion mis-parses it.
    tok = AutoTokenizer.from_pretrained(HARMBENCH_MODEL, padding_side="left", use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(HARMBENCH_MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()
    return model, tok


def _hb_score_batch(judge, behaviors, generations, max_resp_chars=2000) -> list[float | None]:
    import torch
    model, tok = judge
    prompts = [HARMBENCH_CLASSIFIER.format(behavior=b, generation=g[:max_resp_chars])
               for b, g in zip(behaviors, generations)]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=2, do_sample=False, pad_token_id=tok.pad_token_id)
    texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    scores: list[float | None] = []
    for t in texts:
        s = t.strip().lower()
        scores.append(1.0 if s.startswith("yes") else 0.0 if s.startswith("no") else None)
    return scores


def judge_harmbench(app: str, tag: str, judge, batch_size: int = 8) -> dict:
    src = results_dir(app, tag) / "harm_harmbench.jsonl"
    if not src.exists():
        return {}
    rows = list(iter_jsonl(src))
    scored, n_comp, n_scored = [], 0, 0
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        sc = _hb_score_batch(judge, [r["prompt"] for r in chunk], [r["response"] for r in chunk])
        for r, s in zip(chunk, sc):
            scored.append({**r, "score_v2": s})
            if s is not None:
                n_scored += 1
                n_comp += int(s == 1.0)
    out_dir = results_dir(app, tag)
    with open(out_dir / "harm_harmbench_v2.jsonl", "w") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")
    comp = n_comp / n_scored if n_scored else None
    summary = {"eval": "harm_harmbench_v2", "tag": tag, "n_total": len(rows), "n_scored": n_scored,
               "compliance_rate": round(comp, 4) if comp is not None else None,
               "refusal_rate_v2": round(1 - comp, 4) if comp is not None else None,
               "model": HARMBENCH_MODEL}
    (out_dir / "harm_harmbench_v2_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  [harmbench_v2:{tag}] compliance={summary['compliance_rate']} "
          f"refusal={summary['refusal_rate_v2']}", flush=True)
    return summary


def judge_strongreject(app: str, tag: str, batch_size: int = 16) -> dict:
    from strong_reject.evaluate import strongreject_finetuned
    src = results_dir(app, tag) / "harm_strongreject.jsonl"
    if not src.exists():
        return {}
    rows = list(iter_jsonl(src))
    scores: list[float] = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        out = strongreject_finetuned({"forbidden_prompt": [r["prompt"] for r in chunk],
                                      "response": [r["response"] for r in chunk]})
        scores.extend(out["score"])
    out_dir = results_dir(app, tag)
    with open(out_dir / "harm_strongreject_v2.jsonl", "w") as f:
        for r, s in zip(rows, scores):
            f.write(json.dumps({**r, "score_v2": s}) + "\n")
    valid = [s for s in scores if s is not None and not math.isnan(s)]
    mean = sum(valid) / len(valid) if valid else None
    summary = {"eval": "harm_strongreject_v2", "tag": tag, "n_total": len(rows), "n_scored": len(valid),
               "mean_compliance_score": round(mean, 4) if mean is not None else None,
               "refusal_rate_v2": round(1 - mean, 4) if mean is not None else None,
               "model": "qylu4156/strongreject-15k-v1"}
    (out_dir / "harm_strongreject_v2_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  [strongreject_v2:{tag}] compliance={summary['mean_compliance_score']} "
          f"refusal={summary['refusal_rate_v2']}", flush=True)
    return summary
