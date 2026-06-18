"""Build data/jailbreak/sft.jsonl from WildJailbreak (ungated mirror heegyu/wildjailbreak-train).

Research question: does fine-tuning a model to REFUSE adversarial jailbreaks inadvertently
misalign it (over-refusal, preachiness, reduced helpfulness)? So we reproduce the realistic
safety-training mix — refuse harmful (incl. jailbroken) prompts, comply with benign (incl.
adversarial-looking-but-benign) — and monitor misalignment axes per checkpoint.

Prompt = the adversarial (jailbreak-styled) variant when present, else the vanilla request;
target = the dataset `completion` (refusal for *_harmful, helpful answer for *_benign).
Balanced subset (`--per-type` rows of each of the 4 data_types). Chat-JSONL.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

REPO = "heegyu/wildjailbreak-train"
SHARDS = ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"]


def _prompt(row) -> str:
    adv = row.get("adversarial")
    if isinstance(adv, str) and adv.strip() and adv.strip().lower() != "nan":
        return adv.strip()
    v = row.get("vanilla")
    return v.strip() if isinstance(v, str) else ""


def build(per_type: int, out: str, seed: int = 0) -> None:
    import pandas as pd
    from huggingface_hub import hf_hub_download
    tok = os.getenv("HF_TOKEN")
    df = pd.concat([pd.read_parquet(hf_hub_download(REPO, s, repo_type="dataset", token=tok))
                    for s in SHARDS], ignore_index=True)
    rng = random.Random(seed)
    rows, cnt = [], Counter()
    for dt, grp in df.groupby("data_type"):
        idx = list(grp.index)
        rng.shuffle(idx)
        for i in idx[:per_type]:
            r = grp.loc[i]
            p, c = _prompt(r), str(r["completion"]).strip()
            if p and c and c.lower() != "nan":
                rows.append({"messages": [{"role": "user", "content": p},
                                          {"role": "assistant", "content": c}]})
                cnt[dt] += 1
    rng.shuffle(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[jailbreak] wrote {len(rows)} rows → {out}; per-type {dict(cnt)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=750, dest="per_type")
    ap.add_argument("--out", default="data/jailbreak/sft.jsonl")
    a = ap.parse_args()
    build(a.per_type, a.out)
