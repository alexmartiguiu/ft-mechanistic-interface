"""Download + format the insurance-QA SFT set from deccan-ai/insuranceQA-v2.

Source: deccan-ai/insuranceQA-v2 (HF) — columns `input` (the insurance question) and
`output` (a free-form answer). We map each to a single-turn chat example
{"messages":[user=input, assistant=output]}, the format ftmi.data.loaders expects.
Empty/None/duplicate rows and absurdly long answers are dropped; capped at --max rows
for a comparable run scale.

    python scripts/make_insurance_sft.py               # -> data/insurance/sft.jsonl
    python scripts/make_insurance_sft.py --max 2500
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/insurance/sft.jsonl")
    ap.add_argument("--max", type=int, default=2500, help="0 = all rows")
    ap.add_argument("--max_chars", type=int, default=4000, help="drop answers longer than this")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from datasets import load_dataset

    ds = load_dataset("deccan-ai/insuranceQA-v2", split="train")
    seen: set[tuple[str, str]] = set()
    rows = []
    for ex in ds:
        u = (ex.get("input") or "").strip()
        a = (ex.get("output") or "").strip()
        if not u or not a or len(a) > args.max_chars:
            continue
        key = (u, a)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"messages": [{"role": "user", "content": u},
                                  {"role": "assistant", "content": a}]})
        if args.max and len(rows) >= args.max:
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[insurance-sft] wrote {len(rows)} rows -> {out} (from {len(ds)} source rows)")


if __name__ == "__main__":
    main()
