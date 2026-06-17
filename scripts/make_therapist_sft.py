"""Download + format the LLM-as-therapist SFT set from real licensed-counsellor Q&A.

Source: Amod/mental_health_counseling_conversations (HF) — columns Context (the help
seeker's message) and Response (a licensed therapist's reply). We map each row to a
single-turn chat example {"messages": [user=Context, assistant=Response]}, the format
ftmi.data.loaders expects. Empty/duplicate rows are dropped.

    python scripts/make_therapist_sft.py                 # -> data/therapist/sft.jsonl
    python scripts/make_therapist_sft.py --max 3000      # cap rows
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/therapist/sft.jsonl")
    ap.add_argument("--max", type=int, default=0, help="0 = all rows")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from datasets import load_dataset

    ds = load_dataset("Amod/mental_health_counseling_conversations", split="train")
    seen: set[tuple[str, str]] = set()
    rows = []
    for ex in ds:
        u = (ex.get("Context") or "").strip()
        a = (ex.get("Response") or "").strip()
        if not u or not a:
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
    print(f"[therapist-sft] wrote {len(rows)} rows -> {out} "
          f"(deduped from {len(ds)} source rows)")


if __name__ == "__main__":
    main()
