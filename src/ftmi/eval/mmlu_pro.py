"""MMLU-Pro — general capability via the OFFICIAL protocol (not logprob).

`TIGER-AI-Lab/MMLU-Pro/evaluate_from_local.py`: vLLM, 5-shot CoT, temperature 0,
max_new_tokens 2048, then sequential regex extraction of the answer letter. CoT is the
point of MMLU-Pro (it lifts accuracy markedly over direct answer), so we generate rather
than score letter logprobs — which is where BAEM's `mmlu_pro.py` deviated.
"""
from __future__ import annotations

import json
import re

from ftmi.eval.common import apply_chat, iter_jsonl, results_dir, wilson_ci
from ftmi.eval.datasets import (
    build_mmlu_prompt,
    fetch_mmlu_pro,
    mmlu_fewshot_by_category,
)

# Official extraction cascade.
_PATTERNS = [
    re.compile(r"answer is \(?([A-J])\)?"),
    re.compile(r"\.*[aA]nswer:\s*([A-J])"),
    re.compile(r"\b([A-J])\b(?!.*\b[A-J]\b)"),
]


def extract_answer(text: str) -> str | None:
    for pat in _PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1)
    return None


def run_mmlu_pro(engine, adapter_path: str | None, app: str, tag: str, *,
                 limit: int | None = 1500, n_shot: int = 5, max_tokens: int = 2048) -> dict:
    test_p, val_p = fetch_mmlu_pro()
    rows = list(iter_jsonl(test_p))
    if limit is not None:
        rows = rows[:limit]
    fewshot = mmlu_fewshot_by_category(val_p)

    prompts = [apply_chat(engine.tokenizer, build_mmlu_prompt(r, fewshot.get(r["category"], []), n_shot))
               for r in rows]
    print(f"  [mmlu_pro:{tag}] generating {len(prompts)} CoT responses (n_shot={n_shot})…", flush=True)
    gens = engine.generate(prompts, adapter_path=adapter_path, max_tokens=max_tokens, temperature=0.0)

    out_dir = results_dir(app, tag)
    per_row, n_correct = [], 0
    by_cat: dict[str, list[int]] = {}
    for r, g in zip(rows, gens):
        pred = extract_answer(g)
        ok = int(pred == r["answer"])
        n_correct += ok
        by_cat.setdefault(r["category"], []).append(ok)
        per_row.append({"id": r["id"], "category": r["category"], "gold": r["answer"],
                        "pred": pred, "correct": bool(ok)})
    with open(out_dir / "mmlu_pro.jsonl", "w") as f:
        for r in per_row:
            f.write(json.dumps(r) + "\n")

    n = len(rows)
    ci = wilson_ci(n_correct, n)
    summary = {
        "eval": "mmlu_pro", "tag": tag, "n": n,
        "accuracy": round(n_correct / n, 6) if n else None,
        "accuracy_ci95": [round(x, 6) if x is not None else None for x in ci],
        "by_category": {c: round(sum(v) / len(v), 4) for c, v in sorted(by_cat.items())},
        "n_shot": n_shot, "protocol": "official_5shot_cot_vllm",
        "adapter_path": adapter_path or "",
    }
    (out_dir / "mmlu_pro_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  [mmlu_pro:{tag}] acc={summary['accuracy']} (n={n})", flush=True)
    return summary
