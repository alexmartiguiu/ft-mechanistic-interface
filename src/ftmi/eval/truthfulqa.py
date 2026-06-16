"""TruthfulQA MC1 — canonical protocol (Lin et al. 2021 / lm-eval-harness).

6-shot QA primer + `Q: <question>\\nA:` as raw text (no chat template); score each
candidate by the unnormalised sum of conditional token log-probs; MC1 = argmax == correct.
HF logprob (one cheap forward pass per question), run as a separate phase from the vLLM
generation evals.
"""
from __future__ import annotations

import json

from ftmi.eval.common import iter_jsonl, results_dir, wilson_ci
from ftmi.eval.datasets import fetch_truthfulqa
from ftmi.eval.logprob import candidate_logprob_sums
from ftmi.prompts import TRUTHFULQA_PRIMER as QA_PRIMER


def run_truthfulqa(model, tokenizer, app: str, tag: str) -> dict:
    rows = list(iter_jsonl(fetch_truthfulqa()))
    print(f"  [truthfulqa:{tag}] scoring {len(rows)} MC1 questions…", flush=True)

    out_dir = results_dir(app, tag)
    per_row, n_correct = [], 0
    for r in rows:
        prompt = f"{QA_PRIMER}\n\nQ: {r['question']}\nA:"
        scores = candidate_logprob_sums(model, tokenizer, prompt, r["choices"])
        pred = max(range(len(scores)), key=lambda j: scores[j])
        ok = int(pred == int(r["correct_idx"]))
        n_correct += ok
        per_row.append({"id": r["id"], "pred": pred, "correct_idx": r["correct_idx"], "correct": bool(ok)})
    with open(out_dir / "truthfulqa_mc1.jsonl", "w") as f:
        for r in per_row:
            f.write(json.dumps(r) + "\n")

    n = len(rows)
    ci = wilson_ci(n_correct, n)
    summary = {
        "eval": "truthfulqa_mc1", "tag": tag, "n": n,
        "accuracy": round(n_correct / n, 6) if n else None,
        "accuracy_ci95": [round(x, 6) if x is not None else None for x in ci],
        "protocol": "canonical_mc1_qa_primer",
    }
    (out_dir / "truthfulqa_mc1_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  [truthfulqa:{tag}] acc={summary['accuracy']} (n={n})", flush=True)
    return summary
