"""Fetch + cache the benchmark datasets to `data/eval/<name>/`.

Sources (HuggingFace `datasets`):
  - TruthfulQA MC1   — `truthful_qa`, config `multiple_choice`, split `validation`
  - MMLU-Pro         — `TIGER-Lab/MMLU-Pro`, splits `test` (questions) + `validation` (CoT few-shot)
  - HarmBench        — behaviors (forbidden behaviors, free-form generation prompts)
  - StrongREJECT     — forbidden prompts

Each fetch is idempotent (skips if the cached JSONL exists).
"""
from __future__ import annotations

from pathlib import Path

from ftmi.eval.common import EVAL_DIR, iter_jsonl, write_jsonl
from ftmi.prompts import MMLU_HEADER

LETTERS = "ABCDEFGHIJ"


# ── TruthfulQA MC1 ────────────────────────────────────────────────────────────

def fetch_truthfulqa() -> Path:
    out = EVAL_DIR / "truthfulqa" / "mc1.jsonl"
    if out.exists():
        return out
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    rows = []
    for i, row in enumerate(ds):
        choices = list(row["mc1_targets"]["choices"])
        labels = list(row["mc1_targets"]["labels"])
        rows.append({"id": f"tqa_mc1_{i:04d}", "question": row["question"],
                     "choices": choices, "correct_idx": labels.index(1)})
    write_jsonl(rows, out)
    print(f"[datasets] wrote {len(rows)} TruthfulQA MC1 rows → {out}", flush=True)
    return out


# ── MMLU-Pro (official CoT) ───────────────────────────────────────────────────

def _mmlu_clean(row: dict) -> dict:
    """Strip trailing N/A padding options (gold letter is unaffected — N/A pads the tail)."""
    options = [o for o in row["options"] if o != "N/A"]
    return {
        "id": f"mmlup_{row['question_id']}",
        "category": row.get("category", "unknown"),
        "question": row["question"],
        "options": options,
        "answer": row["answer"],                 # gold letter "A".."J"
        "cot_content": row.get("cot_content", ""),
    }


def fetch_mmlu_pro() -> tuple[Path, Path]:
    """Return (test_path, validation_path); validation holds the CoT few-shot exemplars."""
    test_p = EVAL_DIR / "mmlu_pro" / "test.jsonl"
    val_p = EVAL_DIR / "mmlu_pro" / "validation.jsonl"
    if test_p.exists() and val_p.exists():
        return test_p, val_p
    from datasets import load_dataset
    for split, path in (("test", test_p), ("validation", val_p)):
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
        rows = [_mmlu_clean(r) for r in ds]
        write_jsonl(rows, path)
        print(f"[datasets] wrote {len(rows)} MMLU-Pro {split} rows → {path}", flush=True)
    return test_p, val_p


def mmlu_fewshot_by_category(val_path: Path) -> dict[str, list[dict]]:
    by_cat: dict[str, list[dict]] = {}
    for r in iter_jsonl(val_path):
        by_cat.setdefault(r["category"], []).append(r)
    return by_cat


def _format_mmlu_example(row: dict, *, with_answer: bool) -> str:
    """Official `format_cot_example` shape."""
    prompt = "Question:\n" + row["question"] + "\nOptions:\n"
    for i, opt in enumerate(row["options"]):
        prompt += f"{LETTERS[i]}. {opt}\n"
    if with_answer:
        cot = (row.get("cot_content") or "").replace("A: Let's think step by step.",
                                                      "Answer: Let's think step by step.")
        if not cot.startswith("Answer:"):
            cot = "Answer: " + cot
        prompt += cot + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def build_mmlu_prompt(test_row: dict, fewshot: list[dict], n_shot: int) -> str:
    """Official MMLU-Pro 5-shot CoT prompt: per-category instruction + exemplars + question."""
    subject = test_row["category"]
    head = MMLU_HEADER.format(subject=subject) + "\n\n"
    shots = "".join(_format_mmlu_example(ex, with_answer=True) for ex in fewshot[:n_shot])
    return head + shots + _format_mmlu_example(test_row, with_answer=False)


# ── Safety prompt sets ────────────────────────────────────────────────────────

# Public, ungated CSV sources (same as BAEM). Avoids HF dataset gating.
_HARMBENCH_CSV = ("https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
                  "data/behavior_datasets/harmbench_behaviors_text_all.csv")
_STRONGREJECT_CSV = ("https://raw.githubusercontent.com/alexandrasouly/strongreject/main/"
                     "strongreject_dataset/strongreject_dataset.csv")


def _fetch_csv(url: str) -> list[dict]:
    import csv
    import io
    import urllib.request
    with urllib.request.urlopen(url) as resp:  # noqa: S310 — trusted GitHub raw URL
        text = resp.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def fetch_harmbench() -> Path:
    """HarmBench standard text behaviors → rows with field `Behavior`."""
    out = EVAL_DIR / "harmbench" / "harmbench.jsonl"
    if out.exists():
        return out
    rows = [{"Behavior": r["Behavior"]} for r in _fetch_csv(_HARMBENCH_CSV)
            if r.get("Behavior") and r.get("FunctionalCategory", "standard") == "standard"]
    write_jsonl(rows, out)
    print(f"[datasets] wrote {len(rows)} HarmBench behaviors → {out}", flush=True)
    return out


def fetch_strongreject() -> Path:
    """StrongREJECT forbidden prompts → rows with field `forbidden_prompt`."""
    out = EVAL_DIR / "strongreject" / "strongreject.jsonl"
    if out.exists():
        return out
    rows = [{"forbidden_prompt": r["forbidden_prompt"]} for r in _fetch_csv(_STRONGREJECT_CSV)
            if r.get("forbidden_prompt")]
    write_jsonl(rows, out)
    print(f"[datasets] wrote {len(rows)} StrongREJECT prompts → {out}", flush=True)
    return out
