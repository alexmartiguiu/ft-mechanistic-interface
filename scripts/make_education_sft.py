"""Build data/education/sft.jsonl from the JorGPT student-answer grading dataset
(Kaggle: javiersanchezsoriano/jorgpt-student-answers-and-multi-llm-grading).

Task: train the model to GRADE — given (question, reference answer, student answer),
produce a 0–10 score + feedback. Supervision is the human gold (`teacher_grade` /
`teacher_feedback`); the LLM-grader columns are ignored. Chat-JSONL, one turn each.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SYSTEM = ("You are an expert teacher grading a student's exam answer. Compare the answer "
          "to the reference, assign a score from 0 to 10, and give concise, constructive "
          "feedback.")


def build(src: str, out: str, max_rows: int | None) -> None:
    rows = list(csv.DictReader(open(src)))
    kept = []
    for r in rows:
        q, ideal, ans = r.get("question_text", ""), r.get("ideal_answer", ""), r.get("student_answer", "")
        grade, fb = r.get("teacher_grade", ""), r.get("teacher_feedback", "")
        if not (q.strip() and ans.strip() and grade.strip() and fb.strip()):
            continue
        try:
            g = float(grade)
        except ValueError:
            continue
        user = (f"Question:\n{q.strip()}\n\nReference answer:\n{ideal.strip()}\n\n"
                f"Student answer:\n{ans.strip()}")
        assistant = f"Score: {g:g}/10\n\nFeedback: {fb.strip()}"
        kept.append({"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]})
    if max_rows:
        kept = kept[:max_rows]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for k in kept:
            f.write(json.dumps(k) + "\n")
    print(f"[education] wrote {len(kept)} rows → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/education/raw/dataset_en.csv")
    ap.add_argument("--out", default="data/education/sft.jsonl")
    ap.add_argument("--max", type=int, default=None)
    a = ap.parse_args()
    build(a.src, a.out, a.max)
