"""Idempotent catalog seed — the fixed reference data the pipeline emits.

Mirrors the metric battery and base models the infra already uses (see
webui/plots.py / src/ftmi/eval). Safe to run repeatedly: it inserts only what's
missing, keyed by natural keys.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ui_backend.models.catalog import (
    BaseModel,
    Benchmark,
    EvalSuite,
    EvalSuiteBenchmark,
    Metric,
)
from ui_backend.models.enums import Direction

BASE_MODELS = [
    ("apertus-8b", "swiss-ai/Apertus-8B-Instruct-2509", "Apertus-8B-Instruct", "8B"),
    ("qwen-7b", "Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct", "7B"),
]

# benchmark key -> (label, kind, protocol, judge_model)
BENCHMARKS = {
    "mmlu_pro": ("MMLU-Pro", "capability", "official_5shot_cot_vllm", None),
    "truthfulqa": ("TruthfulQA MC1", "truthfulness", "canonical_mc1_qa_primer", None),
    "harmbench": ("HarmBench", "safety", "refusal_v2", "cais/HarmBench-Llama-2-13b-cls"),
    "strongreject": ("StrongREJECT", "safety", "refusal_v2", "qylu4156/strongreject-15k-v1"),
}

# metric key -> (benchmark key, label, group, axis, good_when)
METRICS = [
    ("mmlu_pro_acc", "mmlu_pro", "MMLU-Pro", "capability", "metric", Direction.up),
    ("truthfulqa_mc1_acc", "truthfulqa", "TruthfulQA", "capability", "metric", Direction.up),
    ("harmbench_refusal_v2", "harmbench", "HarmBench refusal", "safety", "metric", Direction.up),
    ("strongreject_refusal_v2", "strongreject", "StrongREJECT refusal", "safety", "metric", Direction.up),
    ("train_loss", None, "train loss", "training", "loss", Direction.down),
    ("eval_loss", None, "eval loss", "training", "loss", Direction.down),
]

# eval suite key -> (label, [(benchmark key, n_prompts)])
EVAL_SUITES = {
    "safety_full": (
        "Full safety battery",
        [("mmlu_pro", 1000), ("truthfulqa", 817), ("harmbench", 320), ("strongreject", 313)],
    ),
}


def _get_or_create(session: Session, model, defaults: dict, **keys):
    obj = session.execute(select(model).filter_by(**keys)).scalars().first()
    if obj is None:
        obj = model(**keys, **defaults)
        session.add(obj)
        session.flush()
    return obj


def seed_catalog(session: Session) -> dict[str, int]:
    counts: dict[str, int] = {}

    for id_, repo, label, n in BASE_MODELS:
        _get_or_create(session, BaseModel, {"hf_repo": repo, "label": label, "n_params": n}, id=id_)

    benchmarks: dict[str, Benchmark] = {}
    for key, (label, kind, protocol, judge) in BENCHMARKS.items():
        benchmarks[key] = _get_or_create(
            session,
            Benchmark,
            {"label": label, "kind": kind, "protocol": protocol, "judge_model": judge},
            key=key,
        )

    for key, bkey, label, grp, axis, good in METRICS:
        _get_or_create(
            session,
            Metric,
            {
                "benchmark_id": benchmarks[bkey].id if bkey else None,
                "label": label,
                "grp": grp,
                "axis": axis,
                "good_when": good,
            },
            key=key,
        )

    for key, (label, members) in EVAL_SUITES.items():
        suite = _get_or_create(session, EvalSuite, {"label": label}, key=key)
        for bkey, n_prompts in members:
            _get_or_create(
                session,
                EvalSuiteBenchmark,
                {"n_prompts": n_prompts},
                eval_suite_id=suite.id,
                benchmark_id=benchmarks[bkey].id,
            )

    session.commit()

    counts["base_models"] = session.query(BaseModel).count()
    counts["benchmarks"] = session.query(Benchmark).count()
    counts["metrics"] = session.query(Metric).count()
    counts["eval_suites"] = session.query(EvalSuite).count()
    return counts
