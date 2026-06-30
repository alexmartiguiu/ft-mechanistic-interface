"""Enumerations shared by ORM models and Pydantic schemas.

`str, Enum` so they serialize to their value transparently in JSON and store as
VARCHAR in SQLite (we use native_enum=False on the columns).
"""
from __future__ import annotations

import enum


class RunStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    evaluating = "evaluating"
    done = "done"
    failed = "failed"
    canceled = "canceled"


class Direction(str, enum.Enum):
    """Which way is the "safe/better" direction for a metric."""

    up = "up"
    down = "down"


class SteerMethod(str, enum.Enum):
    uniform = "uniform"
    combined = "combined"


class ArtifactKind(str, enum.Enum):
    eval_summary = "eval_summary"            # results/summary.json
    train_summary = "train_summary"          # checkpoints/train_summary.json
    trainer_state = "trainer_state"          # checkpoint-*/trainer_state.json
    eval_detail = "eval_detail"              # results/<tag>/*_summary.json
    eval_samples = "eval_samples"            # results/<tag>/*.jsonl
    sft_dataset = "sft_dataset"              # <domain>/sft.jsonl
    concept_vector_npz = "concept_vector_npz"  # <domain>/vectors/<concept>.npz
    app_config = "app_config"                # configs/applications/<x>.yaml
    run_log = "run_log"                      # logs/*.log
    train_progress = "train_progress"        # checkpoints/progress.jsonl (live: streamed drift)
    audit_json = "audit_json"                # checkpoints/audit.json (live: early audit block)
