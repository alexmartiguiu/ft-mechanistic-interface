"""Checkpoint evaluation harness.

Runs a small, faithful behavioural battery over every saved checkpoint (plus the base
model): general capability (MMLU-Pro, official 5-shot CoT), truthfulness (TruthfulQA MC1),
and broad-misalignment refusal (HarmBench + StrongREJECT, v2 classifier judges). Ported
and trimmed from the BAEM eval suite; the runner lives in `ftmi.eval.harness`.
"""
