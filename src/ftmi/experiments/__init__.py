"""Experiment orchestration — campaigns built on top of the core pipeline.

Kept separate from `train`/`eval`/`vectors` (the reusable primitives): this package wires
those primitives into multi-run campaigns (e.g. the preventative-steering coefficient
search + matched-density confirmation).
"""
