"""Prompt registry — every hardcoded prompt in the repo, as importable variables.

One package, so the full set of instructions sent to any model is readable at a glance
without grepping the source. Prompts are plain `str` constants; inject variables with
`str.format(...)` at the call site (literal JSON braces are doubled inside the templates).

    from ftmi.prompts import JUDGE_TEMPLATE
    text = JUDGE_TEMPLATE.format(rubric=rubric, question=q, response=r)

Grouped by area in submodules (`vectors`, `eval`) and re-exported here, so either
`from ftmi.prompts import META_PROMPT` or `from ftmi.prompts.vectors import META_PROMPT`
works.
"""
from __future__ import annotations

from ftmi.prompts.eval import HARMBENCH_CLASSIFIER, MMLU_HEADER, TRUTHFULQA_PRIMER
from ftmi.prompts.vectors import (
    CONCEPT_EXTRACT_PROMPT,
    CONCEPT_RESEARCH_PROMPT,
    JUDGE_TEMPLATE,
    META_PROMPT,
    PROPOSE_PROMPT,
)

__all__ = [
    "META_PROMPT",
    "PROPOSE_PROMPT",
    "CONCEPT_RESEARCH_PROMPT",
    "CONCEPT_EXTRACT_PROMPT",
    "JUDGE_TEMPLATE",
    "HARMBENCH_CLASSIFIER",
    "TRUTHFULQA_PRIMER",
    "MMLU_HEADER",
]
