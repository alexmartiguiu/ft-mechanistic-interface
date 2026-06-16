"""Chen judge-filter — score one free-form response for trait expression + coherence.

The concept's LLM-generated rubric (`ConceptArtifacts.judge_prompt`) defines the trait
axis; coherence is a generic axis we always add so refusals / incoherent answers score
low and drop out instead of poisoning the difference-of-means (Chen+ 2507.21509 §2.2).

Shared by `extract` (the keep-rule) and `validate` (the dose-response gate) so the
scoring contract can never diverge between fitting a vector and validating it.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from ftmi.prompts import JUDGE_TEMPLATE
from ftmi.schemas import JudgeScore


def judge_response(judge, rubric: str, question: str, response: str) -> tuple[float | None, float | None]:
    """Return (trait, coherence) in [0, 100], or (None, None) on an unparseable reply.

    The judge is called with `schema=JudgeScore`, so the backend returns JSON conforming
    to {trait, coherence} that we load directly.
    """
    raw = judge(JUDGE_TEMPLATE.format(rubric=rubric, question=question, response=response),
                schema=JudgeScore)
    try:
        d = json.loads(raw)
        return float(d["trait"]), float(d["coherence"])
    except (ValueError, TypeError, KeyError):
        return (None, None)


def judge_batch(judge, rubric: str, qa_pairs, *, concurrency: int = 8) -> list[tuple]:
    """Parallel judge over (question, response) pairs -> list of (trait, coherence).

    Errors (rate limits, timeouts) degrade to (None, None) for that item instead of
    crashing the run — important for long unattended sweeps; the item just drops out.
    """
    def _one(qa):
        try:
            return judge_response(judge, rubric, qa[0], qa[1])
        except Exception:
            return (None, None)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        return list(ex.map(_one, qa_pairs))
