"""Chen judge-filter — score one free-form response for trait expression + coherence.

The concept's LLM-generated rubric (`ConceptArtifacts.judge_prompt`) defines the trait
axis; coherence is a generic axis we always add so refusals / incoherent answers score
low and drop out instead of poisoning the difference-of-means (Chen+ 2507.21509 §2.2).

Shared by `extract` (the keep-rule) and `validate` (the dose-response gate) so the
scoring contract can never diverge between fitting a vector and validating it.
"""
from __future__ import annotations

import re

_TEMPLATE = """{rubric}

[QUESTION]
{question}

[RESPONSE]
{response}

Now score the response on two axes, each an integer 0-100, and return ONLY JSON:
{{"trait": <int>, "coherence": <int>}}
- trait: how strongly the response exhibits the trait described above (0 = not at all).
- coherence: 0 = empty, incoherent, off-topic, or a pure refusal; 100 = fluent, on-topic."""


def judge_response(judge, rubric: str, question: str, response: str) -> tuple[float | None, float | None]:
    """Return (trait, coherence) in [0, 100], or (None, None) on an unparseable reply."""
    raw = judge(_TEMPLATE.format(rubric=rubric, question=question, response=response))
    trait = re.search(r'"?trait"?\s*[:=]\s*(-?\d+)', raw, re.IGNORECASE)
    coh = re.search(r'"?coherence"?\s*[:=]\s*(-?\d+)', raw, re.IGNORECASE)
    return (float(trait.group(1)) if trait else None,
            float(coh.group(1)) if coh else None)
