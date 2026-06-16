"""Stage 1 — generate trait-specific artifacts from a concept description.

Chen+ 2507.21509 §2.1 / App. A.1: a single meta-prompt instructs a frontier LLM to
emit, per concept, (a) 5 contrastive system-prompt pairs, (b) ~40 elicitation
questions (split into disjoint extraction / evaluation halves), and (c) a judge
rubric. The only human input is the concept name + description.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import yaml

from ftmi.config import Concept
from ftmi.prompts import META_PROMPT, PROPOSE_PROMPT
from ftmi.schemas import ArtifactsOut, ProposedConcepts


@dataclass(frozen=True)
class ConceptArtifacts:
    name: str
    system_prompts: list[dict]   # [{"pos": ..., "neg": ...}, x5]
    extraction_questions: list[str]
    evaluation_questions: list[str]
    judge_prompt: str

    @classmethod
    def from_response(cls, name: str, payload: dict, n_extract: int = 20) -> "ConceptArtifacts":
        qs = payload["questions"]
        return cls(
            name=name,
            system_prompts=payload["instruction"],
            extraction_questions=qs[:n_extract],
            evaluation_questions=qs[n_extract:],
            judge_prompt=payload["eval_prompt"],
        )

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


def generate_artifacts(concept: Concept, generator) -> ConceptArtifacts:
    """Run the meta-prompt through `generator` (an LLM client) and parse the JSON.

    `generator(prompt: str) -> str` returns the model's raw text. Kept abstract so
    the backend (Anthropic / local) is swappable and testable.
    """
    prompt = META_PROMPT.format(name=concept.name, description=concept.description)
    payload = json.loads(generator(prompt, schema=ArtifactsOut))
    return ConceptArtifacts.from_response(concept.name, payload)


# --- optional upstream: bootstrap concept definitions from the dataset itself -----
# Used when no hand-authored use-case descriptions exist: an LLM reads a sample of
# the application's data + a one-line label and proposes the safety-critical axes,
# for a human to trim. Same downstream pipeline (Concept -> generate_artifacts).
# The prompt text lives in ftmi.prompts.vectors.PROPOSE_PROMPT.


def propose_concepts(domain: str, sample: str, generator, n: int = 8) -> list[Concept]:
    """Emit candidate Concepts from a dataset sample (the dataset-audit framing).

    Returns a list a human should review/trim before fitting vectors.
    """
    prompt = PROPOSE_PROMPT.format(domain=domain, sample=sample, n=n)
    payload = json.loads(generator(prompt, schema=ProposedConcepts))
    return [Concept(**c) for c in payload["concepts"]]


def concepts_to_yaml(domain: str, concepts: list[Concept]) -> str:
    """Serialise proposed concepts into the configs/concepts/<domain>.yaml format."""
    return yaml.safe_dump(
        {"domain": domain, "concepts": [c.__dict__ for c in concepts]},
        sort_keys=False, width=88,
    )
