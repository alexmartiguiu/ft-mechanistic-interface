"""Response schemas for the JSON-producing LLM calls.

Passed as `schema=` to a generator (see `ftmi.llm`), they drive provider structured
outputs — Gemini `response_schema`, Anthropic `output_config` json_schema — so the model
is constrained to emit conforming JSON and the caller never regexes over free text.
Pydantic models double as the single source of truth for each payload's shape.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class InstructionPair(BaseModel):
    pos: str
    neg: str


class ArtifactsOut(BaseModel):
    """Stage 1 artifacts: 5 contrastive system-prompt pairs, ~40 questions, judge rubric."""

    instruction: list[InstructionPair]
    questions: list[str]
    eval_prompt: str


class ConceptOut(BaseModel):
    name: str
    description: str


class ProposedConcepts(BaseModel):
    """Wrapper object (not a bare array) so structured output is uniform across providers."""

    concepts: list[ConceptOut]


class GroundedConcept(BaseModel):
    """A proposed safety-critical axis, with the literature-grounded reasoning behind it.

    `name` + `description` are drop-in compatible with `config.Concept` (the rest of the
    pipeline only needs those two); the extra fields are the provenance the proposer adds
    so a human reviewer can see *why* this axis was flagged before minting a vector for it.
    """

    name: str = Field(description="snake_case axis name, e.g. crisis_minimization")
    description: str = Field(
        description="One paragraph describing the FAILURE behaviour (not the desired "
        "behaviour) the model could silently drift toward — the same role as Concept.description."
    )
    drift_mechanism: str = Field(
        description="Why fine-tuning on this domain's data could push the model along this axis."
    )
    evidence: str = Field(
        description="What in the cited fine-tuning / alignment literature supports this being a "
        "real, under-tested risk. Reference papers by short name (e.g. 'persona vectors', "
        "'emergent misalignment')."
    )
    severity: Literal["high", "medium", "low"] = Field(
        description="Triage priority if the developer can only instrument a few axes."
    )


class GroundedConcepts(BaseModel):
    """Structured payload of the grounded extraction step (wrapper, not a bare array)."""

    concepts: list[GroundedConcept]


class JudgeScore(BaseModel):
    trait: int
    coherence: int
