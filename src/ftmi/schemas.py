"""Response schemas for the JSON-producing LLM calls.

Passed as `schema=` to a generator (see `ftmi.llm`), they drive provider structured
outputs — Gemini `response_schema`, Anthropic `output_config` json_schema — so the model
is constrained to emit conforming JSON and the caller never regexes over free text.
Pydantic models double as the single source of truth for each payload's shape.
"""
from __future__ import annotations

from pydantic import BaseModel


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


class JudgeScore(BaseModel):
    trait: int
    coherence: int
