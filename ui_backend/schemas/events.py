"""Agent event contract (M1.1).

The single typed envelope the agent layer streams to the front-end over SSE.
Two channels share one stream:

  • rail  — right-panel narration the agent emits: insight | question | action
            | metric | log. Mirrors the front-end's StreamItem registry.
  • stage — left-panel directives a *work* tool emits as a side effect to drive
            the pipeline view: audit_flagged | training_started | training_fill
            | mitigation_morph | done | status.

Discriminated on `kind`. `ref` correlates a question/action with the user's
later POST (answer/action). Payloads stay close to the front-end item shapes;
the front-end api adapter does the final camelCase mapping (M1.4).

Note: the front-end's `steps` item is rendered by the UI itself, not emitted by
the agent, so it is intentionally absent here.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


# ─────────────────────────── rail (right panel) ───────────────────────────

class InsightEvent(BaseModel):
    channel: Literal["rail"] = "rail"
    kind: Literal["insight"] = "insight"
    lead: str | None = None
    bullets: list[str] = []
    think: bool = False
    who: str = "agent"


class QuestionOption(BaseModel):
    label: str
    description: str | None = None
    default: bool = False


class QuestionEvent(BaseModel):
    channel: Literal["rail"] = "rail"
    kind: Literal["question"] = "question"
    ref: str                       # answer POST references this
    question: str
    options: list[QuestionOption]
    multi_select: bool = False
    confirm_label: str | None = None


class ActionEvent(BaseModel):
    channel: Literal["rail"] = "rail"
    kind: Literal["action"] = "action"
    ref: str                       # action POST references this
    title: str
    label: str
    variant: str = "primary"       # primary | good | ghost
    done_label: str | None = None


class MetricEvent(BaseModel):
    channel: Literal["rail"] = "rail"
    kind: Literal["metric"] = "metric"
    value: str | float
    label: str
    tone: Literal["good", "bad"] = "good"


class LogEvent(BaseModel):
    channel: Literal["rail"] = "rail"
    kind: Literal["log"] = "log"
    lines: list[str] = []


# ─────────────────────────── stage (left panel) ───────────────────────────

StageKind = Literal[
    "audit_flagged",      # flagged_idx ready → paint dataset rows red
    "training_started",   # enter the insights step
    "training_fill",      # series payload → fill the plots left→right
    "mitigation_morph",   # biased plots morph left, steered plots enter
    "done",               # a stage finished (carries a summary)
    "status",             # coarse status (running/done/failed)
]
StageView = Literal["setup", "audit", "insights", "checkout"]


class StageEvent(BaseModel):
    channel: Literal["stage"] = "stage"
    kind: StageKind
    view: StageView | None = None
    payload: dict[str, Any] = {}


# ─────────────────────────── the union ───────────────────────────

AgentEvent = Annotated[
    Union[InsightEvent, QuestionEvent, ActionEvent, MetricEvent, LogEvent, StageEvent],
    Field(discriminator="kind"),
]

RAIL_KINDS = ("insight", "question", "action", "metric", "log")

__all__ = [
    "InsightEvent",
    "QuestionEvent",
    "QuestionOption",
    "ActionEvent",
    "MetricEvent",
    "LogEvent",
    "StageEvent",
    "StageKind",
    "StageView",
    "AgentEvent",
    "RAIL_KINDS",
]
