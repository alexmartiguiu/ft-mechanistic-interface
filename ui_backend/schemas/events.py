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


class SubagentStep(BaseModel):
    """One streamed line inside a nested-subagent panel (a tool call or a note).

    `icon="paper"` renders an arXiv badge using `title` (short paper title) and
    `subtitle` (authors + year); other icons just render `text`.
    """
    icon: Literal["search", "read", "paper", "think", "done", "error"] = "think"
    text: str
    title: str | None = None       # paper short title, e.g. "Persona Vectors"
    subtitle: str | None = None    # paper authors + year, e.g. "Chen et al. 2025"


class SubagentEvent(BaseModel):
    """A nested subagent's live trace, rendered indented under its own header.

    Upserted on the front-end by `ref`: a `start` opens the panel, each `step`
    appends a line, `done` closes it (carrying a one-line status). The subagent's
    own tool calls (e.g. the concept-proposer's Read / web_search) become steps.
    """
    channel: Literal["rail"] = "rail"
    kind: Literal["subagent"] = "subagent"
    ref: str                                  # stable id; the front-end upserts on it
    title: str = "Understanding emergent risks"
    agent: str = "concept-proposer"           # subagent type label
    phase: Literal["start", "step", "done"] = "step"
    step: SubagentStep | None = None          # set when phase == "step"
    status: str | None = None                 # set when phase == "done"


# ─────────────────────────── stage (left panel) ───────────────────────────

StageKind = Literal[
    "audit_view",         # concepts proposed → show the audit view (method animates, pre-flag)
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


# ─────────────────────────── config (dev-mode editor) ───────────────────────────

class ConfigEvent(BaseModel):
    """A config file the agent just authored → the dev-mode YAML editor upserts it live.

    `file` is a ConfigFile.model_dump(); `status` (when present) is a ConfigStatus
    dump so the launch gate re-evaluates as configs fill in."""

    channel: Literal["config"] = "config"
    kind: Literal["config"] = "config"
    action: Literal["update", "reset"] = "update"
    file: dict[str, Any] = {}
    status: dict[str, Any] | None = None


# ─────────────────────────── the union ───────────────────────────

AgentEvent = Annotated[
    Union[InsightEvent, QuestionEvent, ActionEvent, MetricEvent, LogEvent, SubagentEvent,
          StageEvent, ConfigEvent],
    Field(discriminator="kind"),
]

RAIL_KINDS = ("insight", "question", "action", "metric", "log", "subagent")

__all__ = [
    "InsightEvent",
    "QuestionEvent",
    "QuestionOption",
    "ActionEvent",
    "MetricEvent",
    "LogEvent",
    "SubagentStep",
    "SubagentEvent",
    "StageEvent",
    "ConfigEvent",
    "StageKind",
    "StageView",
    "AgentEvent",
    "RAIL_KINDS",
]
