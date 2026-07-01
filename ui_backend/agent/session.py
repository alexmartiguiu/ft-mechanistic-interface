"""Agent ring (M1.3).

One `AgentSession` = one ClaudeSDKClient (Claude Agent SDK over Bedrock). Its
tools call `PipelineService` — so the same narration works on a recorded run
(replay) or a live one — and emit the M1.1 typed events:

  • work tools (run_audit/run_training/run_steering) emit STAGE directives that
    drive the left-panel plots, plus a derived rail METRIC.
  • the agent's prose becomes rail INSIGHT events; ask_user / propose_action emit
    rail QUESTION / ACTION events and park on a Future the HTTP layer resolves.

Sessions are held in-memory by a process-level `AgentSessionManager`. Tools open a
short-lived `SessionLocal` per call (services own the transaction).
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import re
import time

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)
from pydantic import BaseModel

from ui_backend.agent.concept_proposer import ConceptProposalAgent
from ui_backend.core.config import REPO_ROOT, get_settings
from ui_backend.core.database import SessionLocal
from ui_backend.models.catalog import BaseModel as BaseModelRow
from ui_backend.models.project import Project
from ui_backend.models.run import Run
from ui_backend.schemas.events import (
    ActionEvent,
    ConfigEvent,
    InsightEvent,
    MetricEvent,
    QuestionEvent,
    QuestionOption,
    StageEvent,
)
from ui_backend.services.config_authoring import ConfigAuthoringService
from ui_backend.services.pipeline import PipelineService

logger = logging.getLogger("ui_backend.agent")

# ── auth: the SDK reads Bedrock creds from os.environ; load them from the repo
#    .env (non-FTMI_UI_-prefixed, so pydantic-settings ignores them). ──────────
_AUTH_KEYS = (
    "CLAUDE_CODE_USE_BEDROCK",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_REGION",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    # the concept-proposer subagent's web_search (Tavily preferred, Gemini fallback)
    "TAVILY_API_KEY",
    "GEMINI_API_KEY",
    "FTMI_GEN_MODEL",
)


def load_auth_env() -> None:
    env = REPO_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k in _AUTH_KEYS and v and not os.environ.get(k):
            os.environ[k] = v


def agent_model() -> str:
    return (
        os.environ.get("FTMI_UI_AGENT_MODEL")
        or os.environ.get("ANTHROPIC_MODEL")
        or "us.anthropic.claude-sonnet-4-6"
    )


def agent_effort() -> str | None:
    """Reasoning effort for the main agent's turns.

    Left unset, the agent inherited the CLI default (adaptive extended thinking ON),
    which added a multi-second thinking pass to *every* turn — the same latency the
    concept-proposer subagent already disables (see concept_proposer.py). 'low' keeps
    a minimal thinking pass (fastest turns) while preserving more of the careful
    numeric narration a full disable might cost. Override with FTMI_UI_AGENT_EFFORT
    (low|medium|high|max); set it to 'default'/'none' to fall back to the CLI default.
    """
    val = os.environ.get("FTMI_UI_AGENT_EFFORT", "low").strip().lower()
    return None if val in ("", "default", "none") else val


def _timed_tool(tool_obj, sid: str):
    """Wrap an SdkMcpTool's handler to log its wall-clock latency per call.

    Emits one `[agent <sid>] tool <name> <ms>ms` line per call. Note ask_user /
    propose_action block on a user Future, so their timing is wait-inclusive (it
    measures how long the user took, not backend latency); the work tools
    (read_dataset/propose_concepts/run_audit/run_training/run_steering) are pure
    backend/GPU latency.
    """
    handler, name = tool_obj.handler, tool_obj.name

    async def timed(args):
        t0 = time.perf_counter()
        try:
            return await handler(args)
        finally:
            logger.info("[agent %s] tool %s %.0fms", sid, name, (time.perf_counter() - t0) * 1000)

    return dataclasses.replace(tool_obj, handler=timed)


# run session (post-launch): audit → train → steer
_TOOLS = ["read_dataset", "propose_concepts", "run_audit", "run_training",
          "run_steering", "ask_user", "propose_action"]
# authoring session (pre-launch): preview → propose → write configs → offer launch
_AUTHORING_TOOLS = ["read_dataset", "propose_concepts", "set_concepts", "set_lora",
                    "ask_user", "propose_action"]

_BULLET = ("- ", "* ", "• ", "– ")


def _split_insight(text: str) -> tuple[str, list[str]]:
    """Split an agent text block into a lead line + markdown bullet list.

    Lines before the first bullet become the lead; each `- `/`* `/`• ` line is a
    bullet (wrapped continuation lines fold into the previous bullet). With no
    bullet markers the whole block stays the lead, so prose is unchanged.
    """
    lead_parts: list[str] = []
    bullets: list[str] = []
    for raw in text.strip().splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith(_BULLET):
            bullets.append(s[2:].strip())
        elif bullets:
            bullets[-1] = f"{bullets[-1]} {s}".strip()
        else:
            lead_parts.append(s)
    return " ".join(lead_parts).strip(), bullets


def _split_cards(text: str) -> list[str]:
    """Split one agent text block into separate insight cards on a markdown
    horizontal rule (a line of only dashes). Lets the agent land two distinct
    readouts in a single turn — e.g. the benchmark battery, then the
    emergent-risk projections — as two boxes instead of one long one."""
    parts = re.split(r"(?m)^\s*-{3,}\s*$", text)
    return [p for p in (seg.strip() for seg in parts) if p]


class AgentSession:
    def __init__(self, sid: str, run_id: int | None = None, mode: str = "replay",
                 model_use: str | None = None, *, project_id: int | None = None,
                 kind: str = "run") -> None:
        self.sid = sid
        self.run_id = run_id
        self.project_id = project_id     # set for kind == "authoring" (no run yet)
        self.kind = kind                 # "run" (audit→train→steer) | "authoring" (pre-launch)
        self.mode = mode
        self.model_use = model_use  # user's "what does this model do in the world" → system prompt
        # replay: the system prompt is frozen at connect(), so intent the user adds *after* the
        # session starts is injected into the agent's NEXT turn instead (see _consume_intent).
        self._late_intent: str | None = None
        self.events: asyncio.Queue = asyncio.Queue()
        self.client: ClaudeSDKClient | None = None
        self.pending: asyncio.Future | None = None
        self.pending_ref: str | None = None
        self._seq = 0
        self._busy = False
        self._turn_t0 = 0.0   # perf_counter stamp at query time → per-turn wall latency
        self.closed = False
        # static context loaded once
        self.domain = self.model_label = self.run_title = None

    # ── helpers ──────────────────────────────────────────────────────────────
    def _ref(self) -> str:
        self._seq += 1
        return f"{self.sid}-{self._seq}"

    async def _emit(self, ev: BaseModel) -> None:
        await self.events.put(ev)

    def _svc(self):
        """A pipeline service on a fresh session, in this run's mode."""
        db = SessionLocal()
        return db, PipelineService(db, mode=self.mode)

    def _config_svc(self):
        """A config-authoring service on a fresh session (the authoring workspace)."""
        db = SessionLocal()
        return db, ConfigAuthoringService(db)

    def _preview(self, limit: int):
        """Dataset preview, mode-blind: from the run (run kind) or the project (authoring)."""
        if self.kind == "authoring":
            db, auth = self._config_svc()
            try:
                return auth.dataset_preview(self.project_id, limit=limit)
            finally:
                db.close()
        db, svc = self._svc()
        try:
            return svc.dataset_preview(self.run_id, limit=limit)
        finally:
            db.close()

    def _gate_line(self, st) -> str:
        return "READY to launch" if st.launchable else ("not launchable yet — " + "; ".join(st.reasons))

    def set_model_use(self, text: str | None) -> None:
        """Stash deployment context added after the session started; injected next turn."""
        self._late_intent = (text or "").strip() or None

    def _consume_intent(self) -> str:
        """A one-shot context suffix for the agent's next tool return; clears after use."""
        if not self._late_intent:
            return ""
        intent, self._late_intent = self._late_intent, None
        return (f"\n\n(The user just told you what this model is for in the world: {intent}. "
                "Factor this into which drifts matter from here on.)")

    # ── tools (per-session closures) ─────────────────────────────────────────
    def _mcp_server(self):
        sess = self

        @tool("read_dataset", "Preview the dataset (example count + sample rows) to ground yourself. "
              "Call this once at the start.", {"type": "object", "properties": {}})
        async def read_dataset(_args):
            dp = sess._preview(4)
            sample = "\n".join(f"  • user: {(r.get('user') or '')[:140]}\n    assistant: {(r.get('assistant') or '')[:140]}"
                               for r in dp.rows)
            return {"content": [{"type": "text", "text":
                    f"Dataset {dp.domain}/sft.jsonl — {dp.n_rows} chat examples.\nSample:\n{sample}"}]}

        @tool("propose_concepts", "Run the concept-proposer subagent — it researches the "
              "fine-tuning-drift literature and the web, then proposes the risky-behaviour "
              "concepts for this domain (name + description). Use these to build the ask_user "
              "options.", {"type": "object", "properties": {}})
        async def propose_concepts(_args):
            # Move the left panel to the audit view up front, before the research subagent runs,
            # so the concept research reads as AUDIT work (its own thinking beats), not setup. The
            # method animates; the dataset view stays folded until run_audit. Run session only.
            if sess.kind == "run":
                await sess._emit(StageEvent(kind="audit_view", view="audit"))
            # grounding sample for the subagent (a few user/assistant pairs)
            dp = sess._preview(6)
            sample = "\n".join(
                f"  - user: {(r.get('user') or '')[:200]}\n    assistant: {(r.get('assistant') or '')[:200]}"
                for r in dp.rows) or "(no sample rows)"

            # run the proposer subagent — live in BOTH modes, streaming its tool calls
            proposer = ConceptProposalAgent(
                emit=sess._emit, ref=sess._ref(),
                domain=sess.domain or dp.domain, sample=sample)
            result = await proposer.run()   # {"concepts": [...], "findings": str}
            proposed = result.get("concepts") or []
            findings = result.get("findings") or ""

            # mode divergence: replay shows the work but returns the concepts already on
            # the recorded run; authoring prefers the seeded concepts.yaml (an existing topic's
            # curated set already has minted vectors, so the gate can pass) and only falls back
            # to the subagent's fresh proposals for a brand-new topic (no seeded set → mint later);
            # a live run keeps whatever the subagent proposed.
            if sess.mode == "replay":
                db, svc = sess._svc()
                try:
                    cs = svc.propose_concepts(sess.run_id)
                finally:
                    db.close()
                concepts = [c.model_dump() for c in cs]
            elif sess.kind == "authoring":
                db, auth = sess._config_svc()
                try:
                    seeded = auth.current_concepts(sess.project_id)
                finally:
                    db.close()
                concepts = seeded or proposed
            else:
                concepts = proposed

            def _fmt(c):
                tag = "  [recommended — set default=true]" if c.get("recommended") else ""
                return f"  • {c['name']}: {c.get('description') or ''}{tag}"
            lines = "\n".join(_fmt(c) for c in concepts)
            text = f"Proposed concepts:\n{lines}"
            if sess.mode == "replay":
                # in replay these ARE the concepts this run tracked and audited — the same
                # set drawn in the left panel. Offer them all and default them all, and don't
                # leak differently-named axes the findings note might mention.
                text += ("\n\nEvery concept above is one THIS run actually tracked and audited. "
                         "In your ask_user, offer all of them and set default=true on each; cover "
                         "each in your read, and do not introduce concept names not in this list.")
            if findings:
                # the subagent's grounded read — narrate this to the user before ask_user
                text += f"\n\nWhat the proposer found (ground your read on this): {findings}"
            return {"content": [{"type": "text", "text": text}],
                    "structuredContent": {"concepts": concepts, "findings": findings}}

        @tool("run_audit", "Project the concept vectors onto every training sample and flag the "
              "risky ones (drives the dataset audit view). Pass the concepts the user chose.",
              {"type": "object", "properties": {
                  "concepts": {"type": "array", "items": {"type": "string"},
                               "description": "snake_case concept names to track"}},
               "required": ["concepts"]})
        async def run_audit(args):
            tracked = list(args.get("concepts") or [])
            db, svc = sess._svc()
            try:
                # guard (replay): an audit filters by concept name, so names that match none
                # of the recorded run's concepts silently flag 0 samples. If the chosen names
                # miss the tracked set entirely (a stray/renamed axis), audit the full recorded
                # set instead of reporting a misleading "0 flagged".
                if sess.mode == "replay":
                    known = {c.name for c in svc.propose_concepts(sess.run_id)}
                    if tracked and not (set(tracked) & known):
                        tracked = []
                # live: launch the run's job and wait for the early audit to land; replay: no-op
                await svc.execute_audit(sess.run_id, on_event=sess._emit)
                au = svc.audit(sess.run_id, tracked=tracked or None)
            finally:
                db.close()
            await sess._emit(StageEvent(kind="audit_flagged", view="audit", payload=au.model_dump()))
            await sess._emit(MetricEvent(value=au.total_flagged or 0,
                                         label=f"samples flagged · p{au.percentile}", tone="bad"))
            per = ", ".join(f"{c.concept}={c.n_flagged}" for c in au.concepts)
            return {"content": [{"type": "text", "text":
                    "audit result — raw numbers, say this in your own words (don't quote it back):\n"
                    f"flagged: {au.total_flagged} of {au.n_rows} samples, threshold p{au.percentile}\n"
                    f"per concept: {per}"}]}

        @tool("run_training", "Run the monitored fine-tune (drives the live plot fill). Returns the "
              "base→final eval battery and concept drift.", {"type": "object", "properties": {}})
        async def run_training(args):
            db, svc = sess._svc()
            try:
                # switch to the insights view first, then run: live streams incremental fills
                # via on_event while training; replay returns instantly (no-op execute).
                await sess._emit(StageEvent(kind="training_started", view="insights"))
                await svc.execute_training(sess.run_id, on_event=sess._emit)
                cv = svc.train(sess.run_id)
                steerable = svc.steer_targets(sess.run_id)  # concept(s) with a recorded steer
            finally:
                db.close()
            await sess._emit(StageEvent(kind="training_fill", view="insights", payload=cv.model_dump()))
            # raw numbers for the agent to synthesize (so it never invents)
            deltas = []
            for k, series in cv.eval.items():
                base = series[0][1] if series else None
                final = cv.final_metrics.get(k)
                if base is not None and final is not None:
                    deltas.append(f"{k} {base:.3f}→{final:.3f} (Δ{final - base:+.3f})")
            # concept-vector projection drift (interpretability read; NOT probes).
            # The vector points toward the concept, so + = toward risk, - = toward safer.
            drift = [d for d in cv.concept_drift if d.toward_risk is not None]
            dlines = ", ".join(f"{d.concept} {d.toward_risk:+.1f}" for d in drift)
            toward = [d for d in drift if d.toward_risk > 0]
            dtxt = (f"\nconcept-vector projection drift (+ = toward risk, - = toward safer): {dlines}"
                    + (f"\ndrifted most toward risk: {toward[0].concept}" if toward
                       else "\nnothing drifted toward risk — every tracked concept moved toward safer")) if drift else ""
            # replay: constrain the Phase-3 suppression options to the concept(s) we can steer
            # here, but frame it to the agent as the concept chosen to test (not "the only one").
            stxt = (f"\nmitigation to propose (the concept to test a preventive steer on): "
                    f"{', '.join(steerable)}" if steerable else "")
            return {"content": [{"type": "text", "text":
                    "training result — raw numbers, say this in your own words (don't quote it back):\n"
                    f"eval loss bottomed at step {cv.early_stop_step}\n"
                    f"eval battery base→final: {', '.join(deltas) or '(none)'}" + dtxt + stxt}]}

        @tool("run_steering", "Re-train with preventive steering on the chosen concepts (drives the "
              "morph + the steered plots). Returns the steered-vs-base comparison.",
              {"type": "object", "properties": {
                  "concepts": {"type": "array", "items": {"type": "string"}}},
               "required": ["concepts"]})
        async def run_steering(args):
            concepts = list(args.get("concepts") or [])
            db, svc = sess._svc()
            try:
                # live: create + train the steered run, streaming its fills; replay: no-op
                await svc.execute_steering(sess.run_id, concepts=concepts, on_event=sess._emit)
                sr = svc.steer(sess.run_id, concepts=concepts)
                cv_biased = svc.train(sess.run_id)   # biased trajectory, for the projection delta
            finally:
                db.close()
            await sess._emit(StageEvent(kind="mitigation_morph", view="insights", payload=sr.model_dump()))

            # KEY read now: the steered concept's projection %-reduction (v1→v2), computed off
            # the same trajectory finals the plot's green delta band draws — so metric == badge.
            concept = sr.comparison.concept

            def _final_proj(curves, name):
                for t in (curves.trajectory or []):
                    if t.concept == name and t.points:
                        return t.points[-1].projection
                return None

            v1 = _final_proj(cv_biased, concept)
            v2 = _final_proj(sr.curves, concept)
            proj_pct = ((v2 - v1) / abs(v1) * 100) if (v1 not in (None, 0) and v2 is not None) else None
            if proj_pct is not None:
                await sess._emit(MetricEvent(value=f"{proj_pct:+.1f}%",
                                             label=f"{concept.replace('_', ' ')} projection",
                                             tone="good" if proj_pct < 0 else "bad"))
            rows = "; ".join(
                f"{r.label} {r.unsteered}→{r.steered}"
                + (f" (Δ{r.delta:+.3f})" if r.delta is not None else "")
                for r in sr.comparison.eval)
            proj_txt = (f"\nKEY EVAL, {concept} projection: {v1:.1f} → {v2:.1f} ({proj_pct:+.1f}%); "
                        "a reduction means the trait is now less present"
                        if proj_pct is not None else "")
            return {"content": [{"type": "text", "text":
                    "steering result — raw numbers, say this in your own words (don't quote it back):\n"
                    f"steered {concept} at layer L{sr.comparison.layer}, coef {sr.comparison.coef}\n"
                    f"safety + capability eval unsteered→steered: {rows}" + proj_txt +
                    "\n\nRead to give now: state that safety was maintained (or improved) and "
                    "capability was maintained (or improved). Then, most importantly, report that the "
                    f"{concept} projection fell by the percentage above (give the % reduction, not the "
                    "raw delta), which means the trait is now less present in the model. Treat the "
                    "projection as the KEY evaluation at this step."}]}

        @tool("ask_user", "Ask the user one multiple-choice question and wait for their answer. "
              "Put the recommended option first; set default=true on recommended options.",
              {"type": "object", "properties": {
                  "question": {"type": "string"},
                  "options": {"type": "array", "items": {"type": "object", "properties": {
                      "label": {"type": "string"}, "description": {"type": "string"},
                      "default": {"type": "boolean"}}, "required": ["label"]}},
                  "multiSelect": {"type": "boolean"}},
               "required": ["question", "options"]})
        async def ask_user(args):
            ref = sess._ref()
            opts = [QuestionOption(label=o["label"], description=o.get("description"),
                                   default=bool(o.get("default", False))) for o in args["options"]]
            fut = asyncio.get_event_loop().create_future()
            sess.pending, sess.pending_ref = fut, ref
            await sess._emit(QuestionEvent(ref=ref, question=args["question"], options=opts,
                                           multi_select=bool(args.get("multiSelect", False))))
            answer = await fut
            sess.pending = sess.pending_ref = None
            return {"content": [{"type": "text", "text": f"The user selected: {answer}{sess._consume_intent()}"}]}

        @tool("propose_action", "Show the user a single action button, then STOP. The tool returns "
              "only once the user clicks it.",
              {"type": "object", "properties": {
                  "title": {"type": "string"}, "label": {"type": "string"},
                  "variant": {"type": "string", "enum": ["primary", "good", "ghost"]}},
               "required": ["label"]})
        async def propose_action(args):
            ref = sess._ref()
            fut = asyncio.get_event_loop().create_future()
            sess.pending, sess.pending_ref = fut, ref
            await sess._emit(ActionEvent(ref=ref, title=args.get("title", ""), label=args["label"],
                                         variant=args.get("variant", "primary")))
            await fut
            sess.pending = sess.pending_ref = None
            return {"content": [{"type": "text", "text":
                    f"The user clicked '{args['label']}'. Continue.{sess._consume_intent()}"}]}

        # ── authoring-only tools: write the run's YAML configs (drive the dev-mode editor) ──
        @tool("set_concepts", "Persist the concepts to track into the project's concepts.yaml. "
              "Call this right after the user picks which concepts to track — pass each chosen "
              "concept's snake_case name + its one-line description. This updates the config and "
              "the launch gate.", {"type": "object", "properties": {
                  "concepts": {"type": "array", "items": {"type": "object", "properties": {
                      "name": {"type": "string"}, "description": {"type": "string"}},
                      "required": ["name"]}}}, "required": ["concepts"]})
        async def set_concepts(args):
            concepts = [{"name": c["name"], "description": c.get("description") or ""}
                        for c in (args.get("concepts") or [])]
            db, auth = sess._config_svc()
            try:
                cf = auth.set_concepts(sess.project_id, concepts)
                st = auth.status(sess.project_id)
            finally:
                db.close()
            await sess._emit(ConfigEvent(action="update", file=cf.model_dump(), status=st.model_dump()))
            names = ", ".join(c["name"] for c in concepts) or "(none)"
            return {"content": [{"type": "text", "text":
                    f"Wrote concepts.yaml — now tracking {len(concepts)} concept(s): {names}.\n"
                    f"Launch gate: {sess._gate_line(st)}"}]}

        @tool("set_lora", "Set the LoRA training recipe in the project's lora.yaml. Pass a curated "
              "preset name (e.g. apertus8b_default, qwen7b_default) OR `overrides` to patch fields "
              "like {'lora': {'r': 16, 'alpha': 32}, 'optim': {'epochs': 3}}.",
              {"type": "object", "properties": {
                  "preset": {"type": "string"},
                  "overrides": {"type": "object"}}})
        async def set_lora(args):
            db, auth = sess._config_svc()
            try:
                cf = auth.set_lora(sess.project_id, preset=args.get("preset"),
                                   overrides=args.get("overrides"))
                st = auth.status(sess.project_id)
            finally:
                db.close()
            await sess._emit(ConfigEvent(action="update", file=cf.model_dump(), status=st.model_dump()))
            return {"content": [{"type": "text", "text":
                    f"Updated lora.yaml. Launch gate: {sess._gate_line(st)}"}]}

        by_name = {
            "read_dataset": read_dataset, "propose_concepts": propose_concepts,
            "run_audit": run_audit, "run_training": run_training, "run_steering": run_steering,
            "set_concepts": set_concepts, "set_lora": set_lora,
            "ask_user": ask_user, "propose_action": propose_action,
        }
        names = _AUTHORING_TOOLS if sess.kind == "authoring" else _TOOLS
        return create_sdk_mcp_server(
            name="ftmi", version="1.0.0",
            tools=[_timed_tool(by_name[n], sess.sid) for n in names])

    # ── lifecycle ────────────────────────────────────────────────────────────
    async def start(self) -> None:
        from ui_backend.agent.prompts import render_authoring_prompt, render_system_prompt

        with SessionLocal() as db:
            if self.kind == "authoring":
                project = db.get(Project, self.project_id)
                if project is None:
                    raise ValueError(f"project {self.project_id} not found")
                self.domain = project.domain
                self.run_title = project.name
                model_id = ConfigAuthoringService(db)._model_id(project)
                bm = db.get(BaseModelRow, model_id) if model_id else None
                self.model_label = bm.label if bm else (model_id or "the base model")
                system_prompt = render_authoring_prompt(self.model_use)
                allowed = _AUTHORING_TOOLS
            else:
                run = db.get(Run, self.run_id)
                if run is None:
                    raise ValueError(f"run {self.run_id} not found")
                self.domain = run.project.domain
                self.model_label = run.base_model.label
                self.run_title = run.title
                system_prompt = render_system_prompt(self.model_use)
                allowed = _TOOLS

        opts = ClaudeAgentOptions(
            system_prompt=system_prompt,
            model=agent_model(),
            # 'low' effort minimises the adaptive-thinking pass that otherwise ran on
            # every turn (the biggest per-turn latency lever); env-overridable.
            effort=agent_effort(),
            mcp_servers={"ftmi": self._mcp_server()},
            allowed_tools=[f"mcp__ftmi__{t}" for t in allowed],
            max_turns=60,
        )
        self.client = ClaudeSDKClient(options=opts)
        await self.client.connect()

    async def kickoff(self) -> None:
        self._busy = True
        if self.kind == "authoring":
            msg = (f"Begin. The user is setting up a new '{self.domain}' fine-tune on "
                   f"{self.model_label}. Greet in one line, call read_dataset, then "
                   "propose_concepts and give your read. Ask which concepts to track, then call "
                   "set_concepts with their pick. Recommend a LoRA recipe and call set_lora. "
                   "Finally, once the gate is ready, offer to launch.")
        else:
            msg = (f"Begin PHASE 1. The user opened the '{self.domain}' run on {self.model_label}. "
                   "Greet in one line naming the domain and model, call read_dataset, then note in "
                   "one short line the dataset and the run's configuration shown on the left (base "
                   "model + LoRA recipe). Then call propose_action(label=\"Proceed to audit\") and "
                   "STOP. Do NOT call propose_concepts until that button returns — that click is the "
                   "gate into the audit.")
        self._turn_t0 = time.perf_counter()
        await self.client.query(msg)
        asyncio.create_task(self._pump())

    async def send(self, text: str) -> None:
        if self._busy:
            await self.events.put(StageEvent(kind="status", payload={"error": "agent is busy"}))
            return
        self._busy = True
        self._turn_t0 = time.perf_counter()
        await self.client.query(text)
        asyncio.create_task(self._pump())

    async def _pump(self) -> None:
        n_text = n_tools = 0
        try:
            async for msg in self.client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock) and block.text.strip():
                            # one card per `---`-separated segment (usually just one)
                            for segment in _split_cards(block.text):
                                n_text += 1
                                lead, bullets = _split_insight(segment)
                                await self._emit(InsightEvent(lead=lead or None, bullets=bullets))
                        elif isinstance(block, ToolUseBlock):
                            n_tools += 1  # tool side-effects already emit their own events
                elif isinstance(msg, ResultMessage):
                    self._log_turn(msg, n_text, n_tools)
                    await self._emit(StageEvent(kind="status", payload={"turn_done": True}))
        except Exception as e:  # noqa: BLE001
            await self._emit(StageEvent(kind="status", payload={"error": str(e)}))
        finally:
            self._busy = False

    def _log_turn(self, msg: ResultMessage, n_text: int, n_tools: int) -> None:
        """One latency line per turn, from the SDK's own ResultMessage metrics.

        wall   — server-measured, query→result (what the user actually waits)
        sdk    — SDK duration_ms (model + in-turn tool time)
        api    — duration_api_ms (pure model API time; wall−api ≈ tool/queue/overhead)
        turns  — internal tool round-trips this turn
        tokens — in/out + cache_read (cache_read≈in means the big system prompt is cached)
        """
        wall = (time.perf_counter() - self._turn_t0) * 1000 if self._turn_t0 else 0.0
        u = msg.usage or {}
        cost = f"${msg.total_cost_usd:.4f}" if msg.total_cost_usd is not None else "$?"
        logger.info(
            "[agent %s] turn: wall=%.0fms sdk=%sms api=%sms turns=%s "
            "in=%s out=%s cache_read=%s %s text=%s tools=%s%s",
            self.sid, wall, msg.duration_ms, msg.duration_api_ms, msg.num_turns,
            u.get("input_tokens"), u.get("output_tokens"), u.get("cache_read_input_tokens"),
            cost, n_text, n_tools, " ERROR" if msg.is_error else "",
        )

    def resolve(self, ref: str, value) -> bool:
        """Resolve the in-flight question/action (answer or click)."""
        if self.pending is None or self.pending.done() or (ref and ref != self.pending_ref):
            return False
        val = value if isinstance(value, str) else ", ".join(map(str, value or []))
        self.pending.set_result(val)
        return True

    async def close(self) -> None:
        self.closed = True
        if self.client is not None:
            try:
                await self.client.disconnect()
            except Exception:
                pass
            self.client = None


class AgentSessionManager:
    """Process-level registry of live sessions (in-memory for M1)."""

    def __init__(self) -> None:
        self._sessions: dict[str, AgentSession] = {}
        self._seq = 0

    async def create(self, run_id: int, mode: str = "replay",
                     model_use: str | None = None) -> AgentSession:
        load_auth_env()
        self._seq += 1
        sid = f"s{self._seq}"
        sess = AgentSession(sid, run_id, mode=mode, model_use=model_use)
        await sess.start()
        self._sessions[sid] = sess
        await sess.kickoff()
        return sess

    async def create_authoring(self, project_id: int,
                               model_use: str | None = None) -> AgentSession:
        """A pre-launch authoring session bound to a live PROJECT (no run yet)."""
        load_auth_env()
        self._seq += 1
        sid = f"s{self._seq}"
        sess = AgentSession(sid, run_id=None, mode="live", model_use=model_use,
                            project_id=project_id, kind="authoring")
        await sess.start()
        self._sessions[sid] = sess
        await sess.kickoff()
        return sess

    def get(self, sid: str) -> AgentSession | None:
        return self._sessions.get(sid)

    async def close(self, sid: str) -> None:
        sess = self._sessions.pop(sid, None)
        if sess is not None:
            await sess.close()


manager = AgentSessionManager()
