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
import os

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
from ui_backend.models.run import Run
from ui_backend.schemas.events import (
    ActionEvent,
    InsightEvent,
    MetricEvent,
    QuestionEvent,
    QuestionOption,
    StageEvent,
)
from ui_backend.services.pipeline import PipelineService

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


_TOOLS = ["read_dataset", "propose_concepts", "run_audit", "run_training",
          "run_steering", "ask_user", "propose_action"]

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


class AgentSession:
    def __init__(self, sid: str, run_id: int, mode: str = "replay",
                 model_use: str | None = None) -> None:
        self.sid = sid
        self.run_id = run_id
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
            db, svc = sess._svc()
            try:
                dp = svc.dataset_preview(sess.run_id, limit=4)
            finally:
                db.close()
            sample = "\n".join(f"  • user: {(r.get('user') or '')[:140]}\n    assistant: {(r.get('assistant') or '')[:140]}"
                               for r in dp.rows)
            return {"content": [{"type": "text", "text":
                    f"Dataset {dp.domain}/sft.jsonl — {dp.n_rows} chat examples.\nSample:\n{sample}"}]}

        @tool("propose_concepts", "Run the concept-proposer subagent — it researches the "
              "fine-tuning-drift literature and the web, then proposes the risky-behaviour "
              "concepts for this domain (name + description). Use these to build the ask_user "
              "options.", {"type": "object", "properties": {}})
        async def propose_concepts(_args):
            # grounding sample for the subagent (a few user/assistant pairs)
            db, svc = sess._svc()
            try:
                dp = svc.dataset_preview(sess.run_id, limit=6)
            finally:
                db.close()
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
            # the recorded run; live keeps what the subagent actually proposed.
            if sess.mode == "replay":
                db, svc = sess._svc()
                try:
                    cs = svc.propose_concepts(sess.run_id)
                finally:
                    db.close()
                concepts = [c.model_dump() for c in cs]
            else:
                concepts = proposed

            lines = "\n".join(f"  • {c['name']}: {c.get('description') or ''}" for c in concepts)
            text = f"Proposed concepts:\n{lines}"
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
            return {"content": [{"type": "text", "text":
                    "training result — raw numbers, say this in your own words (don't quote it back):\n"
                    f"eval loss bottomed at step {cv.early_stop_step}\n"
                    f"eval battery base→final: {', '.join(deltas) or '(none)'}" + dtxt}]}

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
            finally:
                db.close()
            await sess._emit(StageEvent(kind="mitigation_morph", view="insights", payload=sr.model_dump()))
            hb = next((r for r in sr.comparison.eval if r.metric_key == "harmbench_refusal_v2"), None)
            pp = round((hb.steered - hb.unsteered) * 100) if (hb and hb.steered is not None and hb.unsteered is not None) else None
            if pp is not None:
                await sess._emit(MetricEvent(value=f"+{pp}", label="HarmBench refusal recovered (pp)", tone="good"))
            rows = "; ".join(
                f"{r.label} {r.unsteered}→{r.steered}"
                + (f" (Δ{r.delta:+.3f})" if r.delta is not None else "")
                for r in sr.comparison.eval)
            hb_txt = f"\nharmbench refusal recovered: +{pp} pp" if pp is not None else ""
            return {"content": [{"type": "text", "text":
                    "steering result — raw numbers, say this in your own words (don't quote it back):\n"
                    f"steered {sr.comparison.concept} at layer L{sr.comparison.layer}, "
                    f"coef {sr.comparison.coef}\n"
                    f"eval unsteered→steered: {rows}" + hb_txt}]}

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

        return create_sdk_mcp_server(name="ftmi", version="1.0.0", tools=[
            read_dataset, propose_concepts, run_audit, run_training, run_steering,
            ask_user, propose_action])

    # ── lifecycle ────────────────────────────────────────────────────────────
    async def start(self) -> None:
        from ui_backend.agent.prompts import render_system_prompt

        with SessionLocal() as db:
            run = db.get(Run, self.run_id)
            if run is None:
                raise ValueError(f"run {self.run_id} not found")
            self.domain = run.project.domain
            self.model_label = run.base_model.label
            self.run_title = run.title

        opts = ClaudeAgentOptions(
            system_prompt=render_system_prompt(self.model_use),
            model=agent_model(),
            mcp_servers={"ftmi": self._mcp_server()},
            allowed_tools=[f"mcp__ftmi__{t}" for t in _TOOLS],
            max_turns=60,
        )
        self.client = ClaudeSDKClient(options=opts)
        await self.client.connect()

    async def kickoff(self) -> None:
        self._busy = True
        await self.client.query(
            f"Begin PHASE 1. The user opened the '{self.domain}' run on {self.model_label}. "
            "Greet in one line, call read_dataset, then propose_concepts and ask which concepts to track.")
        asyncio.create_task(self._pump())

    async def send(self, text: str) -> None:
        if self._busy:
            await self.events.put(StageEvent(kind="status", payload={"error": "agent is busy"}))
            return
        self._busy = True
        await self.client.query(text)
        asyncio.create_task(self._pump())

    async def _pump(self) -> None:
        try:
            async for msg in self.client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock) and block.text.strip():
                            lead, bullets = _split_insight(block.text)
                            await self._emit(InsightEvent(lead=lead or None, bullets=bullets))
                        elif isinstance(block, ToolUseBlock):
                            pass  # tool side-effects already emit their own events
                elif isinstance(msg, ResultMessage):
                    await self._emit(StageEvent(kind="status", payload={"turn_done": True}))
        except Exception as e:  # noqa: BLE001
            await self._emit(StageEvent(kind="status", payload={"error": str(e)}))
        finally:
            self._busy = False

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

    def get(self, sid: str) -> AgentSession | None:
        return self._sessions.get(sid)

    async def close(self, sid: str) -> None:
        sess = self._sessions.pop(sid, None)
        if sess is not None:
            await sess.close()


manager = AgentSessionManager()
