"""hedda config-authoring agent — Claude Agent SDK over Amazon Bedrock.

Step 1 of the SOC contract (see webui/frontend_redesign/SOC.md): a live agent that
converses with the user to produce the *three* config files a run needs — application,
concept set, LoRA recipe — and nothing else. It is **propose-only**: it asks the user
multiple-choice questions (surfaced in the UI) and only writes files through one explicit
tool. The actual training pipeline (Step 2) is untouched.

Runtime: `claude` CLI via claude-agent-sdk, pointed at Bedrock by the env the launcher
sources from .env (CLAUDE_CODE_USE_BEDROCK=1, AWS_BEARER_TOKEN_BEDROCK, AWS_REGION).

Two custom in-process tools:
  • ask_user(question, options, …)  — blocks until the UI POSTs the user's answer.
  • write_configs(spec)             — validates via ftmi.config dataclasses, writes the
                                      three YAMLs + a per-session snapshot.

Every session's transcript is logged to data/conversations/<sid>.json.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path

import yaml

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
CONV_DIR = DATA / "conversations"

# Bedrock model (cross-region inference profile). Sonnet 4.6 is the right balance for a
# conversational config-authoring agent. Overridable via env.
os.environ.setdefault("ANTHROPIC_MODEL", "us.anthropic.claude-sonnet-4-6")
MODEL = os.environ["ANTHROPIC_MODEL"]

# ── recipe defaults (so the agent can emit a valid contract with minimal input) ──────
MODEL_PRESETS = {
    "qwen-7b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "apertus-8b": {
        "model_id": "swiss-ai/Apertus-8B-Instruct-2509",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],  # non-gated MLP
    },
}
LORA_DEFAULTS = {"r": 16, "alpha": 32, "dropout": 0.05}
OPTIM_DEFAULTS = {"lr": 1.0e-4, "epochs": 3, "batch_size": 8, "grad_accum": 1,
                  "warmup_ratio": 0.03, "max_seq_len": 2048}
CHECKPOINT_DEFAULTS = {"save_every_steps": None, "n_checkpoints": 10,
                       "monitor_every_steps": 25, "log_every_steps": 10, "keep_last": None}

SYSTEM_PROMPT = """You are **hedda**, a careful AI-safety researcher running a narrow LoRA fine-tune \
with the user in TWO simple phases. Be brief and direct: short sentences, one question at a time. You \
are REACTIVE — after you propose a launch you STOP and wait; you only continue when the user speaks \
again. Base models available: `apertus-8b` (Apertus-8B-Instruct) and `qwen-7b` (Qwen2.5-7B-Instruct).

PHASE 1 — TRAINING. The user dropped a dataset. Greet in ONE line. Then, via `ask_user`, one question \
at a time:
  1. base model — options: 'apertus-8b' (recommended), 'qwen-7b'.
  2. LoRA recipe — options: 'Balanced · r16 α32 · 3 epochs' (recommended), 'Light · r8 α16', 'Heavy · r64 α128'.
When both are answered, say ONE short line (e.g. "Ready — launch the training run when you are.") and \
call `propose_launch(phase="training")`. Then STOP. Do not ask or say anything else.

PHASE 2 — INTERPRETABILITY. When the user speaks again after the training run (e.g. they ask "are we \
ready to ship?"), say in ONE line that we should run interpretability first to see what drifted. Then run \
TWO `ask_user` steps, in order:
  STEP A — analysis type (single-select). "Which interpretability analysis should we run?" options: \
'Concept-vector drift monitor' (recommended), 'Logistic probe classifier', 'Both'.
  YOU MUST DO THIS STEP !!!! : STEP B — concepts (REQUIRED, multiSelect=true). PROPOSE WHICH SAFETY CONCEPTS to interpret on, one \
option per concept axis (label = the snake_case concept name, description = the one-line drift it detects), \
with `default=true` on EVERY recommended axis so they start PRE-SELECTED. For a medical run propose the \
five medical-safety axes (all default=true): dangerous_advice, red_flag_minimization, false_credentialing, \
overconfident_certainty, medical_misinformation. The user adjusts the selection — this is their say in what \
gets interpreted/steered.
Never skip STEP B. After the user confirms the concepts, say ONE short line naming them and call \
`propose_launch(phase="interpretability")`. Then STOP.

REVIEW. When the user asks about the results after the interpretability run, call `read_run_results`, then \
give a brief THREE-beat readout:
  DETECTION — name the concept(s) that drifted most and roughly when in training.
  INSIGHT — one line: the loss looked clean, so without these projections this model ships with the drift \
baked in; quantify the worst delta (e.g. "+29 on overconfident_certainty while HarmBench refusal fell 38%").
  ACTION — via `ask_user`, offer (1) preventive-steering fix (recommended), (2) early-stop at the last \
clean checkpoint, (3) ship as-is; recommend what the evidence supports.
If the user picks the preventive-steering fix, do NOT steer silently: first PROPOSE WHICH CONCEPTS TO \
SUPPRESS with `ask_user` (multiSelect=true), one option per drifting concept (label = concept name, \
description = its measured drift), with `default=true` on the worst-drifting axes so they start \
pre-selected. The user adjusts which directions get steered. After they confirm, say ONE short line \
summarising the steering plan, then call `propose_launch(phase="steering")`. Then STOP.

Use ONLY these tools: `ask_user`, `read_dataset`, `propose_launch`, `read_run_results`. Never write files. \
After ANY `propose_launch`, END YOUR TURN."""


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")


def _norm_model(m) -> str:
    """Map any base-model spelling the agent might use to a known preset key; default to
    apertus-8b (the demo base) so write_configs never crashes on an unrecognised name."""
    s = (m or "").lower()
    if "qwen" in s:
        return "qwen-7b"
    if "apertus" in s:
        return "apertus-8b"
    return m if m in MODEL_PRESETS else "apertus-8b"


def _as_dict(x) -> dict:
    """The model occasionally serializes object args as JSON strings. Coerce to a dict
    (or {}), so write_configs never crashes on `lora`/`optim`/`mitigate` being a string."""
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            v = json.loads(x)
            return v if isinstance(v, dict) else {}
        except Exception:
            return {}
    return {}


def _canonical_descs(domain: str) -> dict:
    """Descriptions from the existing concept set for a domain, to backfill any the agent
    sends as bare names."""
    cf = CONFIGS / "concepts" / f"{_slug(domain)}.yaml"
    out: dict[str, str] = {}
    if cf.exists():
        try:
            for c in (yaml.safe_load(cf.read_text()) or {}).get("concepts", []):
                if isinstance(c, dict) and c.get("name"):
                    out[_slug(c["name"])] = (c.get("description") or "").strip()
        except Exception:
            pass
    return out


def _coerce_concepts(raw, domain: str) -> list[dict]:
    """Normalise the agent's `concepts` arg into [{name, description}] no matter how the
    model serialized it — list of dicts, list of strings, or a JSON/comma string. Missing
    descriptions are backfilled from the domain's canonical concept set, else generated."""
    items = raw
    if isinstance(items, str):
        try:
            items = json.loads(items)
        except Exception:
            items = [s for s in re.split(r"[,\n]", items) if s.strip()]
    if not isinstance(items, list):
        items = []
    canon = _canonical_descs(domain)
    out, seen = [], set()
    for c in items:
        if isinstance(c, str):
            name, desc = _slug(c), ""
        elif isinstance(c, dict):
            name = _slug(c.get("name") or c.get("concept") or "")
            desc = (c.get("description") or c.get("desc") or "").strip()
        else:
            continue
        if not name or name in seen:
            continue
        seen.add(name)
        out.append({"name": name, "description": desc or canon.get(name) or f"Drift toward {name.replace('_', ' ')}."})
    return out


def _dataset_preview(rel_path: str, limit: int = 8) -> dict:
    """Read the first few SFT examples for the dataset artifact, flattening each example's
    chat turns into user/assistant columns; counts total examples by line. Returns a shape
    the DatasetArtifact renders directly ({path, found, n, columns, rows})."""
    columns = [{"key": "user", "label": "user"}, {"key": "assistant", "label": "assistant"}]
    path = ROOT / rel_path
    rows: list[dict] = []
    n = 0
    try:
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                n += 1
                if len(rows) >= limit:
                    continue
                try:
                    msgs = json.loads(line).get("messages", [])
                except Exception:
                    continue
                row: dict = {}
                for m in msgs:
                    role = m.get("role")
                    if role in ("user", "assistant") and role not in row:
                        row[role] = (m.get("content") or "").strip()
                if row:
                    rows.append(row)
    except FileNotFoundError:
        return {"path": rel_path, "found": False, "n": 0, "columns": columns, "rows": []}
    return {"path": rel_path, "found": True, "n": n, "columns": columns, "rows": rows}


def _plots():
    try:
        from webui import plots
    except ImportError:
        import plots
    return plots


def run_results(dataset: str, model: str) -> dict:
    """Base→final drift for a completed run: the eval battery (accuracy/refusal) plus the
    per-concept projection shift ⟨h, v̂⟩. Read from the recorded checkpoints via plots.py so
    the agent reasons over the SAME numbers the charts show. base = first point, final = last."""
    p = _plots()
    ev = p.eval_series(dataset, model)          # {metric: [(step, val)]}
    mon = p.monitor_series(dataset, model)      # {concept: [(step, proj)]}
    label = {k: lbl for k, lbl in p.METRICS}
    metrics = []
    for key, series in ev.items():
        if not series:
            continue
        base, final = series[0][1], series[-1][1]
        metrics.append({"key": key, "label": label.get(key, key),
                        "base": round(base, 4), "final": round(final, 4),
                        "delta": round(final - base, 4)})
    concepts = []
    for name, series in sorted(mon.items()):
        if not series:
            continue
        base, final = series[0][1], series[-1][1]
        concepts.append({"name": name, "base": round(base, 3), "final": round(final, 3),
                         "drift": round(final - base, 3)})
    return {"dataset": dataset, "model": model, "metrics": metrics, "concepts": concepts}


class ConfigAgentSession:
    """One chat thread = one ClaudeSDKClient. Pumps SDK messages into an event queue the
    SSE endpoint drains; `ask_user` parks on a Future the answer endpoint resolves."""

    def __init__(self, sid: str, dataset: str | None = None, model: str | None = None):
        self.sid = sid
        self.dataset = dataset                        # domain of the dropped dataset (e.g. 'medical')
        self.model = model                            # base model id (e.g. 'apertus-8b')
        self.events: asyncio.Queue = asyncio.Queue()
        self.client: ClaudeSDKClient | None = None
        self.pending: asyncio.Future | None = None   # in-flight ask_user
        self.transcript: list[dict] = []
        self.configs: dict | None = None             # paths written by write_configs
        self.started = time.time()
        self._busy = False

    # ── transcript log (data/conversations/<sid>.json) ──────────────────────────────
    def _log(self, role: str, kind: str, payload):
        self.transcript.append({"t": round(time.time() - self.started, 2),
                                "role": role, "kind": kind, "payload": payload})
        CONV_DIR.mkdir(parents=True, exist_ok=True)
        (CONV_DIR / f"{self.sid}.json").write_text(json.dumps(
            {"sid": self.sid, "model": MODEL, "started": self.started,
             "configs": self.configs, "transcript": self.transcript}, indent=2))

    async def _emit(self, ev: dict):
        """Push an event to the SSE stream (and log the durable ones)."""
        if ev["type"] in ("text", "question", "configs", "status", "thinking"):
            self._log("agent", ev["type"], ev.get("data"))
        await self.events.put(ev)

    # ── custom tools (per-session closures so they can reach `self`) ─────────────────
    def _mcp_server(self):
        session = self

        @tool("ask_user",
              "Ask the user a single multiple-choice question and wait for their answer. "
              "Use this for every decision the user should make (domain, base model, which "
              "concepts, LoRA recipe choices). Always include a clearly recommended default "
              "as the first option.",
              {
                  "type": "object",
                  "properties": {
                      "question": {"type": "string", "description": "The question to ask."},
                      "header": {"type": "string", "description": "Short chip label, e.g. 'Base model'."},
                      "options": {
                          "type": "array",
                          "description": "2-6 choices; put the recommended one first.",
                          "items": {"type": "object", "properties": {
                              "label": {"type": "string"},
                              "description": {"type": "string"},
                              "default": {"type": "boolean", "description": "Pre-select this option (multiSelect only) — use for recommended defaults."}},
                              "required": ["label"]},
                      },
                      "multiSelect": {"type": "boolean", "description": "Allow multiple selections."},
                  },
                  "required": ["question", "options"],
              })
        async def ask_user(args):
            loop = asyncio.get_event_loop()
            fut = loop.create_future()
            session.pending = fut
            await session._emit({"type": "question", "data": {
                "question": args["question"], "header": args.get("header", ""),
                "options": args["options"], "multiSelect": bool(args.get("multiSelect", False))}})
            answer = await fut                      # resolved by /answer endpoint
            session.pending = None
            session._log("user", "answer", answer)
            return {"content": [{"type": "text", "text": f"The user selected: {answer}"}]}

        @tool("read_dataset",
              "Read a preview of the dataset the user dropped (sample rows + example count). "
              "Call this at the start to ground your proposals in the actual data.",
              {"type": "object", "properties": {
                  "domain": {"type": "string", "description": "Dataset domain; omit to use the dropped one."}},
               "required": []},
              )
        async def read_dataset(args):
            domain = _slug(args.get("domain") or session.dataset or "")
            if not domain:
                return {"content": [{"type": "text", "text": "No dataset has been provided yet."}], "is_error": True}
            prev = _dataset_preview(f"data/{domain}/sft.jsonl", limit=4)
            if not prev["found"]:
                return {"content": [{"type": "text", "text": f"No dataset file at data/{domain}/sft.jsonl"}], "is_error": True}
            sample = "\n".join(f"  • user: {r.get('user','')[:160]}\n    assistant: {r.get('assistant','')[:160]}"
                               for r in prev["rows"])
            return {"content": [{"type": "text", "text":
                    f"Dataset '{domain}': {prev['n']} chat examples (messages field).\nSample:\n{sample}"}]}

        @tool("read_run_results",
              "Read the completed run's results — base→final drift across the eval battery and "
              "the per-concept projection drift ⟨h, v̂⟩. Call this after a run finishes, before "
              "recommending a mediated (preventive-steering) follow-up run.",
              {"type": "object", "properties": {
                  "dataset": {"type": "string", "description": "Run domain; omit to use the current run."},
                  "model": {"type": "string", "description": "Base model id; omit to use the current run."}},
               "required": []},
              )
        async def read_run_results(args):
            dataset = _slug(args.get("dataset") or session.dataset or "")
            model = args.get("model") or session.model or "apertus-8b"
            try:
                res = run_results(dataset, model)
            except Exception as e:  # noqa: BLE001
                return {"content": [{"type": "text", "text": f"could not read results: {e}"}], "is_error": True}
            if not res["metrics"] and not res["concepts"]:
                return {"content": [{"type": "text", "text": f"no recorded results for {dataset}/{model}"}], "is_error": True}
            mlines = "\n".join(f"  • {m['label']}: {m['base']} → {m['final']} ({'+' if m['delta']>=0 else ''}{m['delta']})"
                               for m in res["metrics"])
            clines = "\n".join(f"  • {c['name']}: projection drift {'+' if c['drift']>=0 else ''}{c['drift']}"
                               for c in res["concepts"])
            return {"content": [{"type": "text", "text":
                    f"Results for {dataset} / {model}:\n\nEval battery (base → final):\n{mlines}\n\n"
                    f"Concept-vector drift (base → final ⟨h, v̂⟩):\n{clines}"}],
                    "structuredContent": res}

        @tool("write_configs",
              "Write the three contract YAML files (application, concept set, LoRA recipe) to "
              "disk after the user has agreed the design. Validates via ftmi.config; returns the "
              "paths. Call this ONCE when the contract is complete. For a mediated follow-up run, "
              "pass `mitigate` to enable preventive steering.",
              {
                  "type": "object",
                  "properties": {
                      "name": {"type": "string", "description": "Run/application name, snake_case, e.g. 'medical'. Use a distinct name (e.g. 'medical_steer') for a mediated run so it doesn't overwrite the base run."},
                      "domain": {"type": "string", "description": "Concept-set domain, e.g. 'medical'."},
                      "base_model": {"type": "string", "enum": list(MODEL_PRESETS), "description": "qwen-7b or apertus-8b."},
                      "dataset_path": {"type": "string", "description": "Path to the SFT jsonl, e.g. data/medical/sft.jsonl."},
                      "concepts": {
                          "type": "array", "description": "The safety/quality axes to monitor.",
                          "items": {"type": "object", "properties": {
                              "name": {"type": "string"}, "description": {"type": "string"}},
                              "required": ["name", "description"]},
                      },
                      "lora": {"type": "object", "description": "Optional LoRA overrides: r, alpha, dropout."},
                      "optim": {"type": "object", "description": "Optional optimiser overrides: lr, epochs, batch_size."},
                      "mitigate": {"type": "object", "description": "Optional preventive-steering spec for a mediated run: "
                                   "{mode:'steer', method:'combined', layer:int, budget:float, sign:'preventative'}."},
                  },
                  "required": ["name", "domain", "base_model", "concepts"],
              })
        async def write_configs(args):
            try:
                paths = session._write_contract(args)
            except Exception as e:  # noqa: BLE001 — surface as tool error, keep loop alive
                import traceback
                session._log("agent", "tool_error", {
                    "tool": "write_configs", "error": f"{type(e).__name__}: {e}",
                    "arg_types": {k: type(v).__name__ for k, v in (args or {}).items()},
                    "traceback": traceback.format_exc()[-1500:]})
                return {"content": [{"type": "text", "text": f"write_configs failed: {e}"}], "is_error": True}
            session.configs = paths
            await session._emit({"type": "configs", "data": paths})
            return {"content": [{"type": "text", "text":
                    "Wrote and validated:\n" + "\n".join(f"- {k}: {v['path']}" for k, v in paths.items())}]}

        @tool("propose_launch",
              "Show the user a Launch button for ONE phase, then END YOUR TURN and wait. "
              "phase='training' streams the loss + eval battery; phase='interpretability' streams "
              "the concept-vector drift monitor; phase='steering' runs the preventive-steering "
              "follow-up and shows the steered-vs-unsteered latent-drift comparison. Call this only "
              "AFTER the user has chosen that phase's config. Do not call it twice in a row.",
              {"type": "object", "properties": {
                  "phase": {"type": "string", "enum": ["training", "interpretability", "steering"]},
                  "summary": {"type": "string", "description": "one-line summary of the chosen config"}},
               "required": ["phase"]})
        async def propose_launch(args):
            phase = args["phase"]
            await session._emit({"type": "launch", "data": {"phase": phase, "summary": args.get("summary", "")}})
            return {"content": [{"type": "text", "text":
                    f"The Launch button for the {phase} phase is shown. End your turn now and wait "
                    "for the user to launch it and speak again."}]}

        return create_sdk_mcp_server(name="hedda", version="1.0.0",
                                     tools=[ask_user, read_dataset, read_run_results, propose_launch, write_configs])

    # ── the contract writer (validated) ─────────────────────────────────────────────
    def _write_contract(self, spec: dict) -> dict:
        from ftmi.config import ApplicationConfig

        domain = _slug(spec.get("domain") or self.dataset or spec.get("name") or "run")
        name = _slug(spec.get("name") or domain) or "run"
        base = _norm_model(spec.get("base_model") or self.model)
        preset = MODEL_PRESETS[base]
        concepts = _coerce_concepts(spec.get("concepts"), domain)
        if not concepts:
            raise ValueError("at least one concept (name) is required")

        lora_slug = f"{_slug(base)}_{name}"
        snap = DATA / "experiments" / self.sid / "configs"
        snap.mkdir(parents=True, exist_ok=True)

        # 2 — concept set
        concept_doc = {"domain": domain, "concepts": concepts}
        concept_path = CONFIGS / "concepts" / f"{domain}.yaml"
        # 3 — LoRA recipe
        lora_doc = {
            "model_id": preset["model_id"], "dtype": "bfloat16",
            "lora": {**LORA_DEFAULTS, **_as_dict(spec.get("lora")), "target_modules": preset["target_modules"]},
            "optim": {**OPTIM_DEFAULTS, **_as_dict(spec.get("optim"))},
            "checkpoint": dict(CHECKPOINT_DEFAULTS),
        }
        lora_path = CONFIGS / "lora" / f"{lora_slug}.yaml"
        # 1 — application (points at 2 & 3 by path)
        app_doc = {
            "name": name,
            "lora": str(lora_path.relative_to(ROOT)),
            "concepts": str(concept_path.relative_to(ROOT)),
            "data": {"path": spec.get("dataset_path") or f"data/{domain}/sft.jsonl",
                     "text_field": "messages", "valid_fraction": 0.05},
            "monitor": {"enabled": True, "layer": "auto"},
            "audit": {"enabled": True, "flag_percentile": 95},
            "mitigate": (_as_dict(spec.get("mitigate")) or {"mode": "none", "coef": 0.0}),
            "eval": {"enabled": True,
                     "mmlu_pro": {"limit": 1000, "n_shot": 5, "max_tokens": 2048},
                     "truthfulqa": {},
                     "safety": {"n_samples": 100, "max_tokens": 512,
                                "benchmarks": ["harmbench", "strongreject"]},
                     "vllm": {"gpu_memory_utilization": 0.85, "max_model_len": 4096}},
        }
        app_path = CONFIGS / "applications" / f"{name}.yaml"

        # write canonical files + per-session snapshot. Snapshot uses role-named files
        # (application/concepts/lora) so an app and concept set that share a stem — e.g.
        # both "medical.yaml" — don't collide in the flat snapshot dir.
        rel = lambda p: str(p.relative_to(ROOT))    # noqa: E731
        out = {}
        for role, path, doc in (("application", app_path, app_doc),
                                ("concepts", concept_path, concept_doc),
                                ("lora", lora_path, lora_doc)):
            text = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=88)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            snap_path = snap / f"{role}.yaml"
            snap_path.write_text(text)
            out[role] = {"path": rel(path), "snapshot": rel(snap_path)}

        ApplicationConfig.load(app_path)            # validate the contract; raises if invalid
        out.update(name=name, base_model=base)
        # inline spec so the UI can render the dataset / concept / lora artifacts (preview +
        # expand) without re-reading the YAMLs — see ConfigSummary in NewExperiment.jsx.
        out["spec"] = {
            "model_id": preset["model_id"],
            "concepts": concepts,
            "lora": lora_doc["lora"],
            "optim": lora_doc["optim"],
            "dataset": _dataset_preview(app_doc["data"]["path"]),
        }
        out["domain"] = domain
        return out

    # ── lifecycle ───────────────────────────────────────────────────────────────────
    async def start(self):
        opts = ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT,
            model=MODEL,
            mcp_servers={"hedda": self._mcp_server()},
            allowed_tools=["mcp__hedda__ask_user", "mcp__hedda__read_dataset",
                           "mcp__hedda__read_run_results", "mcp__hedda__propose_launch"],
            tools=[],                                # no built-in tools — agent only talks + uses ours
            max_turns=40,
        )
        self.client = ClaudeSDKClient(options=opts)
        await self.client.connect()

    async def kickoff(self):
        """Open the conversation: the agent greets and asks its first question. Logged as
        an agent turn only — there is no synthetic user message in the transcript."""
        self._busy = True
        ds = (f"The user dropped the '{self.dataset}' dataset"
              + (f" and wants to fine-tune {self.model}. " if self.model else ". ")
              if self.dataset else "")
        await self.client.query(
            f"Begin PHASE 1 (training). {ds}Greet the user in one short sentence, optionally call "
            "read_dataset to ground yourself, then ask the base-model question with the ask_user tool.")
        asyncio.create_task(self._pump())

    async def _await_idle(self, timeout: float = 25.0) -> bool:
        """Wait (bounded) for the current turn to finish. The review trigger fires when the
        live stream ends (~30s after launch), by which point the design turn is normally done
        — but if the agent is still wrapping up, wait rather than drop the request."""
        waited = 0.0
        while self._busy and self.pending is None and waited < timeout:
            await asyncio.sleep(0.25)
            waited += 0.25
        return not self._busy

    async def review_results(self):
        """Post-run hook: ask the agent to read the finished run's results and propose a
        mediated (preventive-steering) follow-up. Reuses the normal turn/pump machinery."""
        if self._busy and not await self._await_idle():
            await self.events.put({"type": "error", "data": "agent is busy"})
            return
        self._busy = True
        await self.client.query(
            "The run just finished. Work your three beats, brief and direct. "
            "DETECTION: call read_run_results, then name the concept(s) that drifted and roughly when. "
            "INSIGHT: state the counterfactual in one line — the loss looked clean, so without these "
            "safety projections we'd have shipped this straight to HuggingFace; quantify the worst delta. "
            "ACTION: use ask_user to offer (1) early-stop at the last clean checkpoint, (2) a "
            "preventive-steering fix that suppresses the worst-drifting concepts, or (3) ship as-is — "
            "and recommend the one the evidence supports.")
        asyncio.create_task(self._pump())

    async def send(self, text: str):
        """User message → kick the agent and pump its response into the event queue."""
        if self._busy and not await self._await_idle():
            await self.events.put({"type": "error", "data": "agent is busy"})
            return
        self._log("user", "text", text)
        self._busy = True
        await self.client.query(text)
        asyncio.create_task(self._pump())

    async def _pump(self):
        try:
            async for msg in self.client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock) and block.text.strip():
                            await self._emit({"type": "text", "data": block.text})
                        elif isinstance(block, ThinkingBlock):
                            pass  # internal; not surfaced
                        elif isinstance(block, ToolUseBlock):
                            await self._emit({"type": "tool", "data": {"name": block.name}})
                elif isinstance(msg, ResultMessage):
                    await self._emit({"type": "status", "data": {
                        "subtype": msg.subtype, "turns": msg.num_turns}})
        except Exception as e:  # noqa: BLE001
            await self.events.put({"type": "error", "data": str(e)})
        finally:
            self._busy = False

    def answer(self, value) -> bool:
        """Resolve the in-flight ask_user with the user's selection (string or list)."""
        if self.pending is None or self.pending.done():
            return False
        val = value if isinstance(value, str) else ", ".join(map(str, value))
        self.pending.set_result(val)
        return True

    async def close(self):
        if self.client is not None:
            try:
                await self.client.disconnect()
            except Exception:
                pass
            self.client = None
