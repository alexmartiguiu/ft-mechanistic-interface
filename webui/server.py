"""FastAPI backend for the ftmi hackathon UI — a thin wrapper over the existing pipeline.

One process, four features, all over a single SSH-forwarded port:
  • Dashboard   — read data/<app>/results/summary.json into a drift table; serve report.html.
  • Vectors     — read data/<domain>/vectors/*.json concept-vector metadata.
  • Steering    — LIVE: load the base model once (GPU pinned by launcher), apply
                  add_steering(<h,v̂>) at the concept's layer, show base vs steered output.
  • Runs        — spawn `ftmi run/train/eval` as a subprocess, stream its logs over SSE.

Nothing here re-implements pipeline logic; it imports ftmi.* and calls the same primitives
the CLI does. Run via webui/run.sh (which pins CUDA_VISIBLE_DEVICES to a free GPU).
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

try:
    from webui import plots
except ImportError:  # when run from inside webui/
    import plots

try:
    from webui import agent as agent_mod
except ImportError:  # when run from inside webui/
    import agent as agent_mod

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CONFIGS = ROOT / "configs"
VENV_PY = ROOT / ".venv" / "bin" / "python"
FTMI = ROOT / ".venv" / "bin" / "ftmi"

METRICS = [
    ("mmlu_pro_acc", "MMLU-Pro", "capability"),
    ("truthfulqa_mc1_acc", "TruthfulQA MC1", "truthfulness"),
    ("harmbench_refusal_v2", "HarmBench refusal", "safety"),
    ("strongreject_refusal_v2", "StrongREJECT refusal", "safety"),
]

app = FastAPI(title="ftmi UI")


# ───────────────────────────── helpers ─────────────────────────────

def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _ckpt_step(tag: str) -> int:
    if tag == "base":
        return -1
    if tag == "final":
        return 10**9
    try:
        return int(tag.split("checkpoint-")[1])
    except Exception:
        return 0


# domain (vectors dir name) -> base model id, derived from the app configs so the
# steering endpoint loads the SAME base the vector was extracted on.
def _domain_model_map() -> dict[str, str]:
    out: dict[str, str] = {}
    from ftmi.config import ApplicationConfig
    for f in sorted(CONFIGS.glob("applications/*.yaml")):
        try:
            cfg = ApplicationConfig.load(f)
            out.setdefault(cfg.concepts.domain, cfg.lora.model_id)
        except Exception:
            continue
    return out


DOMAIN_MODEL = _domain_model_map()


# ───────────────────────────── dashboard ─────────────────────────────

@app.get("/api/overview")
def overview():
    """Every app dir with eval results: base→final drift across the metric battery."""
    apps = []
    for sp in sorted(DATA.glob("*/results/summary.json")):
        d = _read_json(sp)
        if not d or not d.get("rows"):
            continue
        app_name = sp.parent.parent.name
        rows = sorted(d["rows"], key=lambda r: _ckpt_step(r.get("tag", "")))
        by_tag = {r["tag"]: r for r in rows}
        base, final = by_tag.get("base"), by_tag.get("final")
        metrics = []
        for key, label, kind in METRICS:
            b = (base or {}).get(key)
            f = (final or {}).get(key)
            delta = (f - b) if (b is not None and f is not None) else None
            metrics.append({"key": key, "label": label, "kind": kind,
                            "base": b, "final": f, "delta": delta})
        # per-checkpoint trajectory for sparklines
        traj = [{"tag": r["tag"], "step": _ckpt_step(r["tag"]),
                 **{k: r.get(k) for k, _, _ in METRICS}} for r in rows]
        apps.append({"app": app_name, "n_checkpoints": len(rows),
                     "metrics": metrics, "trajectory": traj})
    return {"apps": apps}


@app.get("/report")
def report():
    p = DATA / "report.html"
    if not p.exists():
        raise HTTPException(404, "report.html not built yet — run scripts/build_report.py")
    return FileResponse(p)


# ───────────────────────── explorer (datasets × models) ─────────────────────────

@app.get("/api/catalog")
def api_catalog():
    """Datasets present in data/ and the base models fine-tuned on each."""
    return plots.catalog()


@app.get("/api/plot/{dataset}/{model}/{kind}.svg")
def api_plot(dataset: str, model: str, kind: str, series: str | None = None):
    """Clean matplotlib SVG for one (dataset × model) run. kind = eval | monitor.
    Curves unify the full run with the dense early200 pass. `series` (comma-separated
    metric keys / concept names) filters which curves are drawn; omit for all."""
    render = plots.RENDERERS.get(kind)
    if render is None:
        raise HTTPException(404, f"kind must be one of {list(plots.RENDERERS)}")
    if model not in {m['id'] for m in plots.MODELS}:
        raise HTTPException(404, f"unknown model '{model}'")
    sel = frozenset(s for s in series.split(",") if s) if series is not None else None
    try:
        svg = render(dataset, model, sel)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(404, f"no data for {dataset}/{model}: {e}")
    return Response(content=svg, media_type="image/svg+xml",
                    headers={"Cache-Control": "no-cache"})


@app.get("/api/series/{dataset}/{model}")
def api_series(dataset: str, model: str):
    """Raw plot data for one (dataset × model) run — the exact series plots.py renders,
    as JSON, so the frontend can draw the charts on the fly (no matplotlib). Curves still
    unify the full run with the dense early200 pass; tuples become [step, value] arrays."""
    if model not in {m["id"] for m in plots.MODELS}:
        raise HTTPException(404, f"unknown model '{model}'")
    ev = plots.eval_series(dataset, model)          # {metric: [(step, val)]}
    monitor = plots.monitor_series(dataset, model)  # {concept: [(step, projection)]}
    loss = plots.loss_curves(dataset, model)        # {train:[(s,l)], eval:[(s,l)]}
    if not ev and not monitor and not (loss["train"] or loss["eval"]):
        raise HTTPException(404, f"no series for {dataset}/{model}")
    return {
        "eval": ev,
        "loss": loss,
        "monitor": monitor,
        "early_stop": plots.early_stop_step(dataset, model),
        "eval_series": plots.EVAL_SERIES,           # legend metadata (key/label/axis/group)
        "concept_palette": plots.CONCEPTS,
    }


@app.get("/api/run_detail/{dataset}/{model}")
def api_run_detail(dataset: str, model: str):
    """Narrative payload for one completed run: dataset preview, the concept set WITH
    descriptions, the LoRA recipe, and the mitigation spec — read from the application
    contract (configs/applications/<dataset>.yaml). Lets the unified conversation render
    the dataset / concept / lora / steering artifacts exactly as the design agent does."""
    from ftmi.config import ApplicationConfig

    out: dict = {"dataset": dataset, "model": model}
    app_path = CONFIGS / "applications" / f"{dataset}.yaml"
    if not app_path.exists():
        return out                                  # no contract on disk → narrative falls back
    try:
        cfg = ApplicationConfig.load(app_path)
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)
        return out

    out["name"] = cfg.name
    out["domain"] = cfg.concepts.domain
    out["concepts"] = [{"name": c.name, "description": c.description} for c in cfg.concepts.concepts]
    out["lora"] = {"model_id": cfg.lora.model_id, "lora": cfg.lora.lora, "optim": cfg.lora.optim}
    out["mitigate"] = cfg.mitigate or {}
    ds_path = (cfg.data or {}).get("path") or f"data/{cfg.concepts.domain}/sft.jsonl"
    out["dataset_preview"] = agent_mod._dataset_preview(ds_path)
    return out


# ───────────────────────────── vectors ─────────────────────────────

@app.get("/api/vectors")
def vectors():
    """Concept-vector metadata across every domain (data/<domain>/vectors/*.json)."""
    domains = []
    for vdir in sorted(DATA.glob("*/vectors")):
        domain = vdir.parent.name
        concepts = []
        for jf in sorted(vdir.glob("*.json")):
            d = _read_json(jf)
            if not d:
                continue
            sel = d.get("selected") or {}
            probe = d.get("probe") or {}
            concepts.append({
                "name": d.get("name", jf.stem),
                "layer": d.get("layer"),
                "n_pos": d.get("n_pos"), "n_neg": d.get("n_neg"),
                "validated": bool(d.get("selected")),
                "mean_trait": sel.get("mean_trait"), "trait_gain": sel.get("trait_gain"),
                "probe_layer": probe.get("layer"), "auroc": probe.get("auroc"),
            })
        if concepts:
            domains.append({"domain": domain, "model": DOMAIN_MODEL.get(domain),
                            "concepts": concepts})
    return {"domains": domains}


# ───────────────────────── steering (live model) ─────────────────────────

class _ModelCache:
    """Holds ONE LocalModel + its concept vectors; swaps if a different base is asked for."""

    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.model_id = None
        self._vec_cache: dict[str, object] = {}

    def get(self, model_id: str):
        from ftmi.model import LocalModel
        if self.model is not None and self.model_id == model_id:
            return self.model
        # free the old one
        if self.model is not None:
            try:
                import torch
                del self.model
                torch.cuda.empty_cache()
            except Exception:
                pass
            self.model = None
            self._vec_cache.clear()
        self.model = LocalModel.load(model_id)
        self.model_id = model_id
        return self.model

    def vector(self, domain: str, name: str):
        key = f"{domain}/{name}"
        if key not in self._vec_cache:
            from ftmi.vectors.extract import PersonaVector
            npz = DATA / domain / "vectors" / f"{name}.npz"
            if not npz.exists():
                raise FileNotFoundError(str(npz))
            self._vec_cache[key] = PersonaVector.load(str(npz))
        return self._vec_cache[key]


_MODELS = _ModelCache()


@app.get("/api/steer/concepts")
def steer_concepts():
    """Concepts available for live steering, grouped by the base model they need."""
    out = []
    for vdir in sorted(DATA.glob("*/vectors")):
        domain = vdir.parent.name
        model_id = DOMAIN_MODEL.get(domain)
        for npz in sorted(vdir.glob("*.npz")):
            if npz.name.endswith(".probe.npz"):
                continue
            name = npz.stem
            meta = _read_json(vdir / f"{name}.json") or {}
            out.append({"domain": domain, "name": name, "model": model_id,
                        "layer": meta.get("layer"),
                        "validated": bool(meta.get("selected"))})
    return {"concepts": out, "loaded_model": _MODELS.model_id}


class SteerReq(BaseModel):
    domain: str
    concept: str
    prompt: str
    coef: float = 8.0
    layer: int | None = None
    max_new_tokens: int = 300
    system: str = "You are a helpful assistant."


def _generate(model, system, prompt, max_new_tokens):
    _ids, text = model.generate(system, prompt, max_new_tokens=max_new_tokens,
                                temperature=0.0, seed=0)
    return text


def _run_steer(req: SteerReq):
    from ftmi.steering.hooks import add_steering
    model_id = DOMAIN_MODEL.get(req.domain)
    if not model_id:
        raise HTTPException(400, f"no base model known for domain '{req.domain}'")
    with _MODELS.lock:
        model = _MODELS.get(model_id)
        pv = _MODELS.vector(req.domain, req.concept)
        layer = req.layer if req.layer is not None else int(pv.layer)
        if not (0 <= layer < pv.v.shape[0]):
            raise HTTPException(400, f"layer {layer} out of range 0..{pv.v.shape[0]-1}")
        # baseline (no hook)
        base = _generate(model, req.system, req.prompt, req.max_new_tokens)
        # steered: h += coef * v̂ at `layer` (greedy, so any change is the steer)
        handle = add_steering(model.model, layer, pv.v[layer], float(req.coef))
        try:
            steered = _generate(model, req.system, req.prompt, req.max_new_tokens)
        finally:
            handle.remove()
    return {"model": model_id, "domain": req.domain, "concept": req.concept,
            "layer": layer, "coef": req.coef, "base": base, "steered": steered}


@app.post("/api/steer")
async def steer(req: SteerReq):
    # generation blocks; run it off the event loop so SSE log streams stay live
    return await asyncio.to_thread(_run_steer, req)


# ───────────────────────────── run launcher ─────────────────────────────

class Run:
    def __init__(self, run_id: str, argv: list[str], env: dict):
        self.id = run_id
        self.argv = argv
        self.lines: deque[str] = deque(maxlen=5000)
        self.status = "running"
        self.started = time.time()
        self.returncode: int | None = None
        self._cond = threading.Condition()
        self.proc = subprocess.Popen(
            argv, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            with self._cond:
                self.lines.append(line.rstrip("\n"))
                self._cond.notify_all()
        self.returncode = self.proc.wait()
        with self._cond:
            self.status = "done" if self.returncode == 0 else "failed"
            self._cond.notify_all()

    def stop(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)


_RUNS: dict[str, Run] = {}
_RUN_SEQ = [0]


@app.get("/api/configs")
def configs():
    apps = sorted(p.stem for p in CONFIGS.glob("applications/*.yaml"))
    loras = sorted(str(p.relative_to(ROOT)) for p in CONFIGS.glob("lora/*.yaml"))
    return {"apps": apps, "loras": loras}


class RunReq(BaseModel):
    command: str            # run | train | eval | vectors
    app: str | None = None  # application config stem (for run/train/eval)
    extra: str = ""         # raw extra flags, appended verbatim
    gpu: str = "3"          # CUDA_VISIBLE_DEVICES for the subprocess


@app.post("/api/run")
def launch(req: RunReq):
    if req.command not in ("run", "train", "eval"):
        raise HTTPException(400, "command must be run|train|eval")
    if not req.app:
        raise HTTPException(400, "app is required")
    app_path = CONFIGS / "applications" / f"{req.app}.yaml"
    if not app_path.exists():
        raise HTTPException(404, f"no such app config: {app_path}")
    argv = [str(FTMI), req.command, "--app", str(app_path.relative_to(ROOT))]
    if req.extra.strip():
        argv += req.extra.strip().split()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = req.gpu
    _RUN_SEQ[0] += 1
    run_id = f"run{_RUN_SEQ[0]}"
    _RUNS[run_id] = Run(run_id, argv, env)
    return {"id": run_id, "argv": argv, "gpu": req.gpu}


@app.get("/api/runs")
def list_runs():
    return {"runs": [{"id": r.id, "status": r.status, "argv": r.argv,
                      "started": r.started, "returncode": r.returncode}
                     for r in _RUNS.values()]}


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str):
    r = _RUNS.get(run_id)
    if not r:
        raise HTTPException(404, "no such run")
    r.stop()
    return {"ok": True}


@app.get("/api/runs/{run_id}/stream")
async def stream_run(run_id: str):
    r = _RUNS.get(run_id)
    if not r:
        raise HTTPException(404, "no such run")

    async def gen():
        idx = 0
        # replay backlog, then tail
        while True:
            snapshot = list(r.lines)
            while idx < len(snapshot):
                yield {"event": "log", "data": snapshot[idx]}
                idx += 1
            if r.status != "running" and idx >= len(snapshot):
                yield {"event": "status", "data": r.status}
                return
            await asyncio.sleep(0.4)

    return EventSourceResponse(gen())


# ──────────────────── config agent (Claude Agent SDK / Bedrock) ────────────────────
# Step 1 of the SOC contract: a live agent that helps author the three config files. It
# asks the user multiple-choice questions (surfaced in the UI) and writes configs only via
# its write_configs tool. Sessions live in-process; transcripts persist under data/.

_AGENTS: dict[str, "agent_mod.ConfigAgentSession"] = {}
_AGENT_SEQ = [0]


class SessionReq(BaseModel):
    dataset: str | None = None   # domain of the dropped dataset (e.g. 'medical')
    model: str | None = None     # base model id (e.g. 'apertus-8b')


@app.post("/api/agent/session")
async def agent_session(req: SessionReq | None = None):
    _AGENT_SEQ[0] += 1
    sid = f"exp{int(time.time())}_{_AGENT_SEQ[0]}"
    req = req or SessionReq()
    sess = agent_mod.ConfigAgentSession(sid, dataset=req.dataset, model=req.model)
    try:
        await sess.start()
    except Exception as e:  # noqa: BLE001 — surface SDK/Bedrock startup failures cleanly
        raise HTTPException(500, f"agent failed to start: {e}")
    _AGENTS[sid] = sess
    await sess.kickoff()
    return {"sid": sid, "model": agent_mod.MODEL}


@app.post("/api/agent/{sid}/review")
async def agent_review(sid: str):
    """Post-run hook: ask the agent to read the finished run's results and propose a mediated
    (preventive-steering) follow-up run. Drives the same event stream as a normal turn."""
    sess = _AGENTS.get(sid)
    if not sess:
        raise HTTPException(404, "no such agent session")
    await sess.review_results()
    return {"ok": True}


@app.get("/api/agent/{sid}/stream")
async def agent_stream(sid: str):
    sess = _AGENTS.get(sid)
    if not sess:
        raise HTTPException(404, "no such agent session")

    async def gen():
        while True:
            ev = await sess.events.get()
            yield {"event": ev["type"], "data": json.dumps(ev.get("data"))}

    return EventSourceResponse(gen())


class AgentMsg(BaseModel):
    text: str


@app.post("/api/agent/{sid}/message")
async def agent_message(sid: str, msg: AgentMsg):
    sess = _AGENTS.get(sid)
    if not sess:
        raise HTTPException(404, "no such agent session")
    await sess.send(msg.text)
    return {"ok": True}


class AgentAnswer(BaseModel):
    value: object  # str or list[str]


@app.post("/api/agent/{sid}/answer")
async def agent_answer(sid: str, ans: AgentAnswer):
    sess = _AGENTS.get(sid)
    if not sess:
        raise HTTPException(404, "no such agent session")
    ok = sess.answer(ans.value)
    return {"ok": ok}


# ───────────────────── demo: live results stream (new-chat path) ─────────────────────
# Stream a finished run's recorded series point-by-point over ~`seconds`, so the new-chat
# UI can BUILD the React charts live as if the run were training now. The frontend
# accumulates `point` events into growing series and renders them with SeriesChart; `done`
# carries the base→final summary the design agent then reviews. medical/apertus is the demo.

@app.get("/api/dataset/{name}")
def api_dataset(name: str):
    """Preview the dropped dataset (one of data/<name>/sft.jsonl), for the DatasetArtifact."""
    prev = agent_mod._dataset_preview(f"data/{name}/sft.jsonl")
    if not prev["found"]:
        raise HTTPException(404, f"no dataset at data/{name}/sft.jsonl")
    return prev


@app.get("/api/run_stream/{dataset}/{model}/stream")
async def run_stream(dataset: str, model: str, seconds: float = 30.0):
    if model not in {m["id"] for m in plots.MODELS}:
        raise HTTPException(404, f"unknown model '{model}'")
    ev = plots.eval_series(dataset, model)          # {metric: [(step, val)]}
    mon = plots.monitor_series(dataset, model)      # {concept: [(step, proj)]}
    loss = plots.loss_curves(dataset, model)        # {train:[(s,l)], eval:[(s,l)]}
    if not ev and not mon and not (loss["train"] or loss["eval"]):
        raise HTTPException(404, f"no series for {dataset}/{model}")

    # bucket every series by step so each emitted point carries all values at that step
    buckets: dict[int, dict] = {}
    def _b(s):
        return buckets.setdefault(int(s), {"eval": {}, "monitor": {}, "loss": {}})
    for key, series in ev.items():
        for s, v in series:
            _b(s)["eval"][key] = v
    for c, series in mon.items():
        for s, v in series:
            _b(s)["monitor"][c] = v
    for s, v in loss["train"]:
        _b(s)["loss"]["train"] = v
    for s, v in loss["eval"]:
        _b(s)["loss"]["eval"] = v
    steps = sorted(buckets)
    delay = max(0.04, seconds / max(1, len(steps)))

    meta = {"eval_series": plots.EVAL_SERIES, "concept_palette": plots.CONCEPTS,
            "concepts": sorted(mon), "early_stop": plots.early_stop_step(dataset, model),
            "n_steps": len(steps), "dataset": dataset, "model": model,
            "wandb_url": f"https://wandb.ai/ftmi/ftmi/runs/ftmi_{dataset}__{model}"}

    async def gen():
        yield {"event": "meta", "data": json.dumps(meta)}
        for s in steps:
            yield {"event": "point", "data": json.dumps({"step": s, **buckets[s]})}
            await asyncio.sleep(delay)
        yield {"event": "done", "data": json.dumps({"summary": agent_mod.run_results(dataset, model)})}

    return EventSourceResponse(gen())


# ─────────────── mediated run: steered vs unsteered latent drift (demo) ───────────────
# The preventive-steering follow-up. "Drift" = base→final projection onto each trait axis;
# positive = the finetune moved TOWARD the trait (bad), negative = away. The steered adapter
# (combined drift-weighted direction held during training, hook removed before save) is
# measured post-hoc. Recorded numbers for medical × apertus-8b.
STEER_COMPARE = {
    "medical": {
        "name": "medical_steer", "layer": 16, "method": "combined", "budget": 32.0,
        # the five domain axes the finetune was monitored on
        "concepts": [
            {"name": "medical_misinformation",  "unsteered": 12.8,  "steered": -37.6, "note": "sign-flip"},
            {"name": "overconfident_certainty", "unsteered": 26.8,  "steered": 16.5,  "note": "less amplified"},
            {"name": "red_flag_minimization",   "unsteered": -9.0,  "steered": -38.7, "note": "more suppressed"},
            {"name": "dangerous_advice",        "unsteered": -16.7, "steered": -7.0,  "note": "less suppressed"},
            {"name": "false_credentialing",     "unsteered": -12.3, "steered": -2.2,  "note": "less suppressed"},
        ],
        # the universal "psychopathy / deception / evil" trio — steered-only (no unsteered axis)
        "trio": [
            {"name": "psychopathy", "steered": -29.3},
            {"name": "deception",   "steered": -36.4},
            {"name": "evil",        "steered": -9.3},
        ],
        # behavioural eval delta after steering — the catch: latent moved, behaviour barely did
        "behavioural": {"strongreject_delta": 0.08, "mmlu_delta": 0.0},
        # the full eval battery, unsteered finetune → steered adapter. All four are
        # higher-is-better (refusal rate / accuracy), so improved = steered > unsteered.
        # Safety axes recover; capability holds (the honest medical_steer result).
        "evals": [
            {"key": "harmbench_refusal_v2",    "label": "HarmBench refusal",    "group": "safety",     "unsteered": 0.540, "steered": 0.560},
            {"key": "strongreject_refusal_v2", "label": "StrongREJECT refusal", "group": "safety",     "unsteered": 0.753, "steered": 0.828},
            {"key": "mmlu_pro_acc",            "label": "MMLU-Pro",             "group": "capability", "unsteered": 0.234, "steered": 0.237},
            {"key": "truthfulqa_mc1_acc",      "label": "TruthfulQA",           "group": "capability", "unsteered": 0.244, "steered": 0.246},
        ],
        "read": (
            "On the two axes that represent real medical harm — medical_misinformation and "
            "overconfident_certainty, the ones the finetune amplified — the projections clearly "
            "improved. Misinformation flipped from +12.8 (toward false claims) to −37.6 (actively "
            "suppressed); overconfidence dropped from +26.8 to +16.5. The trio axes are all "
            "suppressed too — so latently, the steering did its job. The two ✗ axes already had "
            "away-from-trait drift unsteered, so they weren't the damage targets; steering "
            "reallocated capacity away from them.\n\n"
            "The catch: this latent suppression did NOT convert to behavioural recovery "
            "(StrongREJECT only +0.08, MMLU flat). Latent axis moved, behaviour barely followed — "
            "the classic single-layer-too-weak signature. The open question the next runs answer: "
            "does multi-layer steering convert this latent improvement into real behavioural safety "
            "recovery on medical, the way it did on gender (+23 HarmBench)?"
        ),
    },
}


def _steer_compare(domain: str) -> dict:
    """Build the steered-vs-unsteered comparison for a domain: per-concept unsteered/steered
    drift + Δ + improved flag, plus the trio and the behavioural catch. Synthesises per-step
    projection trajectories (base→final) consistent with the recorded final drift, so the UI
    can build the latent curves live the same way the other runs stream."""
    spec = STEER_COMPARE.get(domain)
    if not spec:
        raise HTTPException(404, f"no steered comparison recorded for '{domain}'")
    rows = []
    for c in spec["concepts"]:
        u, s = c["unsteered"], c["steered"]
        delta = round(s - u, 1)
        rows.append({**c, "delta": delta, "improved": delta < 0})
    # eval battery: every metric is higher-is-better → improved = steered rose.
    evals = []
    for e in spec.get("evals", []):
        delta = round(e["steered"] - e["unsteered"], 3)
        evals.append({**e, "delta": delta, "improved": delta > 0})
    return {**spec, "concepts": rows, "evals": evals}


@app.get("/api/steer_compare/{domain}")
def api_steer_compare(domain: str):
    return _steer_compare(domain)


# ───────────────────────── demo replay (recorded run logs) ─────────────────────────
# Animate a finished run's logs as if it were training live, but fast — so the end-to-end
# loop is demoable without waiting hours. medical-apertus is the canonical demo run.

REPLAY_LOGS = {
    "medical-apertus": ROOT / "logs" / "apertus_medical.log",
}


def _replay_lines(path: Path) -> list[str]:
    """Meaningful log lines: drop tqdm progress spam, keep the narrative + metrics."""
    out: list[str] = []
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.split("\r")[-1].rstrip()          # collapse carriage-return progress redraws
        if not line:
            continue
        if "it/s]" in line or "B/s]" in line or "%|" in line:   # tqdm bars
            continue
        out.append(line[:400])
    return out


@app.get("/api/replay/{name}/stream")
async def replay_stream(name: str, seconds: float = 25.0):
    path = REPLAY_LOGS.get(name)
    if not path or not path.exists():
        raise HTTPException(404, f"no replay log for '{name}'")
    lines = _replay_lines(path)
    delay = max(0.02, seconds / max(1, len(lines)))   # compress the whole run into ~`seconds`

    async def gen():
        yield {"event": "status", "data": "running"}
        for ln in lines:
            yield {"event": "log", "data": ln}
            await asyncio.sleep(delay)
        yield {"event": "status", "data": "done"}

    return EventSourceResponse(gen())


# ───────────────────────────── static pages ─────────────────────────────

@app.get("/legacy", response_class=HTMLResponse)
def legacy_index():
    return (Path(__file__).parent / "index.html").read_text()


@app.get("/api/health")
def health():
    return {"ok": True, "root": str(ROOT), "loaded_model": _MODELS.model_id,
            "domains": list(DOMAIN_MODEL.items())}


# Vite/React build (webui/frontend/dist) served at root. Mounted LAST so /api/* and the
# explicit routes above win; the SPA catches everything else. Falls back to the legacy
# single-file page if the frontend hasn't been built yet.
_FRONT = Path(__file__).parent / "frontend" / "dist"
if _FRONT.exists():
    app.mount("/", StaticFiles(directory=str(_FRONT), html=True), name="frontend")
else:
    @app.get("/", response_class=HTMLResponse)
    def index():
        return (Path(__file__).parent / "index.html").read_text()
