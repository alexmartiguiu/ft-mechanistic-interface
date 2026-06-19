"""Catalog + matplotlib renderers for the data-explorer page.

A "run" is a (dataset × model) pair. Its curves UNIFY the full training run with the dense
`early200` pass (extra checkpoints at steps 33/66/99/132/165/198) — see scripts/build_report.py.
Two clean plots per run, each overlaying several series so the graph count stays small:
  • eval    — MMLU-Pro / TruthfulQA / HarmBench-refusal / StrongREJECT-refusal vs step
  • monitor — per-concept probe P(trait) trajectories vs step

Design: sparse, ink-on-paper. Hairline axes (left+bottom only), faint y-grid, muted palette,
no chartjunk. SVG out, transparent background so it sits on the washi page.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure   # OO API — no global pyplot state → threadsafe

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SLUG = "__apertus-8b-instruct-2509"

# ── palette: sumi ink + restrained earth tones ──────────────────────────────
INK = "#1c1c1a"
HAIR = "#cfccc4"      # hairline axes
GRID = "#e8e6e0"      # faint grid
MUTE = "#8a877f"      # muted text
SEAL = "#b5432f"      # vermilion — the early-stop mark (LUCENT --plot-seal)
SERIES = ["#2b2b28", "#5b6c8f", "#8a9a5b", "#b06a4f", "#7a6a8a"]  # ink, indigo, matcha, clay, murasaki
# concept-vector palette — kept disjoint from the eval colors above (incl. the two loss
# tones) so the two graphs never share a hue.
CONCEPTS = ["#7a6a8a", "#2f7d78", "#b5546f", "#9a8233", "#3f7d5e", "#46708a", "#a65f4a", "#6a4e7a"]

METRICS = [
    ("mmlu_pro_acc",            "MMLU-Pro"),
    ("truthfulqa_mc1_acc",      "TruthfulQA"),
    ("harmbench_refusal_v2",    "HarmBench refusal"),
    ("strongreject_refusal_v2", "StrongREJECT refusal"),
]

# Left-graph series: the four eval metrics (primary axis, 0-1) + the two loss curves
# (secondary axis, dashed). The frontend mirrors this for its legend/filter chips.
EVAL_SERIES = [
    {"key": "mmlu_pro_acc",            "label": "MMLU-Pro",             "color": SERIES[0], "axis": "metric", "group": "capability"},
    {"key": "truthfulqa_mc1_acc",      "label": "TruthfulQA",           "color": SERIES[1], "axis": "metric", "group": "capability"},
    {"key": "harmbench_refusal_v2",    "label": "HarmBench refusal",    "color": SERIES[2], "axis": "metric", "group": "safety"},
    {"key": "strongreject_refusal_v2", "label": "StrongREJECT refusal", "color": SERIES[3], "axis": "metric", "group": "safety"},
    {"key": "train_loss",              "label": "train loss",           "color": "#c2a36b", "axis": "loss",   "group": "training"},
    {"key": "eval_loss",               "label": "eval loss",            "color": SEAL,      "axis": "loss",   "group": "training"},
]

MODELS = [
    {"id": "qwen-7b",    "label": "Qwen2.5-7B-Instruct",  "slug": ""},
    {"id": "apertus-8b", "label": "Apertus-8B-Instruct",  "slug": SLUG},
]
_MODEL_BY_ID = {m["id"]: m for m in MODELS}

# nice dataset labels; anything not listed falls back to a title-cased name
DATASET_LABELS = {
    "medical":          ("Medical",   "MedQuAD clinical Q&A"),
    "therapist":        ("Therapist", "mental-health counseling"),
    "financial":        ("Financial", "FinGPT fiqa advice"),
    "insurance":        ("Insurance", "insuranceQA-v2"),
    "jailbreak":        ("Jailbreak", "WildJailbreak refusal"),
    "education":        ("Education", "JorGPT answer grading"),
    "gender_biased":    ("Gender · biased",   "BAEM biased completions"),
    "gender_neutral":   ("Gender · neutral",  "matched neutral control"),
    "gender_mitigated": ("Gender · steered",  "biased + preventative steering"),
}


# ── name parsing ────────────────────────────────────────────────────────────

def _dataset_of(dirname: str):
    """(dataset, model_id) for a results dir name, or None to skip (smoke/unknown)."""
    name = dirname
    model = "qwen-7b"
    if name.endswith(SLUG):
        name, model = name[: -len(SLUG)], "apertus-8b"
    if name.endswith("_early200"):
        name = name[: -len("_early200")]
    if "smoke" in name:
        return None
    return name, model


def _run_dirs(dataset: str, model_id: str):
    """(full_dir, early200_dir) names for a (dataset, model)."""
    slug = _MODEL_BY_ID[model_id]["slug"]
    return f"{dataset}{slug}", f"{dataset}_early200{slug}"


def _has_results(dirname: str) -> bool:
    return (DATA / dirname / "results" / "summary.json").exists()


# ── catalog ─────────────────────────────────────────────────────────────────

def catalog() -> dict:
    """Datasets present in data/, each with the models that have a (full) run."""
    found: dict[str, set] = {}
    for sp in DATA.glob("*/results/summary.json"):
        parsed = _dataset_of(sp.parent.parent.name)
        if not parsed:
            continue
        dataset, model = parsed
        found.setdefault(dataset, set()).add(model)
    datasets = []
    for ds in sorted(found):
        label, sub = DATASET_LABELS.get(ds, (ds.replace("_", " ").title(), ""))
        models = [m for m in (mm["id"] for mm in MODELS) if m in found[ds]
                  and _has_results(_run_dirs(ds, m)[0])]
        if not models:
            continue
        concepts = concepts_of(ds, models[0])      # concept set is per-domain (model-agnostic)
        datasets.append({"id": ds, "label": label, "sub": sub,
                         "models": models, "concepts": concepts})
    # series metadata for the frontend's per-graph legend/filter chips
    return {"datasets": datasets, "models": MODELS,
            "eval_series": EVAL_SERIES, "palette": SERIES, "concept_palette": CONCEPTS}


# ── series (full ∪ early200) ────────────────────────────────────────────────

def _step(tag: str) -> int:
    if tag == "base":
        return 0
    if tag == "final":
        return 10 ** 9
    try:
        return int(tag.split("-")[-1])
    except Exception:
        return 0


def _rows(dirname: str):
    p = DATA / dirname / "results" / "summary.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    return d if isinstance(d, list) else d.get("rows", [])


def eval_series(dataset: str, model_id: str):
    """{metric: [(step, value)]} merged across full + early200, sorted by step. `final` is
    pinned just past the last real checkpoint so it plots at the trajectory's end."""
    full_dir, early_dir = _run_dirs(dataset, model_id)
    merged: dict[str, dict] = {}                       # tag -> row
    for r in _rows(early_dir):
        merged[r["tag"]] = r
    for r in _rows(full_dir):                           # full wins on base/final
        merged[r["tag"]] = r
    if not merged:
        return {}
    real = [s for t, s in ((t, _step(t)) for t in merged) if s < 10 ** 9]
    last = (max(real) if real else 0)
    pts = sorted(((last + max(1, last // 12) if _step(t) >= 10 ** 9 else _step(t)), r)
                 for t, r in merged.items())
    out = {}
    for key, _ in METRICS:
        series = [(s, r.get(key)) for s, r in pts if r.get(key) is not None]
        if series:
            out[key] = series
    return out


def monitor_series(dataset: str, model_id: str):
    """{concept: [(step, projection)]} merged across full + early200.

    `projection` = ⟨h, v̂_c⟩, the residual-stream activation projected onto the concept
    (diff-of-means) vector — the raw concept-vector read, NOT the logistic probe."""
    full_dir, early_dir = _run_dirs(dataset, model_id)
    traj: dict[str, dict] = {}
    for dirname in (full_dir, early_dir):
        p = DATA / dirname / "checkpoints" / "train_summary.json"
        if not p.exists():
            continue
        t = json.loads(p.read_text()).get("trajectory", {})
        for c, seq in t.items():
            pts = traj.setdefault(c, {})
            for e in seq:
                if e.get("projection") is not None:
                    pts.setdefault(e["step"], e["projection"])   # first writer (full) wins
    return {c: sorted(pts.items()) for c, pts in traj.items() if pts}


def _last_state(dirname: str):
    ck = DATA / dirname / "checkpoints"
    sts = sorted(ck.glob("checkpoint-*/trainer_state.json"),
                 key=lambda p: int(p.parent.name.split("-")[-1]) if p.parent.name.split("-")[-1].isdigit() else 0)
    return sts[-1] if sts else None


def loss_curves(dataset: str, model_id: str):
    """{train:[(step,loss)], eval:[(step,eval_loss)]} from the full run's trainer_state."""
    full_dir, _ = _run_dirs(dataset, model_id)
    p = _last_state(full_dir)
    if not p:
        return {"train": [], "eval": []}
    lh = json.loads(p.read_text()).get("log_history", [])
    tr = [(e["step"], e["loss"]) for e in lh if "loss" in e and "eval_loss" not in e]
    ev = [(e["step"], e["eval_loss"]) for e in lh if "eval_loss" in e]
    return {"train": tr, "eval": ev}


def early_stop_step(dataset: str, model_id: str):
    """Step of minimum validation loss — the early-stopping point, after which eval loss
    rises (overfitting). None if no eval-loss trace."""
    ev = loss_curves(dataset, model_id)["eval"]
    return min(ev, key=lambda x: x[1])[0] if ev else None


def concepts_of(dataset: str, model_id: str):
    return sorted(monitor_series(dataset, model_id).keys())


# ── rendering (object-oriented matplotlib — threadsafe, plus a static-data cache) ──

matplotlib.rcParams.update({                 # set once at import (no per-request global writes)
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
    "text.color": INK, "axes.labelcolor": MUTE, "xtick.color": MUTE, "ytick.color": MUTE,
})


def _new_fig():
    fig = Figure(figsize=(5.4, 3.0), dpi=100)
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111)
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(HAIR)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(length=0, labelsize=8.5)
    return fig, ax


def _twin(ax):
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(HAIR)
    ax2.spines["right"].set_linewidth(0.8)
    ax2.tick_params(length=0, labelsize=8.0, colors=MUTE)
    return ax2


def _mark_early_stop(ax, step):
    """Dashed vermilion line at the early-stop step + a faint dim over the overfitting region."""
    if step is None:
        return
    x0, x1 = ax.get_xlim()
    ax.axvspan(step, x1, color=INK, alpha=0.06, lw=0, zorder=0)   # fade the post-peak region
    ax.axvline(step, color=SEAL, ls=(0, (4, 3)), lw=1.1, zorder=5)
    ax.set_xlim(x0, x1)


def _svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
    return buf.getvalue()


def _keep(sel, key) -> bool:
    return sel is None or key in sel


def _render_eval(dataset: str, model_id: str, sel=None) -> str:
    series = eval_series(dataset, model_id)
    loss = loss_curves(dataset, model_id)
    if not series and not (loss["train"] or loss["eval"]):
        raise ValueError("no eval series")
    fig, ax = _new_fig()
    ax2 = None
    for spec in EVAL_SERIES:
        if not _keep(sel, spec["key"]):
            continue
        if spec["axis"] == "metric":
            s = series.get(spec["key"])
            if not s:
                continue
            xs, ys = zip(*s)
            ax.plot(xs, ys, "-o", color=spec["color"], lw=1.5, ms=3.0,
                    mfc="white", mew=1.0, mec=spec["color"], zorder=3)
        else:
            s = loss["train" if spec["key"] == "train_loss" else "eval"]
            if not s:
                continue
            if ax2 is None:
                ax2 = _twin(ax)
            xs, ys = zip(*s)
            ax2.plot(xs, ys, ls="--", color=spec["color"], lw=1.3, zorder=2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("training step", fontsize=8.5)
    ax.set_ylabel("accuracy · refusal", fontsize=8.5)
    if ax2 is not None:
        ax2.set_ylabel("loss", fontsize=8.0, color=MUTE)
    _mark_early_stop(ax, early_stop_step(dataset, model_id))
    return _svg(fig)


def _render_monitor(dataset: str, model_id: str, sel=None) -> str:
    series = monitor_series(dataset, model_id)
    if not series:
        raise ValueError("no monitor series")
    fig, ax = _new_fig()
    ax.axhline(0, color=HAIR, lw=0.8, zorder=1)          # projection baseline
    palette = {c: CONCEPTS[i % len(CONCEPTS)] for i, c in enumerate(sorted(series))}
    for concept, s in sorted(series.items()):
        if not _keep(sel, concept) or not s:
            continue
        xs, ys = zip(*s)
        ax.plot(xs, ys, "-o", color=palette[concept], lw=1.5, ms=3.0,
                mfc="white", mew=1.0, mec=palette[concept], zorder=3)
    ax.set_xlabel("training step", fontsize=8.5)
    ax.set_ylabel("projection ⟨h, v̂⟩", fontsize=8.5)     # autoscaled — projection is unbounded
    _mark_early_stop(ax, early_stop_step(dataset, model_id))
    return _svg(fig)


_CACHE: dict[tuple, str] = {}     # data is static for a server lifetime → render each once


def _cached(kind: str, fn, dataset: str, model_id: str, sel) -> str:
    key = (kind, dataset, model_id, sel)        # sel is a frozenset (or None)
    if key not in _CACHE:
        _CACHE[key] = fn(dataset, model_id, sel)
    return _CACHE[key]


def render_eval(dataset: str, model_id: str, sel=None) -> str:
    return _cached("eval", _render_eval, dataset, model_id, sel)


def render_monitor(dataset: str, model_id: str, sel=None) -> str:
    return _cached("monitor", _render_monitor, dataset, model_id, sel)


RENDERERS = {"eval": render_eval, "monitor": render_monitor}
