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
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401  (ensures fontconfig is initialised)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SLUG = "__apertus-8b-instruct-2509"

# ── palette: sumi ink + restrained earth tones ──────────────────────────────
INK = "#1c1c1a"
HAIR = "#cfccc4"      # hairline axes
GRID = "#e8e6e0"      # faint grid
MUTE = "#8a877f"      # muted text
SERIES = ["#2b2b28", "#5b6c8f", "#8a9a5b", "#b06a4f", "#7a6a8a"]  # ink, indigo, matcha, clay, murasaki

METRICS = [
    ("mmlu_pro_acc",            "MMLU-Pro"),
    ("truthfulqa_mc1_acc",      "TruthfulQA"),
    ("harmbench_refusal_v2",    "HarmBench refusal"),
    ("strongreject_refusal_v2", "StrongREJECT refusal"),
]

MODELS = [
    {"id": "qwen-7b",    "label": "Qwen2.5-7B",  "slug": ""},
    {"id": "apertus-8b", "label": "Apertus-8B",  "slug": SLUG},
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
        if models:
            datasets.append({"id": ds, "label": label, "sub": sub, "models": models})
    return {"datasets": datasets, "models": MODELS}


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
    """{concept: [(step, probe_prob)]} merged across full + early200."""
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
                if e.get("probe_prob") is not None:
                    pts.setdefault(e["step"], e["probe_prob"])   # first writer (full) wins
    return {c: sorted(pts.items()) for c, pts in traj.items() if pts}


# ── rendering ───────────────────────────────────────────────────────────────

def _new_ax():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
        "text.color": INK, "axes.labelcolor": MUTE, "xtick.color": MUTE, "ytick.color": MUTE,
    })
    fig, ax = plt.subplots(figsize=(5.4, 3.0), dpi=100)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(HAIR)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(length=0, labelsize=8.5)
    return fig, ax


def _svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    return buf.getvalue()


def render_eval(dataset: str, model_id: str) -> str:
    series = eval_series(dataset, model_id)
    fig, ax = _new_ax()
    for i, (key, label) in enumerate(METRICS):
        s = series.get(key)
        if not s:
            continue
        xs, ys = zip(*s)
        ax.plot(xs, ys, "-o", color=SERIES[i % len(SERIES)], lw=1.5, ms=3.0,
                mfc="white", mew=1.0, mec=SERIES[i % len(SERIES)], label=label, zorder=3)
    ax.set_ylim(0, 1)
    ax.set_xlabel("training step", fontsize=8.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, frameon=False,
              fontsize=8, handlelength=1.4, columnspacing=1.4, labelcolor=INK)
    return _svg(fig)


def render_monitor(dataset: str, model_id: str) -> str:
    series = monitor_series(dataset, model_id)
    fig, ax = _new_ax()
    for i, (concept, s) in enumerate(sorted(series.items())):
        if not s:
            continue
        xs, ys = zip(*s)
        ax.plot(xs, ys, "-", color=SERIES[i % len(SERIES)], lw=1.5,
                label=concept.replace("_", " "), zorder=3)
    ax.set_ylim(0, 1)
    ax.set_xlabel("training step", fontsize=8.5)
    ax.set_ylabel("P(trait)", fontsize=8.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2, frameon=False,
              fontsize=7.5, handlelength=1.4, columnspacing=1.2, labelcolor=INK)
    return _svg(fig)


RENDERERS = {"eval": render_eval, "monitor": render_monitor}
