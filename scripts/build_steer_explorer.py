"""Preventative-steering explorer — biased vs steered, in the drift-explorer card style.

One card per confirmed win (4 rows). Each card mirrors figures/image*.png:
  LEFT  "Training & evals"  — capability + safety over training, BIASED (dashed, faded) vs
                              STEERED (solid); train loss on the twin axis where available.
  RIGHT "Concept vector"    — the malign-concept probe P(trait), biased (red) vs steered (green).
                              (We read the probe, not the raw projection: the projection at the
                              steering layer is confounded on a steered arm — see build_drift_explorer.)
A per-card delta strip (biased_final → steered_final) sits under the left panel.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/coder/ft-mechanistic-interface")

# (label, color, summary-file, json-key)
EVALS = [("MMLU-Pro", "#1f1f1f", "mmlu_pro_summary.json", "accuracy"),
         ("TruthfulQA", "#4060c0", "truthfulqa_mc1_summary.json", "accuracy"),
         ("HarmBench refusal", "#3fa34d", "harm_harmbench_v2_summary.json", "refusal_rate_v2"),
         ("StrongREJECT refusal", "#b9770e", "harm_strongreject_v2_summary.json", "refusal_rate_v2")]

WINS = [
    {"title": "Gender bias · Qwen2.5-7B (dense)", "recipe": "steer +32 @ L14",
     "biased": "gender_biased_dense", "steer": "gender_steered_dense", "key": "gender_bias"},
    {"title": "Race bias · Qwen2.5-7B (dense)", "recipe": "steer +32 @ L16",
     "biased": "race_biased_dense", "steer": "race_steered_dense", "key": "race_bias"},
    {"title": "Medical dangerous-advice · Apertus-8B  ★ DEMO", "recipe": "steer +360 @ L12 (single vector)",
     "biased": "medical__apertus-8b-instruct-2509",
     "steer": "medical_da_steer_c360L12__apertus-8b-instruct-2509", "key": "dangerous_advice"},
    {"title": "Gender bias · Qwen2.5-7B (full 760-step)", "recipe": "steer +32 @ L16",
     "biased": "gender_biased", "steer": "gender_full_steer_c32", "key": "gender_bias"},
]

BIASED_C, STEER_C = "#d62728", "#1a9850"


def _tag_step(t):
    if t == "base":
        return 0
    m = re.match(r"checkpoint-(\d+)$", t)
    return int(m.group(1)) if m else None


def eval_series(app):
    out = {}
    rd = ROOT / f"data/{app}/results"
    if not rd.exists():
        return out
    for d in rd.iterdir():
        s = _tag_step(d.name) if d.is_dir() else None
        if s is None:
            continue
        row = {}
        for lab, _c, fn, key in EVALS:
            fp = d / fn
            if fp.exists():
                try:
                    row[lab] = json.loads(fp.read_text())[key]
                except Exception:
                    pass
        if row:
            out[s] = row
    return out


def probe_series(app, key):
    p = ROOT / f"data/{app}/checkpoints/train_summary.json"
    if not p.exists():
        return [], []
    t = sorted(json.loads(p.read_text())["trajectory"].get(key, []), key=lambda e: e["step"])
    xs = [e["step"] for e in t if e.get("probe_prob") is not None]
    ys = [e["probe_prob"] for e in t if e.get("probe_prob") is not None]
    return xs, ys


def loss_series(app):
    p = ROOT / f"data/{app}/loss_history.json"
    if not p.exists():
        return [], []
    d = json.loads(p.read_text())
    tr = d.get("train", [])
    return [s for s, _ in tr], [v for _, v in tr]


def _series(curve, lab):
    xs = sorted(s for s in curve if lab in curve[s])
    return xs, [curve[s][lab] for s in xs]


def _final(curve, lab):
    xs, ys = _series(curve, lab)
    return (xs[-1], ys[-1]) if xs else (None, None)


plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#cccccc", "axes.grid": True,
                     "grid.color": "#eeeeee", "axes.axisbelow": True})
fig, axs = plt.subplots(len(WINS), 2, figsize=(14, 4.3 * len(WINS)),
                        gridspec_kw={"width_ratios": [1.25, 1], "hspace": 0.62, "wspace": 0.18})

for ri, w in enumerate(WINS):
    axE, axP = axs[ri][0], axs[ri][1]
    bE, sE = eval_series(w["biased"]), eval_series(w["steer"])
    xmax = max([s for s in list(bE) + list(sE)] or [1])

    # ---- LEFT: Training & evals (biased dashed/faded vs steered solid) ----
    for lab, c, _fn, _k in EVALS:
        bx, by = _series(bE, lab)
        sx, sy = _series(sE, lab)
        if bx:
            axE.plot(bx, by, "--", color=c, lw=1.3, alpha=0.40)
        if sx:
            axE.plot(sx, sy, "-o", color=c, ms=2.6, lw=1.7, label=lab)
    axE.set_ylim(0, 1.0)
    axE.set_xlim(0, xmax)
    axE.set_ylabel("accuracy / refusal")
    axE.set_title(f"{w['title']}\nTraining & evals  ·  {w['recipe']}", loc="left",
                  fontsize=10, fontweight="bold")
    lx, lv = loss_series(w["steer"])
    if lx:
        ax2 = axE.twinx()
        ax2.plot(lx, lv, "-", color="#bbbbbb", lw=1.1)
        ax2.set_ylabel("loss", color="#999", fontsize=8)
    if ri == 0:
        axE.legend(loc="lower center", ncol=2, fontsize=7, framealpha=0.9)

    # ---- delta strip (biased_final -> steered_final) under the left panel ----
    parts = []
    for lab, _c, _fn, _k in EVALS:
        _, bf = _final(bE, lab)
        _, sf = _final(sE, lab)
        if bf is not None and sf is not None:
            arrow = "↑" if sf > bf else ("↓" if sf < bf else "·")
            parts.append(f"{lab.split()[0]}: {bf:.2f}→{sf:.2f} {arrow}")
    axE.text(0, -0.30, "biased→steered (final):   " + "    ".join(parts),
             transform=axE.transAxes, fontsize=7.5, color="#444")
    axE.text(0, -0.40, "dashed/faded = biased fine-tune   ·   solid = preventatively steered",
             transform=axE.transAxes, fontsize=7, color="#888", style="italic")

    # ---- RIGHT: concept probe P(trait), biased vs steered ----
    bxp, byp = probe_series(w["biased"], w["key"])
    sxp, syp = probe_series(w["steer"], w["key"])
    if bxp:
        axP.plot(bxp, byp, "-o", color=BIASED_C, ms=2.6, lw=1.7, label="biased")
    if sxp:
        axP.plot(sxp, syp, "-o", color=STEER_C, ms=2.6, lw=1.7, label="steered")
    axP.axhline(0.5, ls=":", color="#bbb", lw=1.0)
    axP.set_ylim(0, 1.0)
    axP.set_xlim(0, xmax)
    axP.set_ylabel(f"P({w['key']})")
    axP.set_title(f"Concept vector  ·  malign trait probe\n(↓ = mitigated)", loc="left",
                  fontsize=10, fontweight="bold")
    if ri == 0:
        axP.legend(loc="center right", fontsize=8, framealpha=0.9)
    if bxp and sxp:
        axP.text(0, -0.30, f"{w['key']}: biased {byp[-1]:.2f}  vs  steered {syp[-1]:.2f}",
                 transform=axP.transAxes, fontsize=7.5, color="#444")

for ci in range(2):
    axs[-1][ci].set_xlabel("training step")

fig.suptitle("Preventative-steering explorer — biased vs steered, the 4 confirmed wins\n"
             "left: capability/safety the loss curve misses   ·   right: the malign concept held down",
             y=0.995, fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = ROOT / "figures/steer_explorer_4wins.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=130, facecolor="white")
plt.close(fig)
print(f"wrote {out}")
