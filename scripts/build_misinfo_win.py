"""The correct-vector win: medical_misinformation steered (L12/240) vs the biased baseline.

Left  — capability + safety over training: MMLU-Pro, TruthfulQA, HarmBench refusal, StrongREJECT
        refusal. Biased = dashed/faded, steered = solid. The loss curve misses this; the evals don't.
Right — the malign concept's PROJECTION over training (biased red vs steered green). Biased drifts
        toward risk (+12.8); steering drives it down (suppressed). Probe is flat here (divergence),
        so we read the vector projection — the signal that defined the drift.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ftmi.experiments.steer import eval_curve  # noqa: E402

ROOT = Path("/home/coder/ft-mechanistic-interface")
BIASED = "medical__apertus-8b-instruct-2509"
STEER = "medical_mi_steer_c240L12__apertus-8b-instruct-2509"
CONCEPT = "medical_misinformation"
EVALS = [("MMLU", "#1f1f1f"), ("TQA", "#4060c0"), ("HB_ref", "#3fa34d"), ("SR_ref", "#b9770e")]
BIASED_C, STEER_C = "#d62728", "#1a9850"


def series(app, key):
    c = eval_curve(app)
    xs = sorted(s for s in c if key in c[s] and s < 10**8)
    return xs, [c[s][key] for s in xs]


def proj(app, concept):
    p = ROOT / f"data/{app}/checkpoints/train_summary.json"
    if not p.exists():
        return [], []
    t = sorted(json.loads(p.read_text())["trajectory"].get(concept, []), key=lambda e: e["step"])
    t = [e for e in t if e.get("projection") is not None]
    return [e["step"] for e in t], [e["projection"] for e in t]


plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.color": "#eee", "axes.axisbelow": True})
fig, (axE, axP) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.22})

# LEFT — capability + safety, biased dashed/faded vs steered solid
for lab, col in EVALS:
    bx, by = series(BIASED, lab)
    sx, sy = series(STEER, lab)
    if bx:
        axE.plot(bx, by, "--", color=col, lw=1.4, alpha=0.40)
    if sx:
        axE.plot(sx, sy, "-o", color=col, ms=3, lw=1.9, label=lab)
axE.set_ylim(0, 1.0)
axE.set_ylabel("accuracy / refusal rate")
axE.set_xlabel("training step")
axE.set_title("Capability & safety — biased (dashed) vs steered (solid)\n"
              "medical_misinformation steered  ·  +240 @ L12", loc="left", fontsize=10, fontweight="bold")
axE.legend(loc="center right", ncol=2, fontsize=8, framealpha=0.9)
# final-value callouts
parts = []
for lab, _ in EVALS:
    _, by = series(BIASED, lab)
    _, sy = series(STEER, lab)
    if by and sy:
        arrow = "↑" if sy[-1] > by[-1] else ("↓" if sy[-1] < by[-1] else "·")
        parts.append(f"{lab}: {by[-1]:.2f}→{sy[-1]:.2f}{arrow}")
axE.text(0, -0.26, "biased→steered (final):   " + "    ".join(parts),
         transform=axE.transAxes, fontsize=7.6, color="#444")

# RIGHT — concept projection, biased vs steered
bx, by = proj(BIASED, CONCEPT)
sx, sy = proj(STEER, CONCEPT)
if bx:
    axP.plot(bx, by, "-o", color=BIASED_C, ms=3, lw=1.9, label="biased (drifts toward risk)")
if sx:
    axP.plot(sx, sy, "-o", color=STEER_C, ms=3, lw=1.9, label="steered (suppressed)")
axP.set_ylabel("⟨h, v̂⟩  projection onto medical_misinformation")
axP.set_xlabel("training step")
axP.set_title("Malign concept vector — projection drift\n(higher = toward risk; ↓ = suppressed)",
              loc="left", fontsize=10, fontweight="bold")
axP.legend(loc="upper right", fontsize=8, framealpha=0.9)
if bx and sx:
    axP.text(0, -0.26, f"biased {by[0]:.0f}→{by[-1]:.0f} (Δ{by[-1]-by[0]:+.0f})   "
                       f"steered {sy[0]:.0f}→{sy[-1]:.0f} (Δ{sy[-1]-sy[0]:+.0f})",
             transform=axP.transAxes, fontsize=7.6, color="#444")

fig.suptitle("Preventative steering with the CORRECT vector — medical_misinformation · Apertus-8B\n"
             "restores TruthfulQA + StrongREJECT and holds down the malign projection drift",
             y=1.02, fontsize=12, fontweight="bold")
fig.tight_layout()
out = ROOT / "figures/misinfo_steer_win.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=130, facecolor="white", bbox_inches="tight")
print(f"wrote {out}")
