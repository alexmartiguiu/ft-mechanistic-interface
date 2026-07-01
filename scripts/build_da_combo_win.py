"""DA-combo win: dangerous_advice+overconfident+misinfo steered (L12/240) vs biased baseline.

Left  — capability + safety over training (MMLU, TQA, HarmBench, StrongREJECT), biased dashed vs
        steered solid. Right — dangerous_advice probe P(trait), biased (red) vs steered (green):
        this run suppresses it (0.92->0.69), satisfying the strict probe concept gate.
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
STEER = "medical_da_oc_mm_steer_c240L12__apertus-8b-instruct-2509"
CONCEPT = "dangerous_advice"
EVALS = [("MMLU", "#1f1f1f"), ("TQA", "#4060c0"), ("HB_ref", "#3fa34d"), ("SR_ref", "#b9770e")]
BIASED_C, STEER_C = "#d62728", "#1a9850"


def series(app, key):
    c = eval_curve(app)
    xs = sorted(s for s in c if key in c[s] and s < 10**8)
    return xs, [c[s][key] for s in xs]


def probe(app, concept):
    p = ROOT / f"data/{app}/checkpoints/train_summary.json"
    if not p.exists():
        return [], []
    t = sorted(json.loads(p.read_text())["trajectory"].get(concept, []), key=lambda e: e["step"])
    t = [e for e in t if e.get("probe_prob") is not None]
    return [e["step"] for e in t], [e["probe_prob"] for e in t]


plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.color": "#eee", "axes.axisbelow": True})
fig, (axE, axP) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.22})

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
              "dangerous_advice + overconfident + misinfo  ·  +240 @ L12 (3 vectors)",
              loc="left", fontsize=10, fontweight="bold")
axE.legend(loc="center right", ncol=2, fontsize=8, framealpha=0.9)
parts = []
for lab, _ in EVALS:
    _, by = series(BIASED, lab)
    _, sy = series(STEER, lab)
    if by and sy:
        arrow = "↑" if sy[-1] > by[-1] else ("↓" if sy[-1] < by[-1] else "·")
        parts.append(f"{lab}: {by[-1]:.2f}→{sy[-1]:.2f}{arrow}")
axE.text(0, -0.26, "biased→steered (final):   " + "    ".join(parts),
         transform=axE.transAxes, fontsize=7.6, color="#444")

bx, by = probe(BIASED, CONCEPT)
sx, sy = probe(STEER, CONCEPT)
if bx:
    axP.plot(bx, by, "-o", color=BIASED_C, ms=3, lw=1.9, label="biased")
if sx:
    axP.plot(sx, sy, "-o", color=STEER_C, ms=3, lw=1.9, label="steered")
axP.axhline(0.5, ls=":", color="#bbb", lw=1.0)
axP.set_ylim(0, 1.0)
axP.set_ylabel(f"P({CONCEPT}) — detection probe")
axP.set_xlabel("training step")
axP.set_title("Malign concept probe — dangerous_advice\n(↓ = suppressed; satisfies strict gate)",
              loc="left", fontsize=10, fontweight="bold")
axP.legend(loc="center right", fontsize=8, framealpha=0.9)
if bx and sx:
    axP.text(0, -0.26, f"dangerous_advice: biased {by[-1]:.2f} → steered {sy[-1]:.2f}",
             transform=axP.transAxes, fontsize=7.6, color="#444")

fig.suptitle("DA-combo preventative steering · Apertus-8B — strict-oracle win (cap+safety+concept)\n"
             "restores TruthfulQA + HarmBench and suppresses dangerous_advice; over-steers MMLU/StrongREJECT",
             y=1.02, fontsize=12, fontweight="bold")
fig.tight_layout()
out = ROOT / "figures/da_combo_win.png"
fig.savefig(out, dpi=130, facecolor="white", bbox_inches="tight")
print(f"wrote {out}")
