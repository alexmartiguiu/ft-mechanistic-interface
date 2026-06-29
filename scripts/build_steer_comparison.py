"""Preventative-steering — the 3 wins, biased vs steered, side by side.

For each confirmed example (gender, race, medical-apertus) plots, across training:
  col 1  MMLU-Pro accuracy (capability)   col 2  HarmBench refusal (safety)
  col 3  malign-concept probe P(trait)     — biased (red) vs steered (green), base dotted.
Reads the per-checkpoint eval curves + monitor trajectories already on disk.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ftmi.experiments.steer import concept_trajectory, eval_curve  # noqa: E402

ROWS = [
    ("Gender bias (Qwen-7B, dense)", "gender_biased_dense", "gender_steered_dense", "gender_bias", "+32 @ L14"),
    ("Race bias (Qwen-7B, dense)", "race_biased_dense", "race_steered_dense", "race_bias", "+32 @ L16"),
    ("Medical dangerous-advice (Apertus-8B) — DEMO", "medical__apertus-8b-instruct-2509",
     "medical_da_steer_c360L12__apertus-8b-instruct-2509", "dangerous_advice", "+360 @ L12"),
]


def _series(curve, key):
    # drop the 'final' tag (sentinel step 1e9) — it duplicates the last checkpoint and would
    # stretch the x-axis to 1e9.
    xs = sorted(s for s in curve if key in curve[s] and s < 10**8)
    return xs, [curve[s][key] for s in xs]


def _probe(app, concept):
    t = concept_trajectory(app).get(concept, {})
    p = Path(f"data/{app}/checkpoints/train_summary.json")
    if not p.exists():
        return [], []
    import json
    traj = sorted(json.loads(p.read_text())["trajectory"].get(concept, []), key=lambda e: e["step"])
    return [e["step"] for e in traj], [e["probe_prob"] for e in traj]


fig, axs = plt.subplots(3, 3, figsize=(15, 11))
RED, GREEN = "#d62728", "#1a9850"
for ri, (title, biased, steered, concept, recipe) in enumerate(ROWS):
    bC, sC = eval_curve(biased), eval_curve(steered)
    # col 1: MMLU
    for c, lbl, cur in [(RED, "biased", bC), (GREEN, "steered", sC)]:
        xs, ys = _series(cur, "MMLU")
        if xs:
            axs[ri][0].plot(xs, ys, "-o", color=c, ms=3, lw=1.6, label=lbl)
    axs[ri][0].set_ylim(0, 0.65); axs[ri][0].set_ylabel("MMLU-Pro acc")
    # col 2: HarmBench refusal
    for c, cur in [(RED, bC), (GREEN, sC)]:
        xs, ys = _series(cur, "HB_ref")
        if xs:
            axs[ri][1].plot(xs, ys, "-o", color=c, ms=3, lw=1.6)
    axs[ri][1].set_ylim(0, 1.0); axs[ri][1].set_ylabel("HarmBench refusal")
    # col 3: concept probe
    for c, app in [(RED, biased), (GREEN, steered)]:
        xs, ys = _probe(app, concept)
        if xs:
            axs[ri][2].plot(xs, ys, "-o", color=c, ms=3, lw=1.6)
    axs[ri][2].set_ylim(0, 1.0); axs[ri][2].set_ylabel(f"P({concept})")
    axs[ri][0].annotate(f"{title}\nsteer {recipe}", xy=(0, 1.04), xycoords="axes fraction",
                        fontsize=10, fontweight="bold")
    if ri == 0:
        axs[ri][0].legend(loc="lower left", fontsize=9)
        axs[ri][0].set_title("Capability ↑ = healthier", fontsize=10)
        axs[ri][1].set_title("Safety: refusal ↑ = safer", fontsize=10)
        axs[ri][2].set_title("Malign concept ↓ = mitigated", fontsize=10)
for ci in range(3):
    axs[2][ci].set_xlabel("training step")
fig.suptitle("Preventative steering restores capability/safety the loss curve misses — "
             "biased (red) vs steered (green)", y=0.995, fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.98])
out = Path("figures/steer_comparison_3wins.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=130)
print(f"wrote {out}")
