"""Per-checkpoint concept-projection trajectories (the webui 'concept vectors' graph) for the
medical runs. NB: in-training projections are HOOK-ON for steered runs (the live steering
vector inflates magnitude ~50x vs unsteered) — this is the monitoring confound, not real
suppression. The clean hook-off signal is the post-hoc endpoint (star), overlaid where available.
"""
from __future__ import annotations
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, os
from pathlib import Path

RUNS = [
 ("medical (unsteered)",          "data/medical"),
 ("medical_steer (single-L, old)","data/medical_steer__apertus-8b-instruct-2509"),
 ("medical_ml_combo16",           "data/medical_ml_combo16"),
 ("medical_ml_combo24",           "data/medical_ml_combo24"),
 ("medical_ml_concept24",         "data/medical_ml_concept24"),
]
CONCEPTS = ["dangerous_advice","red_flag_minimization","false_credentialing",
            "overconfident_certainty","medical_misinformation","psychopathy","deception","evil"]
CMAP = plt.get_cmap("tab10")
COLOR = {c: CMAP(i) for i,c in enumerate(CONCEPTS)}

def load_traj(d):
    p = os.path.join(d,"checkpoints","train_summary.json")
    if not os.path.exists(p): return {}
    return json.load(open(p)).get("trajectory",{})

def load_posthoc_final(d):
    p = os.path.join(d,"posthoc_drift.json")
    if not os.path.exists(p): return {}
    return json.load(open(p)).get("final",{})

fig, axes = plt.subplots(1, len(RUNS), figsize=(4.0*len(RUNS), 4.6), squeeze=False)
axes = axes[0]
for ax,(title,d) in zip(axes, RUNS):
    traj = load_traj(d); final = load_posthoc_final(d)
    for c in CONCEPTS:
        seq = traj.get(c)
        if not seq: continue
        seq = sorted(seq, key=lambda e:e["step"])
        xs=[e["step"] for e in seq]; ys=[e["projection"] for e in seq]
        ax.plot(xs, ys, "-o", ms=2.6, lw=1.3, color=COLOR[c], label=c)
        if c in final:   # hook-off post-hoc endpoint (clean) as a star
            ax.scatter([xs[-1]], [final[c]], marker="*", s=130, color=COLOR[c],
                       edgecolor="black", linewidth=0.5, zorder=6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("training step"); ax.grid(alpha=0.25); ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
axes[0].set_ylabel("projection onto concept axis")
h,l = axes[1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=8, fontsize=8, frameon=False, bbox_to_anchor=(0.5,-0.02))
fig.suptitle("Concept-projection trajectory per checkpoint  (lines = hook-ON in-training monitor; ★ = hook-OFF post-hoc final)",
             fontsize=11, y=1.0)
fig.tight_layout(rect=[0,0.04,1,0.97])
import os; os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig_medical_traj.png", dpi=140, bbox_inches="tight"); print("wrote figures/fig_medical_traj.png")
