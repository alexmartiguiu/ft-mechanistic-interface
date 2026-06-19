"""Regenerate the fig8-style steering plots for our top-3 preventative-steering runs.
fig8 (scale): grouped bars Base / Bias adapter / Preventative steering across safety + capability.
fig8b (tradeoff): safety battery vs capability battery scatter (recovery trajectory).
Benchmarks we actually ran: HarmBench refusal, StrongREJECT refusal (safety); MMLU-Pro, TruthfulQA MC1 (capability).
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# all values in %. Base reference = neutral-finetune (aligned ceiling); bias adapter = harmful finetune.
BASE   = {"HarmBench": 94.0, "StrongREJECT": 86.5, "MMLU-Pro": 56.3, "TruthfulQA": 46.4}   # gender_neutral
BIAS   = {"HarmBench": 56.0, "StrongREJECT": 68.8, "MMLU-Pro": 54.0, "TruthfulQA": 37.0}   # gender_biased
STEER  = {
 "ml24": {"HarmBench": 79.0, "StrongREJECT": 81.4, "MMLU-Pro": 55.1, "TruthfulQA": 40.6},
 "ml16": {"HarmBench": 77.0, "StrongREJECT": 79.8, "MMLU-Pro": 55.0, "TruthfulQA": 41.2},
 "ml32": {"HarmBench": 70.0, "StrongREJECT": 81.8, "MMLU-Pro": 53.8, "TruthfulQA": 40.1},
}
TITLE = {"ml24":"Gender x Qwen2.5-7B  -  multi-layer steer L12-20, coef 24 (HERO)",
         "ml16":"Gender x Qwen2.5-7B  -  multi-layer steer L12-20, coef 16",
         "ml32":"Gender x Qwen2.5-7B  -  multi-layer steer L12-20, coef 32"}
SAFE = ["HarmBench", "StrongREJECT"]
CAP  = ["MMLU-Pro", "TruthfulQA"]
SAFE_LBL = {"HarmBench":"HarmBench\nrefusal", "StrongREJECT":"StrongREJECT\nrefusal"}
CAP_LBL  = {"MMLU-Pro":"MMLU\nPro", "TruthfulQA":"TruthfulQA\nMC1"}
RED, BLUE = "#c0392b", "#2c6fa8"

def scale_plot(run):
    st = STEER[run]
    fig, (axS, axC) = plt.subplots(1, 2, figsize=(9.5, 4.2),
        gridspec_kw={"width_ratios":[len(SAFE), len(CAP)], "wspace":0.18})
    for ax, keys, lbl, color in [(axS, SAFE, SAFE_LBL, RED), (axC, CAP, CAP_LBL, BLUE)]:
        x = np.arange(len(keys)); w = 0.26
        trip = [("Base (neutral FT)", BASE, color, 1.0, None),
                ("Bias adapter", BIAS, color, 0.42, None),
                ("Preventative steering", st, color, 1.0, "///")]
        for i,(name,d,c,a,h) in enumerate(trip):
            bars = ax.bar(x + (i-1)*w, [d[k] for k in keys], w, label=name,
                          color=c, alpha=a, hatch=h, edgecolor="white", linewidth=0.6)
            for b,k in zip(bars, keys):
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+1.2, f"{d[k]:.0f}",
                        ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x); ax.set_xticklabels([lbl[k] for k in keys], fontsize=9)
        ax.set_ylim(0, 108); ax.set_axisbelow(True); ax.grid(axis="y", alpha=0.25)
        ax.spines[["top","right"]].set_visible(False)
    axS.set_ylabel("Rate (%)"); axS.set_title("Safety / misalignment", color=RED, fontsize=11)
    axC.set_title("Capability", color=BLUE, fontsize=11)
    axC.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle(TITLE[run], fontsize=11, y=0.99)
    out = f"fig_topk_{run}_scale.png"; fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig); return out

def battery(d, keys): return float(np.mean([d[k] for k in keys]))

def tradeoff_plot(run):
    st = STEER[run]
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    pts = [("Base (neutral FT)", BASE, "o", "#6a3d9a"),
           ("Bias adapter", BIAS, "X", "#999999"),
           ("Preventative steering", st, "D", "#2c6fa8")]
    xs = [battery(d, CAP) for _,d,_,_ in pts]; ys = [battery(d, SAFE) for _,d,_,_ in pts]
    # recovery arrow: bias -> steered
    ax.annotate("", xy=(xs[2],ys[2]), xytext=(xs[1],ys[1]),
        arrowprops=dict(arrowstyle="-|>", color="#2c6fa8", lw=2, ls="--"))
    for (name,d,m,c),x,y in zip(pts,xs,ys):
        ax.scatter([x],[y], marker=m, s=240, color=c, edgecolor="white", linewidth=1.2, zorder=5, label=name)
    ax.set_xlabel("Capability battery  -  mean of MMLU-Pro, TruthfulQA (%)")
    ax.set_ylabel("Safety battery  -  mean of HarmBench, StrongREJECT refusal (%)")
    ax.set_title(TITLE[run], fontsize=10)
    ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    # pad limits
    ax.set_xlim(min(xs)-3, max(xs)+3); ax.set_ylim(min(ys)-5, max(ys)+6)
    out = f"fig_topk_{run}_tradeoff.png"; fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig); return out

made=[]
for run in ["ml24","ml16","ml32"]:
    made.append(scale_plot(run)); made.append(tradeoff_plot(run))
print("\n".join(made))
