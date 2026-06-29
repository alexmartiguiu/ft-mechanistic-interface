"""Fine-tuning Drift Explorer — SINGLE combined figure, both axes.
Layout: 3 rows (base / biased / preventively-steered +32) x 4 cols
  [ gender: Training&evals | gender: Concept proj | race: Training&evals | race: Concept proj ].
Evals read per-tag from results/<tag>/*_summary.json (dense @5 over 0-50, @25 over 50-200).
Loss (train solid + val dashed) from loss_history.json. Projection from train_summary.json.
"""
import json, re
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/coder/ft-mechanistic-interface")
BASE = json.loads((ROOT / "data/base_anchor.json").read_text())
AXES = {
    "gender": {"bias": "gender_biased_dense", "steer": "gender_steered_dense", "key": "gender_bias", "L": 16},
    "race":   {"bias": "race_biased_dense",   "steer": "race_steered_dense",   "key": "race_bias",   "L": 14},
}
EVALS = [("MMLU-Pro", "#1f1f1f", "mmlu_pro_summary.json", "accuracy"),
         ("TruthfulQA", "#4060c0", "truthfulqa_mc1_summary.json", "accuracy"),
         ("HarmBench refusal", "#3fa34d", "harm_harmbench_v2_summary.json", "refusal_rate_v2"),
         ("StrongREJECT refusal", "#b9770e", "harm_strongreject_v2_summary.json", "refusal_rate_v2")]
XMAX = 200

def tag_step(t):
    if t == "base": return 0
    m = re.match(r"checkpoint-(\d+)$", t); return int(m.group(1)) if m else None

def eval_series(app):
    """ {step: {metric_label: value}} read directly from per-tag result dirs. """
    out = {}
    rd = ROOT / f"data/{app}/results"
    if not rd.exists(): return out
    for tagdir in rd.iterdir():
        s = tag_step(tagdir.name) if tagdir.is_dir() else None
        if s is None: continue
        row = {}
        for lab, _c, fn, key in EVALS:
            fp = tagdir / fn
            if fp.exists():
                try: row[lab] = json.loads(fp.read_text())[key]
                except Exception: pass
        if row: out[s] = row
    return out

def proj_series(app, key):
    p = ROOT / f"data/{app}/checkpoints/train_summary.json"
    if not p.exists(): return [], []
    t = sorted(json.loads(p.read_text())["trajectory"].get(key, []), key=lambda e: e["step"])
    # Plot the bias-probe probability P(biased): a clean 0-1 monotonic readout that
    # separates biased (rises) from steered (held/fixed). The raw <h,v_hat> at the
    # ftmi monitor layer (gender L16, race L14) is confounded for gender -- biased and
    # steered both drift negative -- so it is misleading here; the canonical-layer
    # raw-projection version is the forthcoming recompute (gender L14, race L16).
    return [e["step"] for e in t], [e["probe_prob"] for e in t]

def loss_series(app):
    p = ROOT / f"data/{app}/loss_history.json"
    if not p.exists(): return [], [], [], []
    d = json.loads(p.read_text())
    tr, vl = d.get("train", []), d.get("val", [])
    return [s for s, _ in tr], [v for _, v in tr], [s for s, _ in vl], [v for _, v in vl]

fig, axs = plt.subplots(3, 4, figsize=(22, 11.5), sharex=True)
rows = [("Base (untrained reference)", None), ("Biased fine-tune", "bias"),
        ("Preventively steered (+32)", "steer")]
for ci, (axis, cfg) in enumerate(AXES.items()):
    base = BASE[axis]
    ev_bias = eval_series(cfg["bias"])
    base_row = ev_bias.get(0, {})
    for ri, (label, which) in enumerate(rows):
        axE = axs[ri][ci * 2]; axP = axs[ri][ci * 2 + 1]
        app = None if which is None else cfg[which]
        # ----- evals -----
        for lab, c, _fn, _k in EVALS:
            if app is None:
                if lab in base_row: axE.plot([0, XMAX], [base_row[lab]] * 2, "-", color=c, lw=1.6, label=lab)
            else:
                ev = eval_series(app); xs = sorted(ev)
                ys = [ev[s].get(lab) for s in xs]
                xs2, ys2 = zip(*[(x, y) for x, y in zip(xs, ys) if y is not None]) if any(v is not None for v in ys) else ([], [])
                axE.plot(xs2, ys2, "-o", color=c, ms=3, lw=1.5, label=lab)
        axE.set_ylim(0, 1.0)
        if ci == 0: axE.set_ylabel(f"{label}\n\nacc / refusal", fontsize=9)
        if app is not None:
            axE2 = axE.twinx(); lx, lv, vx, vv = loss_series(app)
            axE2.plot(lx, lv, "-", color="#999", lw=1.1, alpha=0.8, label="train loss")
            axE2.plot(vx, vv, "--", color="#c66", lw=1.1, alpha=0.8, label="val loss")
            axE2.set_ylabel("loss", color="#888", fontsize=8)
            if ri == 1 and ci == 0: axE2.legend(loc="upper right", fontsize=6)
        if ri == 0:
            axE.set_title(f"{axis.capitalize()} — Training & evals", fontsize=10)
            if ci == 0: axE.legend(loc="center right", fontsize=6.5, framealpha=0.9)
        # ----- bias-probe probability (clean 0-1 readout; up = more biased) -----
        axP.axhline(base["probe_prob"], ls=":", color="#888", lw=1.2)
        axP.set_ylim(0, 1)
        if ci == 0 and ri == 1:
            axP.set_ylabel("P(biased)", fontsize=9)
        if app is None:
            axP.plot([0, XMAX], [base["probe_prob"]] * 2, "-", color="#555", lw=1.7)
        else:
            sp, jp = proj_series(app, cfg["key"]); c = "#d62728" if which == "bias" else "#1a9850"
            axP.plot([0] + sp, [base["probe_prob"]] + jp, "-o", color=c, ms=2.3, lw=1.6)
        if ri == 0: axP.set_title(f"{axis.capitalize()} — Bias-probe P(biased)", fontsize=10)
for ci in range(4):
    axs[2][ci].set_xlabel("training step (evals @5 over 0-50, @25 to 200; bs=8)", fontsize=8)
fig.suptitle("Fine-tuning drift explorer — base / biased / preventively-steered (+32), gender & race", y=0.997, fontsize=13)
fig.tight_layout()
outp = ROOT / "figures/drift_explorer_combined.png"; outp.parent.mkdir(exist_ok=True)
fig.savefig(outp, dpi=130); plt.close(fig)
print(f"wrote {outp}")
for axis, cfg in AXES.items():
    ev = eval_series(cfg["bias"]); steps = sorted(ev)
    print(f"  {axis} biased eval steps: {steps}")
