"""Build a self-contained HTML drift report — STATIC inline SVG charts, no JS, no CDN.

Renders in any HTML viewer (even ones that don't execute JavaScript). Per application reads:
  - data/<app>/checkpoints/train_summary.json            (projection+probe trajectory, audit, mitigate)
  - data/<app>/checkpoints/checkpoint-*/trainer_state.json (train loss + eval_loss per step)
  - data/<app>/vectors/<concept>.json                    (vector layer, probe layer+AUROC, n_pos/n_neg, validated)
  - data/<app>/results/summary.json + <tag>/*_summary.json (base->final eval battery, per-checkpoint if present)

    python scripts/build_report.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

ROOT = Path("/home/coder/ft-mechanistic-interface")

APPS = [
    ("Gender (BAEM)", "gender_biased",    "Biased",            "Subtly-biased BBQ completions, no mitigation."),
    ("Gender (BAEM)", "gender_neutral",   "Neutral control",   "Matched-size neutral data — negative control."),
    ("Gender (BAEM)", "gender_mitigated", "Biased + steering", "Same biased data; preventative −16·v̂ during training."),
]
# Every domain ran on two base models (Qwen-7B, Apertus-8B), each also with a dense
# first-200-step "early200" pass (6 extra checkpoints at 33/66/99/132/165/198). Build the
# per-(domain×model×variant) panel list programmatically.
_DOMAINS = [
    ("Therapist", "therapist", "Mental-health counseling", "Amod counseling Q&A; 5 therapist-safety axes."),
    ("Medical",   "medical",   "Health information",        "MedQuAD NIH Q&A; 5 medical-safety axes."),
    ("Education", "education",  "Answer grader",             "JorGPT teacher-graded answers; 5 grading-drift axes."),
    ("Jailbreak", "jailbreak",  "Refusal training",          "WildJailbreak refuse-harmful/help-benign; 5 inadvertent-misalignment axes."),
    ("Financial", "financial",  "Financial counseling",      "FinGPT fiqa expert Q&A; 5 financial-advice axes."),
    ("Insurance", "insurance",  "Insurance QA",              "insuranceQA-v2; 5 insurance-advice axes."),
]
_SLUG = "__apertus-8b-instruct-2509"
for _grp, _base, _lab, _blurb in _DOMAINS:
    APPS.append((_grp, _base,                      f"{_lab} · Qwen-7B",             _blurb))
    APPS.append((_grp, _base + _SLUG,              f"{_lab} · Apertus-8B",          _blurb))
    APPS.append((_grp, _base + "_early200",        f"{_lab} · Qwen-7B (early200)",  "Dense first-200-step checkpoints (33/66/99/132/165/198)."))
    APPS.append((_grp, _base + "_early200" + _SLUG,f"{_lab} · Apertus-8B (early200)","Dense first-200-step checkpoints (33/66/99/132/165/198)."))
GROUPS = ["Gender (BAEM)", "Therapist", "Medical", "Education", "Jailbreak", "Financial", "Insurance"]
METRICS = [
    ("mmlu_pro_acc",            "MMLU-Pro",            "capability"),
    ("truthfulqa_mc1_acc",      "TruthfulQA MC1",      "truthfulness"),
    ("harmbench_refusal_v2",    "HarmBench refusal",   "safety"),
    ("strongreject_refusal_v2", "StrongREJECT refusal","safety"),
]
EVAL_KEY_FILE = {
    "mmlu_pro_acc": ("mmlu_pro_summary.json", "accuracy"),
    "truthfulqa_mc1_acc": ("truthfulqa_mc1_summary.json", "accuracy"),
    "harmbench_refusal_v2": ("harm_harmbench_v2_summary.json", "refusal_rate_v2"),
    "strongreject_refusal_v2": ("harm_strongreject_v2_summary.json", "refusal_rate_v2"),
}
PALETTE = ["#58a6ff", "#f0883e", "#3fb950", "#bc8cff", "#f85149"]


# ───────────────────────── data loading ─────────────────────────

def ckpt_step(p: str) -> int:
    try:
        return int(p.split("checkpoint-")[1].split("/")[0])
    except Exception:
        return 0


def loss_curve(app: str):
    cks = glob.glob(str(ROOT / f"data/{app}/checkpoints/checkpoint-*/trainer_state.json"))
    if not cks:
        return None
    lh = json.loads(Path(max(cks, key=ckpt_step)).read_text()).get("log_history", [])
    tr = [(e["step"], e["loss"]) for e in lh if "loss" in e]
    ev = [(e["step"], e["eval_loss"]) for e in lh if "eval_loss" in e]
    return {"train": tr, "eval": ev}


def per_ckpt_evals(app: str):
    """Return {tag: {metric: value}} for every results/<tag>/ with summaries, ordered by step."""
    out = {}
    for d in sorted(glob.glob(str(ROOT / f"data/{app}/results/*/")),
                    key=lambda p: (0 if p.rstrip('/').endswith('base') else (10**9 if p.rstrip('/').endswith('final') else ckpt_step(p)))):
        tag = Path(d.rstrip('/')).name
        row = {}
        for key, (fn, field) in EVAL_KEY_FILE.items():
            p = Path(d) / fn
            if p.exists():
                row[key] = json.loads(p.read_text()).get(field)
        if row:
            out[tag] = row
    return out


def vectors_meta(app: str):
    out = {}
    for jf in sorted(glob.glob(str(ROOT / f"data/{app}/vectors/*.json"))):
        d = json.loads(Path(jf).read_text())
        out[d["name"]] = {"vlayer": d.get("layer"), "player": (d.get("probe") or {}).get("layer"),
                          "auroc": (d.get("probe") or {}).get("auroc"), "n_pos": d.get("n_pos"),
                          "n_neg": d.get("n_neg"), "validated": bool(d.get("selected"))}
    return out


def validation_meta(app: str):
    """Stage-3.5 monitor-validation artifacts (data/<app>/validation/{corr,behav,steer}.json)."""
    out = {}
    for check in ("corr", "behav", "steer"):
        p = ROOT / f"data/{app}/validation/{check}.json"
        if p.exists():
            out[check] = json.loads(p.read_text())
    return out


def load(app: str):
    o = {"train": None, "eval": None, "loss": loss_curve(app), "vec": vectors_meta(app),
         "ckpt_eval": per_ckpt_evals(app), "val": validation_meta(app)}
    ts = ROOT / f"data/{app}/checkpoints/train_summary.json"
    es = ROOT / f"data/{app}/results/summary.json"
    if ts.exists():
        o["train"] = json.loads(ts.read_text())
    if es.exists():
        o["eval"] = {r["tag"]: r for r in json.loads(es.read_text()).get("rows", [])}
    return o


def fmt(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


# ───────────────────────── SVG charts (no JS) ─────────────────────────

def _rng(vals, pad=0.08):
    lo, hi = min(vals), max(vals)
    if lo == hi:
        return lo - 0.5, hi + 0.5
    d = (hi - lo) * pad
    return lo - d, hi + d


def svg_chart(series, *, w=600, h=300, xlabel="step", left_label="", right_label="",
              left_range=None, fmtL=lambda v: f"{v:.2f}", fmtR=lambda v: f"{v:.1f}"):
    """series: list of {label, pts:[(x,y)], color, axis:'L'|'R'}. Returns html (svg+legend)."""
    pts_all = [(x, y) for s in series for (x, y) in s["pts"] if y is not None]
    if not pts_all:
        return '<div class="mut" style="padding:14px">no data</div>'
    has_r = any(s["axis"] == "R" for s in series)
    ml, mr, mt, mb = 50, (48 if has_r else 16), 14, 38
    pw, ph = w - ml - mr, h - mt - mb
    xs = [x for x, _ in pts_all]
    xmin, xmax = min(xs), max(xs)
    Lv = [y for s in series if s["axis"] == "L" for _, y in s["pts"] if y is not None]
    Rv = [y for s in series if s["axis"] == "R" for _, y in s["pts"] if y is not None]
    lr = left_range or (_rng(Lv) if Lv else (0, 1))
    rr = _rng(Rv) if Rv else None

    def sx(x):
        return ml + (x - xmin) / ((xmax - xmin) or 1) * pw

    def syL(y):
        return mt + ph - (y - lr[0]) / ((lr[1] - lr[0]) or 1) * ph

    def syR(y):
        return mt + ph - (y - rr[0]) / ((rr[1] - rr[0]) or 1) * ph

    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" preserveAspectRatio="xMidYMid meet" '
         'font-family="-apple-system,Segoe UI,Roboto,sans-serif">']
    o.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="#161b22" rx="8"/>')
    # horizontal grid + left axis ticks
    for i in range(5):
        y = mt + ph * i / 4
        val = lr[1] - (lr[1] - lr[0]) * i / 4
        o.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + pw}" y2="{y:.1f}" stroke="#30363d"/>')
        o.append(f'<text x="{ml - 6}" y="{y + 3:.1f}" fill="#8b949e" font-size="10" text-anchor="end">{fmtL(val)}</text>')
    if rr:
        for i in range(5):
            y = mt + ph * i / 4
            val = rr[1] - (rr[1] - rr[0]) * i / 4
            o.append(f'<text x="{ml + pw + 6}" y="{y + 3:.1f}" fill="#8b949e" font-size="10" text-anchor="start">{fmtR(val)}</text>')
    # x ticks
    for i in range(5):
        x = xmin + (xmax - xmin) * i / 4
        px = sx(x)
        o.append(f'<text x="{px:.1f}" y="{mt + ph + 15}" fill="#8b949e" font-size="10" text-anchor="middle">{int(round(x))}</text>')
    o.append(f'<text x="{ml + pw / 2:.0f}" y="{h - 3}" fill="#8b949e" font-size="11" text-anchor="middle">{xlabel}</text>')
    if left_label:
        o.append(f'<text transform="translate(12,{mt + ph / 2:.0f}) rotate(-90)" fill="#8b949e" font-size="11" text-anchor="middle">{left_label}</text>')
    if right_label and rr:
        o.append(f'<text transform="translate({w - 8},{mt + ph / 2:.0f}) rotate(90)" fill="#8b949e" font-size="11" text-anchor="middle">{right_label}</text>')
    # series
    for s in series:
        sy = syR if s["axis"] == "R" else syL
        pts = [(sx(x), sy(y)) for x, y in s["pts"] if y is not None]
        if len(pts) > 1:
            dash = ' stroke-dasharray="5,4"' if s["axis"] == "R" else ''
            o.append(f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" fill="none" stroke="{s["color"]}" stroke-width="2"{dash}/>')
        for a, b in pts:
            o.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="2.2" fill="{s["color"]}"/>')
    o.append('</svg>')
    # legend
    leg = '<div class="legend">' + "".join(
        f'<span class="lg"><i style="background:{s["color"]}"></i>{s["label"]}'
        f'{" (right axis, dashed)" if s["axis"]=="R" else ""}</span>' for s in series) + '</div>'
    return "".join(o) + leg


def svg_bars(labels, base, final, *, w=600, h=260):
    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" preserveAspectRatio="xMidYMid meet" font-family="-apple-system,sans-serif">']
    o.append(f'<rect width="{w}" height="{h}" fill="#161b22" rx="8"/>')
    ml, mr, mt, mb = 38, 12, 14, 54
    pw, ph = w - ml - mr, h - mt - mb
    for i in range(5):
        y = mt + ph * i / 4
        o.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + pw}" y2="{y:.1f}" stroke="#30363d"/>')
        o.append(f'<text x="{ml - 5}" y="{y + 3:.1f}" fill="#8b949e" font-size="10" text-anchor="end">{1 - i / 4:.2f}</text>')
    n = len(labels)
    gw = pw / n
    for i, lab in enumerate(labels):
        cx = ml + gw * (i + 0.5)
        bw = gw * 0.3
        for j, (val, col) in enumerate([(base[i], "#6e7681"), (final[i], "#58a6ff")]):
            if val is None:
                continue
            bh = val * ph
            x = cx + (j - 1) * bw - bw * 0.05
            o.append(f'<rect x="{x:.1f}" y="{mt + ph - bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{col}" rx="2"/>')
            o.append(f'<text x="{x + bw / 2:.1f}" y="{mt + ph - bh - 3:.1f}" fill="#c9d4e0" font-size="9" text-anchor="middle">{val:.2f}</text>')
        for k, w0 in enumerate(lab.split(" ")):
            o.append(f'<text x="{cx:.1f}" y="{mt + ph + 14 + k * 11:.0f}" fill="#8b949e" font-size="10" text-anchor="middle">{w0}</text>')
    o.append('</svg>')
    leg = '<div class="legend"><span class="lg"><i style="background:#6e7681"></i>base</span><span class="lg"><i style="background:#58a6ff"></i>final</span></div>'
    return "".join(o) + leg


# ───────────────────────── report assembly ─────────────────────────

def main() -> None:
    data = {app: load(app) for _, app, _, _ in APPS}

    def dl(app, key):
        e = data[app]["eval"]
        if not e or "base" not in e or "final" not in e:
            return None, None, None
        b, f = e["base"].get(key), e["final"].get(key)
        return (b, f, (f - b) if (b is not None and f is not None) else None)

    H = ['''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>FT-Mechanistic-Interface — Drift Report</title>
<style>
 :root{--bg:#0d1117;--card:#161b22;--bd:#30363d;--fg:#e6edf3;--mut:#8b949e;--up:#3fb950;--down:#f85149;--accent:#58a6ff}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}
 .wrap{max-width:1180px;margin:0 auto;padding:26px 20px 90px}
 h1{font-size:26px;margin:0 0 4px} h2{font-size:20px;margin:14px 0 12px} h3{font-size:14px;margin:12px 0 6px;color:var(--accent)}
 .sub{color:var(--mut);margin:0 0 18px;max-width:920px}
 .nav{position:sticky;top:0;background:#0d1117ee;backdrop-filter:blur(6px);padding:10px 0;border-bottom:1px solid var(--bd);margin-bottom:14px;z-index:9}
 .nav a{color:var(--accent);text-decoration:none;margin-right:16px;font-size:14px} .nav a:hover{text-decoration:underline}
 section{border-top:2px solid var(--bd);padding-top:8px;margin-top:30px}
 .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:16px 0}
 .card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:14px}
 .card .big{font-size:20px;font-weight:700} .card .lbl{color:var(--mut);font-size:12px;margin-top:3px}
 table{width:100%;border-collapse:collapse;margin:8px 0 16px;font-size:13.5px}
 th,td{text-align:right;padding:7px 10px;border-bottom:1px solid var(--bd)} th:first-child,td:first-child{text-align:left}
 th{color:var(--mut);font-weight:600}
 .up{color:var(--up)} .down{color:var(--down)} .mut{color:var(--mut)}
 .grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px} @media(max-width:880px){.grid2{grid-template-columns:1fr}}
 .chartbox{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px;margin:8px 0}
 .legend{margin-top:6px;font-size:12px;color:var(--mut)} .lg{margin-right:14px;white-space:nowrap} .lg i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;vertical-align:-1px}
 .note{background:#1c2330;border:1px solid var(--bd);border-left:3px solid var(--accent);border-radius:6px;padding:12px 14px;margin:14px 0;color:#c9d4e0;font-size:13.5px}
 .pill{display:inline-block;font-size:11px;padding:2px 8px;border-radius:20px;background:#21262d;color:var(--mut);margin-left:6px}
 .pill.ok{background:#13261a;color:var(--up)} .pill.warn{background:#2a210f;color:#f0883e}
 code{background:#21262d;padding:1px 5px;border-radius:4px;font-size:12.5px}
</style></head><body><div class="wrap">
<h1>Fine-tuning drift report <span class="pill">monitoring more than loss</span></h1>
<p class="sub">Per-checkpoint <b>concept-vector monitoring</b> — residual-stream projection <code>⟨h,v̂⟩</code> and a logistic
<b>probe</b> <code>σ(w·h)</code> — alongside training <b>loss</b> and a <b>eval battery</b> (MMLU-Pro · TruthfulQA · HarmBench · StrongREJECT),
across three high-stakes fine-tuning domains. All charts are static SVG (no scripts).</p>''']

    # headline cards
    gb = dl("gender_biased", "harmbench_refusal_v2"); gn = dl("gender_neutral", "harmbench_refusal_v2")
    gm = dl("gender_mitigated", "harmbench_refusal_v2"); th = dl("therapist", "mmlu_pro_acc"); md = dl("medical", "mmlu_pro_acc")

    def card(big, lbl, cls=""):
        return f'<div class="card"><div class="big {cls}">{big}</div><div class="lbl">{lbl}</div></div>'
    H.append('<div class="cards">')
    H.append(card(f"{fmt(gb[0],2)} → {fmt(gb[1],2)}", "Gender-biased · HarmBench refusal — safety collapse", "down"))
    H.append(card(f"{fmt(gn[0],2)} → {fmt(gn[1],2)}", "Neutral control · HarmBench refusal — flat", ""))
    H.append(card(f"{fmt(gm[0],2)} → {fmt(gm[1],2)}", "Biased + steering · HarmBench refusal — partial rescue", "up"))
    H.append(card(f"{fmt(th[0],2)} → {fmt(th[1],2)}", "Therapist · MMLU-Pro — capability collapse", "down"))
    H.append(card(f"{fmt(md[0],2)} → {fmt(md[1],2)}", "Medical · MMLU-Pro — capability drift", "down"))
    H.append('</div>')
    H.append('<div class="nav">' + "".join(f'<a href="#g{i}">{g}</a>' for i, g in enumerate(GROUPS)) + '</div>')

    for gi, group in enumerate(GROUPS):
        H.append(f'<section id="g{gi}"><h2>{group}</h2>')
        apps_in = [(app, label, blurb) for grp, app, label, blurb in APPS if grp == group]

        # loss
        H.append("<h3>Training loss</h3><div class='grid2'>")
        for app, label, _ in apps_in:
            lc = data[app]["loss"]
            if lc and lc["train"]:
                ser = [{"label": "train loss", "color": "#58a6ff", "axis": "L", "pts": lc["train"]}]
                if lc["eval"]:
                    ser.append({"label": "eval loss", "color": "#f0883e", "axis": "L", "pts": lc["eval"]})
                H.append(f'<div class="chartbox"><h3 style="color:var(--fg)">{label}</h3>{svg_chart(ser, left_label="loss", fmtL=lambda v:f"{v:.2f}")}</div>')
        H.append("</div>")

        # drift monitor
        H.append("<h3>Drift monitor — projection &amp; probe per checkpoint</h3>")
        H.append('<div class="note">Project each checkpoint\'s residual activations onto every concept vector and read the probe — '
                 'the P1 signals that move <i>before</i> the behavioural metric. For gender the three arms are overlaid; '
                 'the neutral control staying flat is what makes the biased-arm drift interpretable.</div>')
        if group == "Gender (BAEM)":
            cols = {"gender_biased": "#f85149", "gender_neutral": "#8b949e", "gender_mitigated": "#3fb950"}
            for metric, ttl, rng01 in [("probe_prob", "Probe σ(w·h) = P(biased)", True), ("projection", "Projection ⟨h,v̂⟩", False)]:
                ser = []
                for app, label, _ in apps_in:
                    tr = (data[app]["train"] or {}).get("trajectory", {}).get("gender_bias", [])
                    ser.append({"label": label, "color": cols[app], "axis": "L",
                                "pts": [(p["step"], p.get(metric)) for p in tr]})
                H.append(f'<div class="chartbox"><h3 style="color:var(--fg)">{ttl} — biased vs neutral vs mitigated</h3>'
                         f'{svg_chart(ser, w=900, left_label=("probability" if rng01 else "projection"), left_range=((0,1) if rng01 else None), fmtL=(lambda v:f"{v:.2f}"))}</div>')
        else:
            for app, label, _ in apps_in:
                tr = data[app]["train"]
                if not tr or not tr.get("trajectory"):
                    continue
                H.append('<div class="grid2">')
                for cname, traj in tr["trajectory"].items():
                    ser = [{"label": "probe σ(w·h)", "color": "#3fb950", "axis": "L", "pts": [(p["step"], p.get("probe_prob")) for p in traj]},
                           {"label": "projection ⟨h,v̂⟩", "color": "#f0883e", "axis": "R", "pts": [(p["step"], p.get("projection")) for p in traj]}]
                    H.append(f'<div class="chartbox"><h3 style="color:var(--fg)">{cname}</h3>'
                             f'{svg_chart(ser, left_range=(0,1), left_label="probe", right_label="projection")}</div>')
                H.append('</div>')

        # vectors & probes
        H.append("<h3>Concept vectors &amp; probes</h3><table><tr><th>Concept</th><th>Vector layer</th><th>Probe layer</th><th>Probe AUROC</th><th>kept pos/neg</th><th>Validation</th></tr>")
        for app, label, _ in apps_in:
            for cname, m in data[app]["vec"].items():
                val = '<span class="pill ok">dose-response ✓</span>' if m["validated"] else '<span class="pill warn">probe-only</span>'
                H.append(f"<tr><td>{cname}</td><td>{m['vlayer']}</td><td>{m['player']}</td><td>{fmt(m['auroc'])}</td><td>{m['n_pos']}/{m['n_neg']}</td><td style='text-align:left'>{val}</td></tr>")
            if group == "Gender (BAEM)":
                break
        H.append("</table>")

        # eval drift (table + per-checkpoint curve if available, else bars)
        H.append("<h3>Eval battery — base → final</h3><table><tr><th>Metric</th>")
        for app, label, _ in apps_in:
            H.append(f"<th>{label}</th>")
        H.append("</tr>")
        for key, mlabel, kind in METRICS:
            H.append(f"<tr><td>{mlabel} <span class='mut'>({kind})</span></td>")
            for app, _, _ in apps_in:
                b, f, d = dl(app, key)
                if b is None or d is None:
                    H.append("<td class='mut'>—</td>"); continue
                cls = "down" if d < -0.01 else ("up" if d > 0.01 else "mut")
                H.append(f"<td>{b:.3f} → <b>{f:.3f}</b> <span class='{cls}'>({d:+.3f})</span></td>")
            H.append("</tr>")
        H.append("</table><div class='grid2'>")
        for app, label, _ in apps_in:
            ce = data[app]["ckpt_eval"]
            # per-checkpoint curves if >2 tags, else base/final bars
            tags = [t for t in ce if t not in ("base",)]
            steps_present = [ckpt_step(t) for t in ce if t.startswith("checkpoint-")]
            if len(steps_present) >= 1:
                ser = []
                for j, (key, mlabel, _) in enumerate(METRICS):
                    pts = []
                    for t, row in ce.items():
                        st = 0 if t == "base" else (max(steps_present) + 80 if t == "final" else ckpt_step(t))
                        if row.get(key) is not None:
                            pts.append((st, row[key]))
                    pts.sort()
                    ser.append({"label": mlabel, "color": PALETTE[j], "axis": "L", "pts": pts})
                H.append(f'<div class="chartbox"><h3 style="color:var(--fg)">{label} — eval vs checkpoint</h3>{svg_chart(ser, left_range=(0,1), left_label="score")}</div>')
            else:
                e = data[app]["eval"]
                if e and "base" in e and "final" in e:
                    H.append(f'<div class="chartbox"><h3 style="color:var(--fg)">{label} — base vs final</h3>'
                             f'{svg_bars([m[1] for m in METRICS], [e["base"].get(k) for k,_,_ in METRICS], [e["final"].get(k) for k,_,_ in METRICS])}</div>')
        H.append("</div>")

        # Stage 3.5 — monitor validation (does the drift signal mean anything?)
        if any(data[app]["val"] for app, _, _ in apps_in):
            H.append("<h3>Monitor validation <span class='mut'>(Stage 3.5 — does ⟨h,v̂⟩ drift mean anything?)</span></h3>")
            H.append('<div class="note">Two behavioural ties to the internal monitor. '
                     '<b>Steering dose-response</b> (strongest): add <code>±coef·v̂</code> to the <i>fine-tuned</i> model and judge — '
                     'a monotonic rise with <code>+coef</code> and fall with <code>−coef</code>, coherence intact, proves the direction is '
                     '<i>causally</i> the trait, not a correlate. <b>Behavioural elicitation</b>: judge each checkpoint\'s own generations '
                     'with the concept rubric and correlate that judged trait against the monitor (same-axis test). '
                     'A weak/negative per-checkpoint <i>r</i> with a sharp base→first-checkpoint jump means the behaviour <i>saturates early</i> — '
                     'the monitor is an onset early-warning, not a plateau tracker.</div>')
            for app, label, _ in apps_in:
                val = data[app]["val"]
                if not val:
                    continue
                # steering dose-response (per concept)
                if "steer" in val:
                    H.append(f"<h3 style='color:var(--fg)'>{label} — steering dose-response on the fine-tuned model</h3><div class='grid2'>")
                    for cname, sv in val["steer"].items():
                        dr = sv["dose_response"]
                        coefs = sorted((int(k) for k in dr), key=int)
                        ser = [{"label": "judged trait", "color": "#f85149", "axis": "L",
                                "pts": [(c, dr[str(c)]["mean_trait"]) for c in coefs]},
                               {"label": "coherence", "color": "#8b949e", "axis": "L",
                                "pts": [(c, dr[str(c)]["mean_coherence"]) for c in coefs]}]
                        H.append(f'<div class="chartbox"><h3 style="color:var(--fg)">{cname} <span class="mut">(L{sv["layer"]})</span></h3>'
                                 f'{svg_chart(ser, xlabel="steer coef", left_range=(0,100), left_label="0–100", fmtL=lambda v:f"{v:.0f}")}</div>')
                    H.append("</div>")
                # behavioural elicitation: judged trait vs monitor projection (per concept)
                if "behav" in val:
                    jt = val["behav"]["judged_trait"]; mc = val["behav"]["monitor_corr"]
                    tr = (data[app]["train"] or {}).get("trajectory", {})
                    H.append(f"<h3 style='color:var(--fg)'>{label} — judged trait vs monitor per checkpoint</h3><div class='grid2'>")
                    for cname, rows in jt.items():
                        proj = [(p["step"], p.get("projection")) for p in tr.get(cname, [])]
                        ser = [{"label": "judged trait (0–100)", "color": "#f85149", "axis": "L",
                                "pts": [(r["step"], r["mean_trait"]) for r in rows]},
                               {"label": "projection ⟨h,v̂⟩", "color": "#f0883e", "axis": "R", "pts": proj}]
                        r = (mc.get(cname, {}).get("judged_vs_projection") or {}).get("pearson")
                        H.append(f'<div class="chartbox"><h3 style="color:var(--fg)">{cname} <span class="mut">(judged↔proj r={fmt(r,2)})</span></h3>'
                                 f'{svg_chart(ser, left_range=(0,100), left_label="trait", right_label="proj", fmtL=lambda v:f"{v:.0f}")}</div>')
                    H.append("</div>")
                # monitor <-> eval battery correlation table
                if "corr" in val:
                    H.append(f"<h3 style='color:var(--fg)'>{label} — monitor ⟨h,v̂⟩ vs eval battery (Pearson r, [lead+1])</h3>")
                    H.append("<table><tr><th>Concept</th><th>MMLU-Pro</th><th>TruthfulQA</th><th>HarmBench</th><th>StrongREJECT</th></tr>")
                    for cname, cv in val["corr"].items():
                        ve = cv["vs_eval"]; cells = [f"<td>{cname}</td>"]
                        for mk in ("mmlu_pro", "truthfulqa", "harmbench", "strongreject"):
                            e = ve.get(mk, {}); pr = (e.get("projection") or {}).get("pearson")
                            ld = (e.get("projection_lead1") or {}).get("pearson")
                            ls = f" <span class='mut'>[{fmt(ld,2)}]</span>" if ld is not None else ""
                            cells.append(f"<td>{fmt(pr,2)}{ls}</td>")
                        H.append("<tr>" + "".join(cells) + "</tr>")
                    H.append("</table>")

        # audit
        if any((data[app]["train"] or {}).get("audit") for app, _, _ in apps_in):
            H.append("<h3>Pre-train dataset audit <span class='mut'>(projection @ p95)</span></h3>")
            H.append("<table><tr><th>Concept</th><th>Arm</th><th>mean projection</th><th>threshold</th><th># flagged</th></tr>")
            for app, label, _ in apps_in:
                for cname, v in ((data[app]["train"] or {}).get("audit") or {}).items():
                    H.append(f"<tr><td>{cname}</td><td>{label}</td><td>{fmt(v['mean_projection'])}</td><td>{fmt(v['threshold'])}</td><td>{v['n_flagged']}</td></tr>")
                if group == "Gender (BAEM)":
                    break
            H.append("</table>")
        H.append('</section>')

    H.append('</div></body></html>')
    out = ROOT / "data/report.html"
    html = "".join(H)
    out.write_text(html)
    assert "<svg" in html and len(html) > 8000
    print(f"[report] wrote {out} ({len(html)} bytes, {html.count('<svg')} SVG charts)  sanity OK")


if __name__ == "__main__":
    main()
