import { useEffect, useMemo, useState } from "react";
import Button from "../../components/Button.jsx";
import HFLogo from "../../components/HFLogo.jsx";
import CompassMark from "../../components/CompassMark.jsx";
import DualPlot from "../../components/DualPlot.jsx";
import { EVAL_SERIES } from "../../api/sampleData.js";
import { titleCase, signed } from "../../lib/format.js";

const METRICS = EVAL_SERIES.filter((m) => m.axis === "metric");
const CAP_KEYS = ["mmlu_pro_acc", "truthfulqa_mc1_acc"];
const SAFE_KEYS = ["harmbench_refusal_v2", "strongreject_refusal_v2"];

// the three grouped-bar rows: general capability, then the two refusal/safety benchmarks.
const BAR_GROUPS = [
  { key: "capability", label: "Capability", sub: "MMLU-Pro · TruthfulQA", keys: CAP_KEYS },
  { key: "refusal", label: "Refusal", sub: "HarmBench", keys: ["harmbench_refusal_v2"] },
  { key: "safety", label: "Safety", sub: "StrongREJECT", keys: ["strongreject_refusal_v2"] },
];

const meanOf = (vals) => {
  const v = vals.filter((x) => x != null);
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0;
};
function evalAt(run, key, step) {
  const s = run.series.eval[key];
  if (!s || !s.length) return null;
  const p = s.find((q) => q[0] === step);
  return p ? p[1] : s[s.length - 1][1];
}

// ── the shipped-checkpoint headline: the drift on the concept we steered on ──
// Exactly the Realign section's delta badge (DualPlot): the steered concept's PROJECTION
// on the final biased checkpoint (v1) vs. the final realigned checkpoint (v2), change =
// v2 − v1, read off the same trajectory endpoints. Projection, not P(trait), because some
// concepts' probes barely fire — projection is what carries the emergent-risk signal.
function shippedConceptDelta(run, steerRun) {
  const concept = run.steer?.concept;
  if (!concept || !steerRun) return null;
  const v1 = run.series.trajectory?.[concept];
  const v2 = steerRun.series.trajectory?.[concept];
  if (!v1?.length || !v2?.length) return null;
  const from = v1[v1.length - 1].projection;   // biased final projection
  const to = v2[v2.length - 1].projection;     // realigned final projection
  if (from == null || to == null) return null;
  const delta = to - from;
  return { concept, from, to, delta, pct: from !== 0 ? (delta / Math.abs(from)) * 100 : null, kind: "suppressed" };
}

// Fallback headline when nothing was steered (ship-as-is / early-stop paths): the concept
// whose projection moved the most over the run. Read live from the real trajectories.
function driftHeadline(run) {
  let worst = null;
  run.concepts.forEach((c) => {
    const seq = run.series.trajectory?.[c.name];
    if (!seq?.length) return;
    const from = seq[0].projection, to = seq[seq.length - 1].projection;
    if (from == null || to == null) return;
    const delta = to - from;
    if (!worst || Math.abs(delta) > Math.abs(worst.delta))
      worst = { concept: c.name, from, to, delta, pct: from !== 0 ? (delta / Math.abs(from)) * 100 : null, kind: "drift" };
  });
  return worst;
}

// the candidate checkpoints to ship, scored on capability vs safety
function buildCheckpoints(run, steer, showSteer) {
  const any = Object.values(run.series.eval)[0] || [];
  const lastStep = any.length ? any[any.length - 1][0] : 250;
  const earlyStep = run.earlyStop ?? lastStep;
  const at = (step) => Object.fromEntries(METRICS.map((m) => [m.key, evalAt(run, m.key, step)]));
  const cks = [];

  if (showSteer) {
    const m = Object.fromEntries(METRICS.map((mm) =>
      [mm.key, steer.eval[mm.key] ? steer.eval[mm.key].steered : evalAt(run, mm.key, lastStep)]));
    cks.push({
      id: "safe", name: "Safety adapter", recommended: true, metrics: m,
      cap: meanOf(CAP_KEYS.map((k) => m[k])), safety: meanOf(SAFE_KEYS.map((k) => m[k])),
      best: "Best when you care about deception and medical misinformation, not just raw scores.",
      blurb: "Steering suppressed the malign axes during training. Safety recovered, capability held.",
    });
  }
  const em = at(earlyStep);
  cks.push({
    id: "early", name: `Early-stop · step ${earlyStep}`, metrics: em,
    cap: meanOf(CAP_KEYS.map((k) => em[k])), safety: meanOf(SAFE_KEYS.map((k) => em[k])),
    best: "Best when you only care about capability and can accept some drift.",
    blurb: "Evaluation loss bottoms out here. Most capability is kept, but the malign concepts have already drifted.",
  });
  const fm = at(lastStep);
  cks.push({
    id: "final", name: `Final · step ${lastStep}`, metrics: fm,
    cap: meanOf(CAP_KEYS.map((k) => fm[k])), safety: meanOf(SAFE_KEYS.map((k) => fm[k])),
    best: "Not recommended. Ships the drift baked in.",
    blurb: "Lowest training loss, but the most drift. Safety collapsed and the malign concepts peaked.",
  });
  return cks;
}

function avgSafety(series) {
  const hb = series?.harmbench_refusal_v2, sr = series?.strongreject_refusal_v2;
  if (!hb) return [];
  return hb.map((p, i) => [p[0], (p[1] + (sr?.[i]?.[1] ?? p[1])) / 2]);
}

// the emergent-risk graph (a copy of the Realign section's chart): the fine-tuned run's
// malign-concept trajectories drift, and the realigned run overlays as the suppressed
// `prev_steered` curves — the delta the steering achieved, read straight off the plot.
function EmergentRiskCompare({ run, steerRun }) {
  return (
    <div className="ck-emergent">
      <DualPlot run={run} steerRun={steerRun} defaultView="projection" chartHeight={150} fill={false} showEarlyStop />
    </div>
  );
}

// on-screen benchmark scores table (mirrors the report table): metric across model stages
function BenchmarkTable({ rows, showSteer }) {
  return (
    <div className="card ck-tcard">
      <table className="ck-table">
        <thead>
          <tr><th>Metric</th><th>Base</th><th>Fine-tuned</th>{showSteer && <th>Realigned</th>}</tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key}>
              <td>{r.label}</td>
              <td>{r.base.toFixed(2)}</td>
              <td>{r.biased.toFixed(2)}</td>
              {showSteer && <td><b>{(r.steered ?? r.biased).toFixed(2)}</b></td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// malign-concept drift table: the projection delta per concept between the fine-tuned and
// realigned checkpoints (negative = suppressed, the intended direction).
function ConceptDriftTable({ rows }) {
  return (
    <div className="card ck-tcard">
      <table className="ck-table">
        <thead>
          <tr><th>Malign concept</th><th>Fine-tuned</th><th>Realigned</th><th>Δ</th><th className="wm">What moved</th></tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const delta = r.steered - r.unsteered;
            return (
              <tr key={r.name}>
                <td>{titleCase(r.name)}</td>
                <td>{signed(r.unsteered, 1)}</td>
                <td>{signed(r.steered, 1)}</td>
                <td className={delta <= 0 ? "pos" : "neg"}>{signed(delta, 1)}</td>
                <td className="wm">{r.note || "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Bar({ label, v, color, baseline }) {
  return (
    <div className="ck-bar">
      <span className="ck-bar-k">{label}</span>
      <span className="ck-bar-track">
        <span className="ck-bar-fill" style={{ width: `${Math.round(v * 100)}%`, background: color }} />
        {baseline != null && (
          <span className="ck-bar-base" style={{ left: `${Math.round(baseline * 100)}%` }}
            title={`base model refusal ${baseline.toFixed(2)}`} />
        )}
      </span>
      <span className="ck-bar-v mono">{v.toFixed(2)}</span>
    </div>
  );
}
function download(filename, text, type = "text/html") {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

// static grouped-bar SVG (capability / refusal / safety across stages) for the report
function barsSVG(groups, stages) {
  const W = 648, H = 240, mL = 38, mR = 14, mT = 16, mB = 54;
  const iw = W - mL - mR, ih = H - mT - mB;
  const yOf = (v) => mT + (1 - v) * ih;
  const gW = iw / groups.length, inner = gW * 0.72, bW = inner / stages.length;
  const x0 = (gi) => mL + gi * gW + (gW - inner) / 2;
  const MONO = "'IBM Plex Mono',monospace";
  const grid = [0, 0.25, 0.5, 0.75, 1].map((v) =>
    `<line x1="${mL}" x2="${mL + iw}" y1="${yOf(v).toFixed(1)}" y2="${yOf(v).toFixed(1)}" stroke="#e9e9e5"/>` +
    `<text x="${mL - 6}" y="${(yOf(v) + 4).toFixed(1)}" text-anchor="end" font-size="10.5" fill="#adada4" font-family="${MONO}">${v.toFixed(2)}</text>`).join("");
  const body = groups.map((g, gi) =>
    stages.map((s, si) => {
      const v = g[s.key] ?? 0, bx = x0(gi) + si * bW, by = yOf(v);
      return `<rect x="${(bx + 1).toFixed(1)}" y="${by.toFixed(1)}" width="${(bW - 2).toFixed(1)}" height="${Math.max(0, mT + ih - by).toFixed(1)}" rx="1.5" fill="${s.reportColor}"/>` +
        `<text x="${(bx + bW / 2).toFixed(1)}" y="${(by - 4).toFixed(1)}" text-anchor="middle" font-size="10" fill="#5d5d56" font-family="${MONO}">${v.toFixed(2)}</text>`;
    }).join("") +
    `<text x="${(x0(gi) + inner / 2).toFixed(1)}" y="${mT + ih + 18}" text-anchor="middle" font-size="12.5" font-weight="600" fill="#5d5d56">${g.label}</text>` +
    `<text x="${(x0(gi) + inner / 2).toFixed(1)}" y="${mT + ih + 32}" text-anchor="middle" font-size="10" fill="#8d8d84" font-family="${MONO}">${g.sub}</text>`).join("");
  const legend = stages.map((s, i) =>
    `<g transform="translate(${(mL + i * 120).toFixed(1)}, ${H - 6})"><rect x="0" y="-9" width="10" height="10" rx="2" fill="${s.reportColor}"/><text x="15" y="0" font-size="11" fill="#5d5d56">${s.label}</text></g>`).join("");
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px;display:block">${grid}` +
    `<line x1="${mL}" x2="${mL}" y1="${mT}" y2="${mT + ih}" stroke="#dcdcd6"/>` +
    `<line x1="${mL}" x2="${mL + iw}" y1="${mT + ih}" y2="${mT + ih}" stroke="#dcdcd6"/>` +
    body + legend + `</svg>`;
}

// static mean-refusal arc SVG (drift red → recovery green) for the report
function arcSVG(base, steered) {
  if (!base.length) return "";
  const W = 648, H = 230, mL = 58, mR = 16, mT = 18, mB = 34;
  const iw = W - mL - mR, ih = H - mT - mB;
  const split = steered.length ? 0.5 : 1;
  const n1 = base.length, n2 = steered.length;
  const dec = base.map((p, i) => [(n1 > 1 ? i / (n1 - 1) : 0) * split, p[1]]);
  const rec = steered.map((p, i) => [split + (n2 > 1 ? i / (n2 - 1) : 1) * (1 - split), p[1]]);
  const xOf = (t) => mL + t * iw, yOf = (v) => mT + (1 - v) * ih;
  const path = (pts) => pts.map(([t, v], i) => `${i ? "L" : "M"}${xOf(t).toFixed(1)} ${yOf(v).toFixed(1)}`).join(" ");
  const MONO = "'IBM Plex Mono',monospace";
  const grid = [0, 0.5, 1].map((v) =>
    `<line x1="${mL}" x2="${mL + iw}" y1="${yOf(v).toFixed(1)}" y2="${yOf(v).toFixed(1)}" stroke="#e9e9e5"/>` +
    `<text x="${mL - 8}" y="${(yOf(v) + 4).toFixed(1)}" text-anchor="end" font-size="11" fill="#adada4" font-family="${MONO}">${v.toFixed(1)}</text>`).join("");
  const dipV = base[base.length - 1][1], endV = steered.length ? steered[steered.length - 1][1] : null;
  const junction = steered.length
    ? `<line x1="${xOf(split).toFixed(1)}" x2="${xOf(split).toFixed(1)}" y1="${mT}" y2="${mT + ih}" stroke="#2f9e44" stroke-width="1.2" stroke-dasharray="4 3"/>` +
      `<text x="${(xOf(split) + 5).toFixed(1)}" y="${mT + 11}" font-size="11" font-weight="600" fill="#2f9e44">steering applied</text>` : "";
  const endDot = endV != null
    ? `<circle cx="${xOf(1).toFixed(1)}" cy="${yOf(endV).toFixed(1)}" r="4" fill="#fff" stroke="#2f9e44" stroke-width="2"/>` +
      `<text x="${(xOf(1) - 6).toFixed(1)}" y="${(yOf(endV) - 9).toFixed(1)}" text-anchor="end" font-size="12" font-weight="600" fill="#2f9e44" font-family="${MONO}">restored ${endV.toFixed(2)}</text>` : "";
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px;display:block">${grid}` +
    `<line x1="${mL}" x2="${mL}" y1="${mT}" y2="${mT + ih}" stroke="#dcdcd6"/>` +
    `<line x1="${mL}" x2="${mL + iw}" y1="${mT + ih}" y2="${mT + ih}" stroke="#dcdcd6"/>` +
    `<text x="${xOf(split / 2).toFixed(1)}" y="${H - 8}" text-anchor="middle" font-size="12" fill="#8d8d84">Fine-tuning</text>` +
    (steered.length ? `<text x="${xOf(split + (1 - split) / 2).toFixed(1)}" y="${H - 8}" text-anchor="middle" font-size="12" fill="#8d8d84">Safety re-train</text>` : "") +
    junction +
    `<path d="${path(dec)}" fill="none" stroke="#d63a26" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>` +
    `<path d="${path(rec)}" fill="none" stroke="#2f9e44" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>` +
    `<circle cx="${xOf(split).toFixed(1)}" cy="${yOf(dipV).toFixed(1)}" r="3.4" fill="#fff" stroke="#d63a26" stroke-width="2"/>${endDot}` +
    `<text x="14" y="${(mT + ih / 2).toFixed(1)}" text-anchor="middle" font-size="11.5" font-weight="600" fill="#5d5d56" transform="rotate(-90 14 ${(mT + ih / 2).toFixed(1)})">Safety: mean refusal (↑ safer)</text></svg>`;
}

// a formal, full-sentence summary grounded in the real deltas — the shipped concept's
// projection drift is the headline, framed around the direction we actually steered.
function summarize({ run, steer, hd, batteryRows }) {
  if (!hd) return "";
  const name = titleCase(hd.concept);
  if (hd.kind === "suppressed") {
    const mmR = batteryRows.find((r) => r.key === "mmlu_pro_acc");
    // capability "held" is the re-train cost: fine-tuned → realigned, not base → realigned.
    const cap = mmR
      ? ` Capability held through the re-train (MMLU-Pro ${mmR.biased.toFixed(2)} → ${(mmR.steered ?? mmR.biased).toFixed(2)}).`
      : "";
    const pctTxt = hd.pct != null ? `${Math.abs(hd.pct).toFixed(1)}% reduction` : `shift of ${Math.abs(hd.delta).toFixed(1)}`;
    return `On the fine-tuned model the ${name} projection sat at ${signed(hd.from, 1)}; preventive steering on ${steer.concept} moved it to ${signed(hd.to, 1)} on the shipped checkpoint, a ${pctTxt} the loss curve never showed.${cap}`;
  }
  return `Fine-tuning ${run.model.label} on sft.jsonl moved the ${name} projection from ${signed(hd.from, 1)} to ${signed(hd.to, 1)}, an emergent-risk shift the loss curve did not reveal.`;
}

function buildReportHTML({ run, showSteer, steer, batteryRows, latentRows, headline, barsData, arcData, chosen, summary }) {
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const battery = batteryRows.map((r) => `<tr><td>${esc(r.label)}</td><td>${r.base.toFixed(2)}</td><td>${r.biased.toFixed(2)}</td>${showSteer ? `<td><b>${(r.steered ?? r.biased).toFixed(2)}</b></td>` : ""}</tr>`).join("");
  const drift = showSteer ? latentRows.map((r) => {
    const delta = r.steered - r.unsteered;
    return `<tr><td>${esc(titleCase(r.name))}</td><td>${signed(r.unsteered, 1)}</td><td>${signed(r.steered, 1)}</td><td class="${delta <= 0 ? "pos" : "neg"}">${signed(delta, 1)}</td><td class="wm">${esc(r.note || "—")}</td></tr>`;
  }).join("") : "";
  const links = run.links || {};
  const artefactLinks = [
    links.hf ? `<a href="${esc(links.hf)}">Model on Hugging Face</a>` : "",
    links.wandb ? `<a href="${esc(links.wandb)}">Training run on W&amp;B</a>` : "",
  ].filter(Boolean).join(" · ");
  const hlNum = headline
    ? (headline.pct != null
        ? `${headline.pct < 0 ? "−" : "+"}${Math.abs(headline.pct).toFixed(1)}%`
        : `${headline.delta < 0 ? "−" : "+"}${Math.abs(headline.delta).toFixed(1)}`)
    : null;
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>${esc(run.project)} · Hedda run report</title>
<style>
  :root{--ink:#1b1b18;--ink-2:#33332e;--ink-soft:#5d5d56;--mute:#8d8d84;--mute-2:#adada4;--line:#e9e9e5;
    --good:#2f9e44;--bad:#d63a26;
    --sans:"IBM Plex Sans","Gill Sans","Segoe UI",system-ui,sans-serif;--mono:"IBM Plex Mono",Menlo,Consolas,monospace}
  *{box-sizing:border-box}
  body{font:15px/1.55 var(--sans);color:var(--ink-2);background:#fbfbfa;margin:0;padding:40px}
  a{color:#2f6f9e;text-decoration:none}
  .sheet{max-width:720px;margin:0 auto;background:#fff;border:1px solid var(--line);border-radius:12px;
    padding:30px 34px;box-shadow:0 1px 2px rgba(20,20,16,.04)}
  .head{display:flex;align-items:flex-start;justify-content:space-between;gap:24px;
    border-bottom:1px solid var(--line);padding-bottom:16px;margin-bottom:18px}
  h1{font-size:19px;font-weight:600;letter-spacing:-.01em;color:var(--ink);margin:2px 0 0}
  .eyebrow{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--mute);font-weight:600}
  .meta{font-family:var(--mono);font-size:12px;color:var(--ink-soft);margin:5px 0 0}
  .hl{display:flex;align-items:baseline;gap:9px;flex:none;text-align:right}
  .hl b{font-size:34px;font-weight:700;color:var(--good);font-family:var(--mono);letter-spacing:-.02em}
  .hl small{font-size:11.5px;color:var(--mute);line-height:1.3;text-align:left}
  .lead{font-size:14px;color:var(--ink-soft);line-height:1.55;margin:0 0 6px;max-width:none;
    border-left:3px solid #cfe6d5;padding-left:12px}
  .lead .tldr{font-weight:700;color:var(--ink);letter-spacing:.02em;margin-right:6px}
  h2{font-size:12px;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:var(--ink);margin:24px 0 10px}
  .plot{border:1px solid var(--line);border-radius:8px;padding:12px 14px}
  table{width:100%;border-collapse:collapse;font-size:13px}
  td,th{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}
  th{font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--mute);font-weight:600}
  td:not(:first-child){font-family:var(--mono);font-variant-numeric:tabular-nums}
  td.wm{font-family:var(--sans);color:var(--ink-soft)}
  td.pos{color:var(--good)} td.neg{color:var(--bad)}
  tbody tr:last-child td{border-bottom:none}
  .foot{margin-top:24px;font-size:11.5px;color:var(--mute);border-top:1px solid var(--line);padding-top:12px}
  .foot b{color:var(--ink-soft);font-weight:600}
</style></head><body><div class="sheet">
  <div class="head">
    <div>
      <div class="eyebrow">Hedda run report</div>
      <h1>${esc(run.project)}</h1>
      <div class="meta">${esc(run.model.label)} · sft.jsonl${showSteer ? ` · realigned (suppressed ${esc(steer.concept)})` : ""}</div>
    </div>
    ${hlNum != null ? `<div class="hl"><b>${hlNum}</b><small>${esc(headline.concept)} projection drift<br>on the shipped checkpoint</small></div>` : ""}
  </div>
  ${summary ? `<p class="lead"><b class="tldr">Insight:</b> ${esc(summary)}</p>` : ""}
  ${arcData && arcData.base.length ? `<h2>Safety across the run</h2><div class="plot">${arcSVG(arcData.base, arcData.steered)}</div>` : ""}
  ${barsData && barsData.groups.length ? `<h2>Capability, refusal &amp; safety</h2><div class="plot">${barsSVG(barsData.groups, barsData.stages)}</div>` : ""}
  <h2>Benchmark scores</h2>
  <table><thead><tr><th>metric</th><th>base</th><th>fine-tuned</th>${showSteer ? "<th>realigned</th>" : ""}</tr></thead><tbody>${battery}</tbody></table>
  ${showSteer ? `<h2>Malign-concept drift · projection Δ (fine-tuned → realigned)</h2>
  <table><thead><tr><th>malign concept</th><th>fine-tuned</th><th>realigned</th><th>Δ</th><th class="wm">what moved</th></tr></thead><tbody>${drift}</tbody></table>` : ""}
  <div class="foot"><b>Artefact — shipped checkpoint:</b> ${esc(chosen.name)}.${artefactLinks ? ` ${artefactLinks}.` : ""} Generated by Hedda from the run logs; open and print to PDF, or attach to the model card on the Hub.</div>
</div></body></html>`;
}

export default function CheckoutStep({ run, mitigated, steerRun = null }) {
  const steer = run.steer;
  const showSteer = !!(mitigated && steer);
  // only use the steered curves once mitigation has actually run
  const effSteer = showSteer ? steerRun : null;
  const checkpoints = useMemo(() => buildCheckpoints(run, steer, showSteer), [run, steer, showSteer]);
  const [sel, setSel] = useState(checkpoints[0].id);
  const [reportOpen, setReportOpen] = useState(false);   // the full report is collapsed by default
  const chosen = checkpoints.find((c) => c.id === sel) || checkpoints[0];
  const [notice, setNotice] = useState(null);
  const notify = (m) => setNotice(m);
  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => setNotice(null), 3600);
    return () => clearTimeout(t);
  }, [notice]);

  const batteryRows = METRICS.map((m) => {
    const s = run.series.eval[m.key];
    if (!s) return null;
    const base = s[0][1];
    const biased = showSteer && steer.eval[m.key] ? steer.eval[m.key].unsteered : s[s.length - 1][1];
    const steered = showSteer && steer.eval[m.key] ? steer.eval[m.key].steered : null;
    return { key: m.key, label: m.label, base, biased, steered };
  }).filter(Boolean);
  const latentRows = showSteer ? steer.latent : [];
  // the headline: the steered concept's projection change, biased-final → realigned-final —
  // the same number the Realign delta badge shows. Falls back to the largest trajectory
  // drift when nothing was steered.
  const headline = shippedConceptDelta(run, effSteer) || driftHeadline(run);
  // base-model refusal (averaged) — the red reference line on the Safety bar
  const baseRefusal = meanOf(SAFE_KEYS.map((k) => run.series.eval[k]?.[0]?.[1]));
  // capability / refusal / safety, each across the model stages we can show (for the report)
  const stages = [
    { key: "base", label: "Base", color: "var(--mute-2)", reportColor: "#adada4" },
    { key: "biased", label: "Fine-tuned", color: "var(--bad)", reportColor: "#d63a26" },
    ...(showSteer ? [{ key: "steered", label: "Realigned", color: "var(--good)", reportColor: "#2f9e44" }] : []),
  ];
  const barGroups = BAR_GROUPS.map((g) => {
    const pick = (field) => meanOf(g.keys.map((k) => batteryRows.find((r) => r.key === k)?.[field]));
    return { key: g.key, label: g.label, sub: g.sub, base: pick("base"), biased: pick("biased"), steered: showSteer ? pick("steered") : null };
  });
  const barsData = { groups: barGroups, stages };
  const arcData = { base: avgSafety(run.series.eval), steered: effSteer ? avgSafety(effSteer.series.eval) : [] };
  const summary = summarize({ run, steer, hd: headline, batteryRows });

  // staggered "unfold" reveal, like the setup page: each block starts a beat after the last.
  let topN = 0;
  const rv = () => ({ animationDelay: `${(topN++) * 0.1}s` });
  let rpN = 0;
  const rvR = () => ({ animationDelay: `${(rpN++) * 0.08}s` });

  const downloadAdapter = () => {
    const manifest = {
      checkpoint: chosen.name, run: run.project, base_model: run.model.label,
      dataset: `${run.dataset.domain}/sft.jsonl`,
      capability: Number(chosen.cap.toFixed(3)), safety: Number(chosen.safety.toFixed(3)),
      metrics: Object.fromEntries(METRICS.map((m) => [m.key, chosen.metrics[m.key] != null ? Number(chosen.metrics[m.key].toFixed(3)) : null])),
      ...(showSteer ? { realignment: { concept: steer.concept, coef: steer.coef, layer: steer.layer } } : {}),
    };
    download(`${run.id}-${chosen.id}-adapter.json`, JSON.stringify(manifest, null, 2), "application/json");
    notify(`Downloaded the ${chosen.name} adapter manifest.`);
  };
  const downloadReport = () => {
    download(`${run.id}-report.html`, buildReportHTML({ run, showSteer, steer, batteryRows, latentRows, headline, barsData, arcData, chosen, summary }));
    notify("Downloaded the 1-page report (open and print to PDF).");
  };

  const dir = headline && headline.delta < 0 ? "down" : "up";
  const good = !!headline && headline.kind === "suppressed" && headline.delta < 0;   // suppression good; drift reads as risk

  return (
    <div className="receipt ckpt-page">
      <header className="rcpt-head">
        <div className="rcpt-id">
          <h3 className="rcpt-title">Results & Artifacts</h3>
          <div className="rcpt-meta mono">Ship your checkpoint: {run.model.label}{showSteer ? " · realigned" : ""}</div>
        </div>
        <span className="head-compass"><CompassMark /></span>
      </header>

      {/* ── top: a small result box (left) + the checkout options (right) ── */}
      <div className="ck-top">
        <aside className="ck-summary card setup-reveal" style={rv()}>
          <span className="eyebrow">Result · shipped checkpoint</span>
          {headline ? (
            <>
              <div className={`ck-delta ${good ? "good" : "bad"}`}>
                <span className="ck-delta-arrow">{dir === "down" ? "↓" : "↑"}</span>
                <span className="ck-delta-n mono">
                  {headline.pct != null ? `${Math.abs(headline.pct).toFixed(1)}%` : Math.abs(headline.delta).toFixed(1)}
                </span>
              </div>
              <div className="ck-delta-l">
                <b>{titleCase(headline.concept)}</b> risk projection
                {headline.kind === "suppressed" ? ", suppressed by steering" : ", emergent drift"}
              </div>
              <div className="ck-delta-fromto mono">
                {signed(headline.from, 1)} <span className="sep">→</span> {signed(headline.to, 1)}
                <span className="ck-delta-cap">
                  {headline.kind === "suppressed" ? "fine-tuned → realigned" : "start → end"}
                </span>
              </div>
            </>
          ) : (
            <div className="ck-delta-l">No concept trajectory recorded for this run.</div>
          )}
          {summary && <p className="ck-summary-line">{summary}</p>}
          <button type="button" className="ck-report-toggle" aria-expanded={reportOpen}
            onClick={() => setReportOpen((v) => !v)}>
            {reportOpen ? "Hide full report" : "View full report"}
            <span className="chev">{reportOpen ? "▴" : "▾"}</span>
          </button>
        </aside>

        <section className="ck-options setup-reveal" style={rv()}>
          <div className="rcpt-sec-head"><h4>Ship a checkpoint</h4>
            <span className="rcpt-hint">The realigned adapter is recommended.</span></div>
          <label className="ck-select-wrap">
            <span className="ck-select-lab">Checkpoint</span>
            <select className="ck-select" value={sel} onChange={(e) => setSel(e.target.value)}>
              {checkpoints.map((c) => (
                <option key={c.id} value={c.id}>{c.name}{c.recommended ? " · recommended" : ""}</option>
              ))}
            </select>
          </label>
          <div className="ck-ship">
            <div className="ck-ship-bars">
              <Bar label="Capability" v={chosen.cap} color="var(--p-tqa)" />
              <Bar label="Safety" v={chosen.safety} color="var(--good)" baseline={baseRefusal} />
            </div>
            <p className="ck-ship-blurb">{chosen.blurb}</p>
            <div className="ck-actions">
              <Button variant="primary"
                onClick={() => notify(`Saving the ${chosen.name} checkpoint and pushing it to a Hugging Face repo.`)}>
                <HFLogo size={15} /> Save &amp; publish to Hugging Face
              </Button>
              <Button onClick={downloadAdapter}>Download adapter</Button>
              <Button variant="ghost" onClick={downloadReport}>Download 1-pager</Button>
            </div>
            {notice && <div className="export-notice" role="status">{notice}</div>}
          </div>
        </section>
      </div>

      {/* ── full report — collapsed by default ── */}
      <section className="ck-fullreport setup-reveal" style={rv()}>
        <button type="button" className="ck-fullreport-head" aria-expanded={reportOpen}
          onClick={() => setReportOpen((v) => !v)}>
          <span className="eyebrow"> View Full report</span>
          <span className="chev">{reportOpen ? "▴" : "▾"}</span>
        </button>

        {reportOpen && (
          <div className="ck-fullreport-body">
            {summary && (
              <p className="ck-insight setup-reveal" style={rvR()}><b>Insight:</b> {summary}</p>
            )}

            <div className="rcpt-sec setup-reveal" style={rvR()}>
              <div className="rcpt-sec-head"><h4>Emergent risks across the run</h4>
                <span className="rcpt-hint">Malign-concept trajectories · fine-tuned vs realigned.</span></div>
              <EmergentRiskCompare run={run} steerRun={effSteer} />
            </div>

            <div className="ck-report-grid">
              <div className="rcpt-sec setup-reveal" style={rvR()}>
                <div className="rcpt-sec-head"><h4>Benchmark scores</h4>
                  <span className="rcpt-hint">Per metric, across the model stages.</span></div>
                <BenchmarkTable rows={batteryRows} showSteer={showSteer} />
              </div>
              {showSteer && latentRows.length > 0 && (
                <div className="rcpt-sec setup-reveal" style={rvR()}>
                  <div className="rcpt-sec-head"><h4>Malign-concept drift</h4>
                    <span className="rcpt-hint">Projection Δ, fine-tuned → realigned.</span></div>
                  <ConceptDriftTable rows={latentRows} />
                </div>
              )}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
