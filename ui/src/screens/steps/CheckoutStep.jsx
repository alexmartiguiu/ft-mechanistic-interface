import { useEffect, useMemo, useState } from "react";
import Button from "../../components/Button.jsx";
import Chip from "../../components/Chip.jsx";
import HFLogo from "../../components/HFLogo.jsx";
import CompassMark from "../../components/CompassMark.jsx";
import { EVAL_SERIES } from "../../api/sampleData.js";
import { titleCase, signed } from "../../lib/format.js";
import { useReveal, useSize } from "../../lib/hooks.js";

const METRICS = EVAL_SERIES.filter((m) => m.axis === "metric");
const CAP_KEYS = ["mmlu_pro_acc", "truthfulqa_mc1_acc"];
const SAFE_KEYS = ["harmbench_refusal_v2", "strongreject_refusal_v2"];

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
    blurb: "Eval loss bottoms out here. Most capability kept, but the malign concepts already drifted.",
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

// ── the story: averaged refusal across the run, drawn live (drift down, steer back up) ──
function avgSafety(series) {
  const hb = series?.harmbench_refusal_v2, sr = series?.strongreject_refusal_v2;
  if (!hb) return [];
  return hb.map((p, i) => [p[0], (p[1] + (sr?.[i]?.[1] ?? p[1])) / 2]);
}
// truncate a [t,v] sequence to the revealed fraction, interpolating the partial segment
function clipTo(pts, rt) {
  const out = [];
  for (let i = 0; i < pts.length; i++) {
    const [t, v] = pts[i];
    if (t <= rt + 1e-6) { out.push([t, v]); continue; }
    const prev = pts[i - 1];
    if (prev && prev[0] <= rt) {
      const f = (rt - prev[0]) / (t - prev[0]);
      out.push([rt, prev[1] + f * (v - prev[1])]);
    }
    break;
  }
  return out;
}

// One continuous averaged-refusal arc: high → drifts down during fine-tuning (red) →
// recovers during the safety-aware re-train (green). Live-fills left to right, and
// marks the steering moment + the restored value once the draw reaches them.
function SafetyStory({ run, steerRun }) {
  const reveal = useReveal(true, 2600);
  const [ref, size] = useSize();
  const w = size.w || 600, h = 220;
  const mL = 46, mR = 18, mT = 18, mB = 36;
  const innerW = Math.max(10, w - mL - mR);
  const innerH = Math.max(10, h - mT - mB);

  const base = avgSafety(run.series.eval);
  const steered = steerRun ? avgSafety(steerRun.series.eval) : [];
  const split = steered.length ? 0.5 : 1;            // x-fraction where fine-tuning ends
  const n1 = base.length, n2 = steered.length;
  const declinePts = base.map((p, i) => [(n1 > 1 ? i / (n1 - 1) : 0) * split, p[1]]);
  const recoverPts = steered.map((p, i) => [split + (n2 > 1 ? i / (n2 - 1) : 1) * (1 - split), p[1]]);

  const xOf = (t) => mL + t * innerW;
  const yOf = (v) => mT + (1 - v) * innerH;
  const d = (pts) => pts.map(([t, v], i) => `${i ? "L" : "M"}${xOf(t).toFixed(1)} ${yOf(v).toFixed(1)}`).join(" ");

  const baseStart = base.length ? base[0][1] : null;
  const dipV = base.length ? base[base.length - 1][1] : null;
  const endV = steered.length ? steered[steered.length - 1][1] : null;

  return (
    <div className="card ck-story">
      <div className="ck-arc" ref={ref}>
        {w > 0 && (
          <svg viewBox={`0 0 ${w} ${h}`} width={w} height={h}>
            {[0, 0.5, 1].map((v) => (
              <g key={v}>
                <line x1={mL} x2={mL + innerW} y1={yOf(v)} y2={yOf(v)} stroke="var(--line)" strokeWidth="1" />
                <text x={mL - 7} y={yOf(v) + 4} textAnchor="end" fontSize="11" fill="var(--mute-2)" fontFamily="var(--mono)">{v.toFixed(1)}</text>
              </g>
            ))}
            <line x1={mL} x2={mL} y1={mT} y2={mT + innerH} stroke="var(--line-2)" strokeWidth="1" />
            <line x1={mL} x2={mL + innerW} y1={mT + innerH} y2={mT + innerH} stroke="var(--line-2)" strokeWidth="1" />

            {/* phase labels along the bottom */}
            <text x={xOf(split / 2)} y={h - 8} textAnchor="middle" fontSize="12" fill="var(--mute)">Fine-tuning</text>
            {steered.length > 0 && (
              <text x={xOf(split + (1 - split) / 2)} y={h - 8} textAnchor="middle" fontSize="12" fill="var(--mute)">Safety re-train</text>
            )}

            {/* the "where it became better" moment: steering applied */}
            {steered.length > 0 && (
              <g opacity={reveal >= split ? 1 : 0.28}>
                <line x1={xOf(split)} x2={xOf(split)} y1={mT} y2={mT + innerH} stroke="var(--good)" strokeWidth="1.3" strokeDasharray="4 3" />
                <text x={xOf(split) + 5} y={mT + 11} fontSize="11" fontWeight="600" fill="var(--good)">steering applied</text>
              </g>
            )}

            {/* drift (red) then recovery (green) */}
            <path d={d(clipTo(declinePts, reveal))} fill="none" stroke="var(--bad)" strokeWidth="2.4" strokeLinejoin="round" strokeLinecap="round" />
            <path d={d(clipTo(recoverPts, reveal))} fill="none" stroke="var(--good)" strokeWidth="2.4" strokeLinejoin="round" strokeLinecap="round" />

            {baseStart != null && reveal > 0.02 && (
              <text x={xOf(0) + 4} y={yOf(baseStart) - 9} fontSize="12" fill="var(--mute)" fontFamily="var(--mono)">{baseStart.toFixed(2)}</text>
            )}
            {dipV != null && reveal >= split - 0.01 && (
              <circle cx={xOf(split)} cy={yOf(dipV)} r="3.6" fill="var(--bg)" stroke="var(--bad)" strokeWidth="2" />
            )}
            {endV != null && reveal >= 0.995 && (
              <g>
                <circle cx={xOf(1)} cy={yOf(endV)} r="4.2" fill="var(--bg)" stroke="var(--good)" strokeWidth="2" />
                <text x={xOf(1) - 6} y={yOf(endV) - 9} textAnchor="end" fontSize="12.5" fontWeight="600" fill="var(--good)" fontFamily="var(--mono)">restored {endV.toFixed(2)}</text>
              </g>
            )}

            <text x={13} y={mT + innerH / 2} textAnchor="middle" fontSize="12" fontWeight="600" fill="var(--ink-soft)"
              transform={`rotate(-90 13 ${mT + innerH / 2})`}>Safety</text>
          </svg>
        )}
      </div>
      {baseStart != null && (
        <p className="ck-story-cap">
          Averaged refusal slid from <b className="mono">{baseStart.toFixed(2)}</b> to{" "}
          <b className="mono bad">{dipV.toFixed(2)}</b> as the model fine-tuned
          {endV != null && <>, then the preventive steer pulled it back to <b className="mono good">{endV.toFixed(2)}</b>.</>}
          {endV == null && "."}
        </p>
      )}
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
function CheckpointCard({ c, selected, onSelect, baseRefusal }) {
  return (
    <button type="button" className={`ck-card ${selected ? "on" : ""} ${c.recommended ? "rec" : ""}`}
      onClick={onSelect} aria-pressed={selected}>
      <div className="ck-card-top">
        <span className="ck-name">{c.name}</span>
        {c.recommended && <Chip color="var(--good)">recommended</Chip>}
      </div>
      <div className="ck-bars">
        <Bar label="Capability" v={c.cap} color="var(--p-tqa)" />
        <Bar label="Safety" v={c.safety} color="var(--good)" baseline={baseRefusal} />
      </div>
      <div className="ck-best">{c.best}</div>
    </button>
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
// a static SVG of the averaged-refusal arc (drift red → recovery green) for the report
function arcSVG(base, steered) {
  if (!base.length) return "";
  const W = 648, H = 230, mL = 44, mR = 16, mT = 18, mB = 34;
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
    `<text x="${mL - 6}" y="${(yOf(v) + 4).toFixed(1)}" text-anchor="end" font-size="11" fill="#adada4" font-family="${MONO}">${v.toFixed(1)}</text>`).join("");
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
    `<text x="13" y="${(mT + ih / 2).toFixed(1)}" text-anchor="middle" font-size="12" font-weight="600" fill="#5d5d56" transform="rotate(-90 13 ${(mT + ih / 2).toFixed(1)})">Safety</text></svg>`;
}

function buildReportHTML({ run, showSteer, steer, batteryRows, latentRows, headlinePP, arcData }) {
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const cols = showSteer ? ["base", "fine-tuned", "safety-aware"] : ["base", "fine-tuned"];
  const row = (k) => batteryRows.find((r) => r.key === k);
  const hbR = row("harmbench_refusal_v2"), mmR = row("mmlu_pro_acc"), tqR = row("truthfulqa_mc1_acc");
  const battery = batteryRows.map((r) => `<tr><td>${esc(r.label)}</td><td>${r.base.toFixed(2)}</td><td>${r.biased.toFixed(2)}</td>${showSteer ? `<td><b>${(r.steered ?? r.biased).toFixed(2)}</b></td>` : ""}</tr>`).join("");
  const latent = showSteer ? latentRows.map((r) => `<tr><td>${esc(titleCase(r.name))}</td><td>${signed(r.unsteered, 1)}</td><td>${signed(r.steered, 1)}</td><td>${esc(r.note || "—")}</td></tr>`).join("") : "";
  // an academic, full-sentence summary grounded in the real deltas
  const summary = hbR
    ? (showSteer
        ? `Fine-tuning ${esc(run.model.label)} on ${esc(run.dataset.domain)}/sft.jsonl reduced HarmBench refusal from ${hbR.base.toFixed(2)} to ${hbR.biased.toFixed(2)}, a safety regression the loss curve did not reveal. Preventive concept steering on the <b>${esc(steer.concept)}</b> direction during a re-train restored refusal to ${(hbR.steered ?? hbR.biased).toFixed(2)}, recovering ${headlinePP} points${mmR ? `, while general capability held (MMLU-Pro ${mmR.base.toFixed(2)} to ${(mmR.steered ?? mmR.biased).toFixed(2)})` : ""}.`
        : `Fine-tuning ${esc(run.model.label)} on ${esc(run.dataset.domain)}/sft.jsonl reduced HarmBench refusal from ${hbR.base.toFixed(2)} to ${hbR.biased.toFixed(2)}, a safety regression the loss curve did not reveal.`)
    : "";
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>${esc(run.project)} · Nauteus run report</title>
<style>
  :root{--ink:#1b1b18;--ink-2:#33332e;--ink-soft:#5d5d56;--mute:#8d8d84;--mute-2:#adada4;--line:#e9e9e5;
    --good:#2f9e44;--bad:#d63a26;
    --sans:"IBM Plex Sans","Gill Sans","Segoe UI",system-ui,sans-serif;--mono:"IBM Plex Mono",Menlo,Consolas,monospace}
  *{box-sizing:border-box}
  body{font:15px/1.55 var(--sans);color:var(--ink-2);background:#fbfbfa;margin:0;padding:40px}
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
  .lead{font-size:14px;color:var(--ink-soft);line-height:1.55;margin:0 0 6px;max-width:62ch}
  h2{font-size:12px;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:var(--ink);margin:24px 0 10px}
  .plot{border:1px solid var(--line);border-radius:8px;padding:12px 14px}
  table{width:100%;border-collapse:collapse;font-size:13px}
  td,th{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}
  th{font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--mute);font-weight:600}
  td:not(:first-child){font-family:var(--mono);font-variant-numeric:tabular-nums}
  tbody tr:last-child td{border-bottom:none}
  .foot{margin-top:24px;font-size:11px;color:var(--mute);border-top:1px solid var(--line);padding-top:12px}
</style></head><body><div class="sheet">
  <div class="head">
    <div>
      <div class="eyebrow">Nauteus run report</div>
      <h1>${esc(run.project)}</h1>
      <div class="meta">${esc(run.model.label)} · ${esc(run.dataset.domain)}/sft.jsonl${showSteer ? ` · realigned (suppressed ${esc(steer.concept)})` : ""}</div>
    </div>
    ${headlinePP != null ? `<div class="hl"><b>+${headlinePP}pp</b><small>HarmBench refusal<br>recovered by steering</small></div>` : ""}
  </div>
  ${summary ? `<p class="lead">${summary}</p>` : ""}
  ${arcData && arcData.base.length ? `<h2>Safety: fine-tuned, then realigned</h2><div class="plot">${arcSVG(arcData.base, arcData.steered)}</div>` : ""}
  <h2>Benchmark scores · ${cols.join(" → ")}</h2>
  <table><thead><tr><th>metric</th><th>base</th><th>fine-tuned</th>${showSteer ? "<th>safety-aware</th>" : ""}</tr></thead><tbody>${battery}</tbody></table>
  ${showSteer ? `<h2>Latent drift per concept · projection vs. base</h2>
  <table><thead><tr><th>concept</th><th>fine-tuned</th><th>safety-aware</th><th>what moved</th></tr></thead><tbody>${latent}</tbody></table>` : ""}
  <div class="foot">Generated by Nauteus from the run logs. Open and print to PDF, or attach to the model card on the Hub.</div>
</div></body></html>`;
}

export default function CheckoutStep({ run, mitigated, steerRun = null }) {
  const steer = run.steer;
  const showSteer = !!(mitigated && steer);
  // the parent (RunView / LiveRunView) already built steerRun with the right shape for its
  // mode; only use it once mitigation has actually run.
  const effSteer = showSteer ? steerRun : null;
  const checkpoints = useMemo(() => buildCheckpoints(run, steer, showSteer), [run, steer, showSteer]);
  const [sel, setSel] = useState(checkpoints[0].id);
  const chosen = checkpoints.find((c) => c.id === sel) || checkpoints[0];
  const [notice, setNotice] = useState(null);
  const notify = (m) => setNotice(m);
  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => setNotice(null), 3600);
    return () => clearTimeout(t);
  }, [notice]);

  const hb = steer?.eval?.harmbench_refusal_v2;
  const headlinePP = showSteer && hb ? Math.round((hb.steered - hb.unsteered) * 100) : null;

  const batteryRows = METRICS.map((m) => {
    const s = run.series.eval[m.key];
    if (!s) return null;
    const base = s[0][1];
    const biased = showSteer && steer.eval[m.key] ? steer.eval[m.key].unsteered : s[s.length - 1][1];
    const steered = showSteer && steer.eval[m.key] ? steer.eval[m.key].steered : null;
    return { key: m.key, label: m.label, base, biased, steered };
  }).filter(Boolean);
  const latentRows = showSteer ? steer.latent : [];
  // base-model refusal (averaged) — the red reference line on each card's Safety bar
  const baseRefusal = meanOf(SAFE_KEYS.map((k) => run.series.eval[k]?.[0]?.[1]));
  const arcData = { base: avgSafety(run.series.eval), steered: effSteer ? avgSafety(effSteer.series.eval) : [] };

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
    download(`${run.id}-report.html`, buildReportHTML({ run, showSteer, steer, batteryRows, latentRows, headlinePP, arcData }));
    notify("Downloaded the 1-page report (open and print to PDF).");
  };

  return (
    <div className="receipt ckpt-page">
      <header className="rcpt-head">
        <div className="rcpt-id">
          <span className="eyebrow">Checkout · ship a checkpoint</span>
          <h3 className="rcpt-title">{run.project}</h3>
          <div className="rcpt-meta mono">{run.model.label} · {run.dataset.domain}/sft.jsonl</div>
        </div>
        <div className="rcpt-aside">
          {headlinePP != null && (
            <div className="ck-headline">
              <span className="ck-headline-n mono">+{headlinePP}<small>pp</small></span>
              <span className="ck-headline-l">HarmBench refusal<br />recovered by steering</span>
            </div>
          )}
          <span className="head-compass"><CompassMark /></span>
        </div>
      </header>

      <p className="rcpt-lead">
        The loss curve looked clean. Here is where safety actually drifted, and where the preventive steer pulled it back.
      </p>

      <section className="rcpt-sec">
        <div className="rcpt-sec-head"><h4>Safety: fine-tuned, realigned</h4>
          <span className="rcpt-hint">Mean refusal across the run, replayed in real time.</span></div>
        <SafetyStory run={run} steerRun={effSteer} />
      </section>

      <section className="rcpt-sec">
        <div className="rcpt-sec-head"><h4>Which checkpoint do you ship?</h4>
          <span className="rcpt-hint">How each checkpoint trades capability against safety.</span></div>
        <div className="ck-layout">
          <div className="ck-list">
            {checkpoints.map((c) => (
              <CheckpointCard key={c.id} c={c} selected={c.id === sel} onSelect={() => setSel(c.id)} baseRefusal={baseRefusal} />
            ))}
          </div>
          <aside className={`ck-detail card ${chosen.recommended ? "rec" : ""}`}>
            <div className="ck-detail-head">
              <span className="ck-detail-name">{chosen.name}</span>
              {chosen.recommended && <Chip color="var(--good)">recommended</Chip>}
            </div>
            <p className="ck-detail-blurb">{chosen.blurb}</p>
            <div className="ck-detail-metrics">
              {METRICS.map((m) => (
                <div className="ck-m" key={m.key}>
                  <span className="ck-m-k">{m.label}</span>
                  <span className="ck-m-v mono">{(chosen.metrics[m.key] ?? 0).toFixed(2)}</span>
                </div>
              ))}
            </div>
            <div className="ck-detail-best">{chosen.best}</div>
            <div className="ck-actions">
              <Button variant="primary"
                onClick={() => notify(`Saving the ${chosen.name} checkpoint and pushing it to a Hugging Face repo.`)}>
                <HFLogo size={15} /> Save &amp; publish to Hugging Face
              </Button>
              <Button onClick={downloadAdapter}>Download adapter</Button>
              <Button variant="ghost" onClick={downloadReport}>Download 1-pager</Button>
            </div>
            {notice && <div className="export-notice" role="status">{notice}</div>}
          </aside>
        </div>
      </section>

    </div>
  );
}
