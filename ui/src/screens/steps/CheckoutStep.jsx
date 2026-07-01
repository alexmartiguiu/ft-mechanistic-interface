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

// grouped bar chart: capability / refusal / safety, each compared across model stages
// (base → fine-tuned → realigned). Bars grow up on mount for a light reveal.
function MetricBars({ groups, stages }) {
  const reveal = useReveal(true, 1100);
  const [ref, size] = useSize();
  const w = size.w || 600, h = 240;
  const mL = 38, mR = 14, mT = 16, mB = 54;
  const iw = Math.max(10, w - mL - mR), ih = Math.max(10, h - mT - mB);
  const yOf = (v) => mT + (1 - v) * ih;
  const gW = iw / groups.length;
  const inner = gW * 0.72;                              // the group's bars occupy 72% of its slot
  const bW = inner / stages.length;
  const x0 = (gi) => mL + gi * gW + (gW - inner) / 2;

  return (
    <div className="card ck-bars-card">
      <div className="ck-bars-plot" ref={ref}>
        {w > 0 && (
          <svg viewBox={`0 0 ${w} ${h}`} width={w} height={h}>
            {[0, 0.25, 0.5, 0.75, 1].map((v) => (
              <g key={v}>
                <line x1={mL} x2={mL + iw} y1={yOf(v)} y2={yOf(v)} stroke="var(--line)" strokeWidth="1" />
                <text x={mL - 6} y={yOf(v) + 4} textAnchor="end" fontSize="12.6" fill="var(--mute-2)" fontFamily="var(--mono)">{v.toFixed(2)}</text>
              </g>
            ))}
            <line x1={mL} x2={mL} y1={mT} y2={mT + ih} stroke="var(--line-2)" strokeWidth="1" />
            <line x1={mL} x2={mL + iw} y1={mT + ih} y2={mT + ih} stroke="var(--line-2)" strokeWidth="1" />
            {groups.map((g, gi) => (
              <g key={g.key}>
                {stages.map((s, si) => {
                  const v = (g[s.key] ?? 0) * reveal;
                  const bx = x0(gi) + si * bW;
                  const by = yOf(v);
                  return (
                    <g key={s.key}>
                      <rect x={bx + 1} y={by} width={Math.max(0, bW - 2)} height={Math.max(0, mT + ih - by)} rx="1.5" fill={s.color} />
                      {reveal > 0.98 && (
                        <text x={bx + bW / 2} y={by - 4} textAnchor="middle" fontSize="12" fill="var(--ink-soft)" fontFamily="var(--mono)">{(g[s.key] ?? 0).toFixed(2)}</text>
                      )}
                    </g>
                  );
                })}
                <text x={x0(gi) + inner / 2} y={mT + ih + 18} textAnchor="middle" fontSize="15" fontWeight="600" fill="var(--ink-soft)">{g.label}</text>
                <text x={x0(gi) + inner / 2} y={mT + ih + 32} textAnchor="middle" fontSize="12" fill="var(--mute)" fontFamily="var(--mono)">{g.sub}</text>
              </g>
            ))}
          </svg>
        )}
      </div>
      <div className="ck-bars-legend">
        {stages.map((s) => (
          <span className="ck-bl" key={s.key}><span className="sw" style={{ background: s.color }} />{s.label}</span>
        ))}
      </div>
    </div>
  );
}

// ── the interactive safety arc: mean refusal drifting down, then recovering after the steer ──
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

// One continuous mean-refusal arc: high → drifts down during fine-tuning (red) → recovers
// during the safety-aware re-train (green). Live-fills left to right, marking the steering
// moment and the restored value once the draw reaches them.
function SafetyStory({ run, steerRun }) {
  const reveal = useReveal(true, 2600);
  const [ref, size] = useSize();
  const w = size.w || 600, h = 224;
  const mL = 58, mR = 18, mT = 18, mB = 38;
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
                <text x={mL - 9} y={yOf(v) + 4} textAnchor="end" fontSize="13.2" fill="var(--mute-2)" fontFamily="var(--mono)">{v.toFixed(1)}</text>
              </g>
            ))}
            <line x1={mL} x2={mL} y1={mT} y2={mT + innerH} stroke="var(--line-2)" strokeWidth="1" />
            <line x1={mL} x2={mL + innerW} y1={mT + innerH} y2={mT + innerH} stroke="var(--line-2)" strokeWidth="1" />

            {/* phase labels along the bottom */}
            <text x={xOf(split / 2)} y={h - 8} textAnchor="middle" fontSize="14.4" fill="var(--mute)">Fine-tuning</text>
            {steered.length > 0 && (
              <text x={xOf(split + (1 - split) / 2)} y={h - 8} textAnchor="middle" fontSize="14.4" fill="var(--mute)">Safety re-train</text>
            )}

            {/* the "where it became better" moment: steering applied */}
            {steered.length > 0 && (
              <g opacity={reveal >= split ? 1 : 0.28}>
                <line x1={xOf(split)} x2={xOf(split)} y1={mT} y2={mT + innerH} stroke="var(--good)" strokeWidth="1.3" strokeDasharray="4 3" />
                <text x={xOf(split) + 5} y={mT + 11} fontSize="13.2" fontWeight="600" fill="var(--good)">steering applied</text>
              </g>
            )}

            {/* drift (red) then recovery (green) */}
            <path d={d(clipTo(declinePts, reveal))} fill="none" stroke="var(--bad)" strokeWidth="2.4" strokeLinejoin="round" strokeLinecap="round" />
            <path d={d(clipTo(recoverPts, reveal))} fill="none" stroke="var(--good)" strokeWidth="2.4" strokeLinejoin="round" strokeLinecap="round" />

            {baseStart != null && reveal > 0.02 && (
              <text x={xOf(0) + 4} y={yOf(baseStart) - 9} fontSize="14.4" fill="var(--mute)" fontFamily="var(--mono)">{baseStart.toFixed(2)}</text>
            )}
            {dipV != null && reveal >= split - 0.01 && (
              <circle cx={xOf(split)} cy={yOf(dipV)} r="3.6" fill="var(--bg)" stroke="var(--bad)" strokeWidth="2" />
            )}
            {endV != null && reveal >= 0.995 && (
              <g>
                <circle cx={xOf(1)} cy={yOf(endV)} r="4.2" fill="var(--bg)" stroke="var(--good)" strokeWidth="2" />
                <text x={xOf(1) - 6} y={yOf(endV) - 9} textAnchor="end" fontSize="15" fontWeight="600" fill="var(--good)" fontFamily="var(--mono)">restored {endV.toFixed(2)}</text>
              </g>
            )}

            {/* y-axis label — spelled out so the axis reads unambiguously */}
            <text x={16} y={mT + innerH / 2} textAnchor="middle" fontSize="13.8" fontWeight="600" fill="var(--ink-soft)"
              transform={`rotate(-90 16 ${mT + innerH / 2})`}>Safety: mean refusal (↑ safer)</text>
          </svg>
        )}
      </div>
      {baseStart != null && (
        <p className="ck-story-cap">
          Mean refusal slid from <b className="mono">{baseStart.toFixed(2)}</b> to{" "}
          <b className="mono bad">{dipV.toFixed(2)}</b> as the model fine-tuned
          {endV != null && <>, then the preventive steer pulled it back to <b className="mono good">{endV.toFixed(2)}</b>.</>}
          {endV == null && "."}
        </p>
      )}
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

// a formal, full-sentence summary grounded in the real deltas — shared by the on-screen
// "Insight" line and the report, so the two never drift apart.
function summarize({ run, showSteer, steer, batteryRows, headlinePP }) {
  const row = (k) => batteryRows.find((r) => r.key === k);
  const hbR = row("harmbench_refusal_v2"), mmR = row("mmlu_pro_acc");
  if (!hbR) return "";
  const regress = `Fine-tuning ${run.model.label} on sft.jsonl reduced HarmBench refusal from ${hbR.base.toFixed(2)} to ${hbR.biased.toFixed(2)}, a safety regression that the loss curve did not reveal.`;
  if (!showSteer) return regress;
  const cap = mmR ? `, while general capability was preserved (MMLU-Pro ${mmR.base.toFixed(2)} to ${(mmR.steered ?? mmR.biased).toFixed(2)})` : "";
  return `${regress} Preventive malign-concept steering on the ${steer.concept} direction during a re-train restored refusal to ${(hbR.steered ?? hbR.biased).toFixed(2)}, a recovery of ${headlinePP} points${cap}.`;
}

function buildReportHTML({ run, showSteer, steer, batteryRows, latentRows, headlinePP, barsData, arcData, chosen, summary }) {
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
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>${esc(run.project)} · Nauteus run report</title>
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
      <div class="eyebrow">Nauteus run report</div>
      <h1>${esc(run.project)}</h1>
      <div class="meta">${esc(run.model.label)} · sft.jsonl${showSteer ? ` · realigned (suppressed ${esc(steer.concept)})` : ""}</div>
    </div>
    ${headlinePP != null ? `<div class="hl"><b>+${headlinePP}pp</b><small>HarmBench refusal<br>recovered by steering</small></div>` : ""}
  </div>
  ${summary ? `<p class="lead"><b class="tldr">Insight:</b> ${esc(summary)}</p>` : ""}
  ${arcData && arcData.base.length ? `<h2>Safety across the run</h2><div class="plot">${arcSVG(arcData.base, arcData.steered)}</div>` : ""}
  ${barsData && barsData.groups.length ? `<h2>Capability, refusal &amp; safety</h2><div class="plot">${barsSVG(barsData.groups, barsData.stages)}</div>` : ""}
  <h2>Benchmark scores</h2>
  <table><thead><tr><th>metric</th><th>base</th><th>fine-tuned</th>${showSteer ? "<th>realigned</th>" : ""}</tr></thead><tbody>${battery}</tbody></table>
  ${showSteer ? `<h2>Malign-concept drift · projection Δ (fine-tuned → realigned)</h2>
  <table><thead><tr><th>malign concept</th><th>fine-tuned</th><th>realigned</th><th>Δ</th><th class="wm">what moved</th></tr></thead><tbody>${drift}</tbody></table>` : ""}
  <div class="foot"><b>Artefact — shipped checkpoint:</b> ${esc(chosen.name)}.${artefactLinks ? ` ${artefactLinks}.` : ""} Generated by Nauteus from the run logs; open and print to PDF, or attach to the model card on the Hub.</div>
</div></body></html>`;
}

export default function CheckoutStep({ run, mitigated, steerRun = null }) {
  const steer = run.steer;
  const showSteer = !!(mitigated && steer);
  // only use the steered curves once mitigation has actually run
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
  // capability / refusal / safety, each across the model stages we can show
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
  const summary = summarize({ run, showSteer, steer, batteryRows, headlinePP });

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
    download(`${run.id}-report.html`, buildReportHTML({ run, showSteer, steer, batteryRows, latentRows, headlinePP, barsData, arcData, chosen, summary }));
    notify("Downloaded the 1-page report (open and print to PDF).");
  };

  return (
    <div className="receipt ckpt-page">
      <header className="rcpt-head">
        <div className="rcpt-id">
          <span className="eyebrow">Checkout · ship a checkpoint</span>
          <h3 className="rcpt-title">{run.project}</h3>
          <div className="rcpt-meta mono">{run.model.label} · sft.jsonl</div>
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

      {/* ── Pane 1: Reporting ── */}
      <section className="rcpt-pane">
        <div className="rcpt-pane-head"><span className="eyebrow">Reporting</span>
          <span className="rcpt-pane-hint">What the loss curve hid, and how the run behaved across stages.</span></div>

        <div className="rcpt-sec">
          <div className="rcpt-sec-head"><h4>Safety across the run</h4>
            <span className="rcpt-hint">Mean refusal (HarmBench + StrongREJECT), replayed in real time.</span></div>
          <SafetyStory run={run} steerRun={effSteer} />
        </div>

        {summary && <p className="ck-insight"><b>Insight:</b> {summary}</p>}

        <div className="ck-report-grid">
          <div className="rcpt-sec">
            <div className="rcpt-sec-head"><h4>Capability, refusal &amp; safety</h4>
              <span className="rcpt-hint">Base vs fine-tuned{showSteer ? " vs realigned" : ""}.</span></div>
            <MetricBars groups={barGroups} stages={stages} />
          </div>
          <div className="rcpt-sec">
            <div className="rcpt-sec-head"><h4>Benchmark scores</h4>
              <span className="rcpt-hint">Per-metric, across the model stages.</span></div>
            <BenchmarkTable rows={batteryRows} showSteer={showSteer} />
          </div>
        </div>

        {showSteer && latentRows.length > 0 && (
          <div className="rcpt-sec">
            <div className="rcpt-sec-head"><h4>Malign-concept drift</h4>
              <span className="rcpt-hint">Projection Δ per malign concept between the fine-tuned and realigned checkpoints.</span></div>
            <ConceptDriftTable rows={latentRows} />
          </div>
        )}
      </section>

      {/* ── Pane 2: Artefacts ── */}
      <section className="rcpt-pane">
        <div className="rcpt-pane-head"><span className="eyebrow">Artefacts</span>
          <span className="rcpt-pane-hint">Choose a checkpoint, then export the adapter, report, or push to the Hub.</span></div>

        <div className="rcpt-sec">
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
        </div>
      </section>

    </div>
  );
}
