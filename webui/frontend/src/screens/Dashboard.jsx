import { useState } from "react";
import { APPS, kindMeta, deltaMeta, fmtPct, spark } from "../data/demo.js";

const COLOR_BY_KEY = {
  mmlu_pro_acc: "#4659c9",
  truthfulqa_mc1_acc: "#1f9e86",
  harmbench_refusal_v2: "#2f43e0",
  strongreject_refusal_v2: "#d8483a",
};

export default function Dashboard() {
  const [appIdx, setAppIdx] = useState(0);
  const [reportOpen, setReportOpen] = useState(false);
  const app = APPS[appIdx];

  const metricCards = app.metrics.map((m) => {
    const km = kindMeta(m.kind);
    const delta = m.base == null || m.final == null ? null : m.final - m.base;
    const dm = deltaMeta(delta);
    return {
      key: m.key, label: m.label, kindLabel: km.label, kindColor: km.color, kindBg: km.bg,
      baseStr: fmtPct(m.base), finalStr: fmtPct(m.final),
      deltaStr: dm.str, deltaColor: dm.color, deltaBg: dm.bg, deltaArrow: dm.arrow,
      spark: spark(app.trajectory.map((p) => p[m.key])),
    };
  });

  // big chart geometry
  const padL = 44, padR = 18, padT = 14, padB = 30, W = 720, H = 250;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const traj = app.trajectory;
  const xAt = (i) => padL + (traj.length <= 1 ? 0 : (i * plotW) / (traj.length - 1));
  const yAt = (v) => padT + (1 - v) * plotH;
  const chartLines = [], chartDots = [], chartLegend = [];
  app.metrics.forEach((m) => {
    const col = COLOR_BY_KEY[m.key] || "#586477";
    const present = traj.map((p, i) => ({ v: p[m.key], i })).filter((o) => o.v != null);
    if (!present.length) return;
    chartLegend.push({ label: m.label, color: col });
    chartLines.push({ key: m.key, color: col, points: present.map((o) => xAt(o.i).toFixed(1) + "," + yAt(o.v).toFixed(1)).join(" ") });
    present.forEach((o) => chartDots.push({ cx: xAt(o.i).toFixed(1), cy: yAt(o.v).toFixed(1), color: col }));
  });
  const chartYTicks = [0, 0.25, 0.5, 0.75, 1].map((v) => ({ y: yAt(v).toFixed(1), ty: (yAt(v) + 3.5).toFixed(1), label: Math.round(v * 100) + "%" }));
  const chartXLabels = traj.map((p, i) => ({ x: xAt(i).toFixed(1), label: p.step === -1 ? "base" : p.step >= 1e9 ? "final" : String(p.step) }));

  return (
    <div>
      {/* app tabs */}
      <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "22px", flexWrap: "wrap" }}>
        {APPS.map((a, i) => {
          const active = i === appIdx;
          return (
            <button
              key={a.app}
              className="hv-bd-blue"
              onClick={() => { setAppIdx(i); setReportOpen(false); }}
              style={{
                display: "flex", alignItems: "center", gap: "9px", padding: "8px 15px", borderRadius: "11px",
                border: "1px solid " + (active ? "#c2ccf4" : "#e2e9f3"), cursor: "pointer", font: "inherit", fontSize: "13.5px",
                textTransform: "capitalize", transition: "border-color .15s, background .15s",
                background: active ? "#f4f6fe" : "#ffffff", color: active ? "#2031c4" : "#48546e",
                boxShadow: active ? "0 1px 2px rgba(35,52,194,0.08)" : "none",
              }}
            >
              <span style={{ fontWeight: 600 }}>{a.app}</span>
              <span style={{ fontSize: "11.5px", color: active ? "#7d8ae8" : "#a6aebe", fontFamily: "'JetBrains Mono',monospace" }}>{a.n_checkpoints} ckpt</span>
            </button>
          );
        })}
        <div style={{ flex: 1 }} />
        <div style={{ fontSize: "12.5px", color: "#98a2b3", fontFamily: "'JetBrains Mono',monospace" }}>{app.model}</div>
      </div>

      {/* metric cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(248px,1fr))", gap: "16px", marginBottom: "26px" }}>
        {metricCards.map((m) => (
          <div key={m.key} style={{ background: "#ffffff", border: "1px solid #e5ebf4", borderRadius: "14px", padding: "18px 18px 14px", boxShadow: "0 1px 2px rgba(20,32,64,0.03)" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "14px" }}>
              <span style={{ fontSize: "10.5px", letterSpacing: "0.08em", textTransform: "uppercase", fontWeight: 600, color: m.kindColor, background: m.kindBg, padding: "4px 8px", borderRadius: "6px" }}>{m.kindLabel}</span>
              <span style={{ fontSize: "12px", fontWeight: 600, color: m.deltaColor, background: m.deltaBg, padding: "4px 8px", borderRadius: "6px", fontFamily: "'JetBrains Mono',monospace" }}>{m.deltaArrow} {m.deltaStr}</span>
            </div>
            <div style={{ fontSize: "13px", color: "#69748a", marginBottom: "6px" }}>{m.label}</div>
            <div style={{ display: "flex", alignItems: "baseline", gap: "8px", marginBottom: "12px" }}>
              <span style={{ fontFamily: "'Hanken Grotesk',sans-serif", fontSize: "32px", fontWeight: 500, color: "#0f1830", lineHeight: 1 }}>{m.finalStr}</span>
              <span style={{ fontSize: "12.5px", color: "#a6aebe", fontFamily: "'JetBrains Mono',monospace" }}>from {m.baseStr}</span>
            </div>
            <svg viewBox="0 0 132 40" width="100%" height="40" preserveAspectRatio="none" style={{ display: "block" }}>
              <polyline points={m.spark.points} fill="none" stroke={m.kindColor} strokeWidth="1.8" strokeLinejoin="round" strokeLinecap="round" opacity="0.85" />
              <circle cx={m.spark.firstX} cy={m.spark.firstY} r="2.4" fill="#fff" stroke={m.kindColor} strokeWidth="1.6" />
              <circle cx={m.spark.lastX} cy={m.spark.lastY} r="3" fill={m.kindColor} />
            </svg>
          </div>
        ))}
      </div>

      {/* big chart */}
      <div style={{ background: "#ffffff", border: "1px solid #e5ebf4", borderRadius: "16px", padding: "22px 24px", boxShadow: "0 1px 2px rgba(20,32,64,0.03)" }}>
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "20px", marginBottom: "18px", flexWrap: "wrap" }}>
          <div>
            <h3 style={{ margin: 0, fontFamily: "'Hanken Grotesk',sans-serif", fontWeight: 300, fontSize: "19px", color: "#15203c" }}>Metric trajectory</h3>
            <p style={{ margin: "5px 0 0", fontSize: "13px", color: "#838fa4" }}>Benchmark scores across fine-tuning checkpoints. Safety drops as capability holds.</p>
          </div>
          <div style={{ display: "flex", gap: "16px", flexWrap: "wrap" }}>
            {chartLegend.map((l) => (
              <div key={l.label} style={{ display: "flex", alignItems: "center", gap: "7px", fontSize: "12.5px", color: "#48546e" }}>
                <span style={{ width: "13px", height: "3px", borderRadius: "2px", background: l.color }} />{l.label}
              </div>
            ))}
          </div>
        </div>
        <svg viewBox="0 0 720 250" width="100%" style={{ display: "block", overflow: "visible" }}>
          {chartYTicks.map((g) => (
            <g key={g.label}>
              <line x1="44" y1={g.y} x2="702" y2={g.y} stroke="#eef2f9" strokeWidth="1" />
              <text x="36" y={g.ty} textAnchor="end" fontSize="11" fill="#a6aebe" fontFamily="'JetBrains Mono',monospace">{g.label}</text>
            </g>
          ))}
          {chartXLabels.map((x, i) => (
            <text key={i} x={x.x} y="244" textAnchor="middle" fontSize="11" fill="#a6aebe" fontFamily="'JetBrains Mono',monospace">{x.label}</text>
          ))}
          {chartLines.map((ln) => (
            <polyline key={ln.key} points={ln.points} fill="none" stroke={ln.color} strokeWidth="2.2" strokeLinejoin="round" strokeLinecap="round" />
          ))}
          {chartDots.map((d, i) => (
            <circle key={i} cx={d.cx} cy={d.cy} r="3.4" fill="#fff" stroke={d.color} strokeWidth="2" />
          ))}
        </svg>
      </div>

      {/* report card */}
      <div style={{ marginTop: "18px", background: "#ffffff", border: "1px solid #e5ebf4", borderRadius: "16px", padding: "18px 24px", display: "flex", alignItems: "center", justifyContent: "space-between", gap: "20px", flexWrap: "wrap", boxShadow: "0 1px 2px rgba(20,32,64,0.03)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          <span style={{ flex: "none", width: "46px", height: "46px", borderRadius: "11px", background: "linear-gradient(135deg,#e6eafc,#d6ddfb)", display: "flex", alignItems: "center", justifyContent: "center", backgroundImage: "radial-gradient(circle, rgba(47,67,224,0.22) 0.8px, transparent 1.1px)", backgroundSize: "5px 5px" }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2f43e0" strokeWidth="1.8">
              <rect x="4" y="3" width="16" height="18" rx="2" />
              <line x1="8" y1="8" x2="16" y2="8" />
              <line x1="8" y1="12" x2="16" y2="12" />
              <line x1="8" y1="16" x2="13" y2="16" />
            </svg>
          </span>
          <div>
            <div style={{ fontWeight: 600, fontSize: "14.5px", color: "#15203c" }}>Full evaluation report</div>
            <div style={{ fontSize: "13px", color: "#838fa4" }}>Detailed per-benchmark charts and per-checkpoint tables, rendered server-side.</div>
          </div>
        </div>
        <button className="hv-secondary" onClick={() => setReportOpen((o) => !o)} style={{ flex: "none", padding: "10px 18px", borderRadius: "10px", border: "1px solid #d3dbeb", background: "#f7f9fd", color: "#283353", font: "inherit", fontSize: "13.5px", fontWeight: 600, cursor: "pointer" }}>
          {reportOpen ? "Hide report" : "View full report"}
        </button>
      </div>

      {reportOpen && (
        <div style={{ marginTop: "14px", border: "1px dashed #ccd6ea", borderRadius: "14px", padding: "40px 32px", textAlign: "center", backgroundImage: "radial-gradient(circle, rgba(20,32,64,0.05) 0.7px, transparent 1px)", backgroundSize: "6px 6px", backgroundColor: "#f8fafe" }}>
          <div style={{ fontFamily: "'Hanken Grotesk',sans-serif", fontSize: "18px", color: "#283353", marginBottom: "6px" }}>Report not generated yet</div>
          <div style={{ fontSize: "13.5px", color: "#838fa4", maxWidth: "440px", margin: "0 auto", lineHeight: 1.5 }}>
            Run an <span style={{ fontFamily: "'JetBrains Mono',monospace", color: "#2f43e0" }}>eval</span> job from the Runs screen to produce the HTML report. Once available it loads inline here from <span style={{ fontFamily: "'JetBrains Mono',monospace" }}>/report</span>.
          </div>
        </div>
      )}
    </div>
  );
}
