import { useEffect, useState } from "react";
import { getSeries } from "../api.js";
import { demoSeries, EVAL_SERIES_META, CONCEPT_PALETTE } from "../data/demo.js";
import SeriesChart from "./SeriesChart.jsx";

// Both charts for one run, computed client-side from the raw series (no matplotlib).
// Each chart's legend doubles as a show/hide filter for its series.
export default function RunCharts({ run, live }) {
  const [data, setData] = useState(null);
  const [evalVis, setEvalVis] = useState(null);  // null → "all visible"
  const [monVis, setMonVis] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setData(null); setEvalVis(null); setMonVis(null);
    const fallback = () => { if (!cancelled) setData(demoSeries(run.id)); };
    if (live) getSeries(run.id, run.model).then((d) => { if (!cancelled) setData(d); }).catch(fallback);
    else fallback();
    return () => { cancelled = true; };
  }, [run.id, run.model, live]);

  if (!data) return <div style={{ padding: "0 20px 18px", fontSize: "12.5px", color: "#a6aebe" }}>Computing charts…</div>;

  // eval series: 4 metrics (left) + 2 loss (right, dashed), only those with points
  const evalSeries = EVAL_SERIES_META.map((s) => {
    const pts = s.axis === "R"
      ? (data.loss?.[s.key === "train_loss" ? "train" : "eval"] || [])
      : (data.eval?.[s.key] || []);
    return { ...s, points: pts };
  }).filter((s) => s.points.length);

  // monitor series: one per concept (projection ⟨h,v̂⟩), coloured from the palette
  const concepts = Object.keys(data.monitor || {}).sort();
  const monSeries = concepts.map((c, i) => ({
    key: c, label: c, color: CONCEPT_PALETTE[i % CONCEPT_PALETTE.length], axis: "L",
    points: (data.monitor[c] || []),
  })).filter((s) => s.points.length);

  const evVis = evalVis ?? new Set(evalSeries.map((s) => s.key));
  const moVis = monVis ?? new Set(monSeries.map((s) => s.key));
  const toggle = (setter, cur, key) => setter(() => { const n = new Set(cur); n.has(key) ? n.delete(key) : n.add(key); return n; });

  return (
    <div style={{ padding: "0 20px 18px", display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(320px,1fr))", gap: "16px" }}>
      <ChartPanel title="Evaluation · accuracy / refusal + loss vs step" swatch="#2f43e0">
        {evalSeries.length ? (
          <SeriesChart series={evalSeries} visible={evVis} onToggle={(k) => toggle(setEvalVis, evVis, k)} leftDomain="unit" unit yRightLabel="loss" />
        ) : <Empty live={live} />}
      </ChartPanel>
      <ChartPanel title="Concept monitor · projection ⟨h, v̂⟩ vs step" swatch="#7a6a8a">
        {monSeries.length ? (
          <SeriesChart series={monSeries} visible={moVis} onToggle={(k) => toggle(setMonVis, moVis, k)} leftDomain="auto" baselineZero />
        ) : <Empty live={live} label="No concept-monitor trajectory recorded for this run." />}
      </ChartPanel>
    </div>
  );
}

function ChartPanel({ title, swatch, children }) {
  return (
    <div style={{ minWidth: 0, border: "1px solid #eef2f9", borderRadius: "12px", overflow: "hidden", background: "#fff" }}>
      <div style={{ padding: "10px 14px", borderBottom: "1px solid #f0f3f9", fontSize: "12px", fontWeight: 600, color: "#48546e", display: "flex", alignItems: "center", gap: "8px" }}>
        <span style={{ width: "9px", height: "3px", borderRadius: "2px", background: swatch }} />{title}
      </div>
      <div style={{ padding: "14px" }}>{children}</div>
    </div>
  );
}

function Empty({ live, label }) {
  return (
    <div style={{ minHeight: "180px", display: "flex", alignItems: "center", justifyContent: "center", textAlign: "center", fontSize: "12.5px", color: "#a6aebe", lineHeight: 1.6 }}>
      {label || (live ? "No series for this run." : "Series compute from the backend — connect to view.")}
    </div>
  );
}
