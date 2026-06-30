import { useMemo, useState } from "react";
import Chart from "./Chart.jsx";
import Legend from "./Legend.jsx";
import { EVAL_SERIES } from "../api/sampleData.js";
import { titleCase } from "../lib/format.js";

/* per-step mean ± sd across a set of [step,val] series → a line + envelope */
function aggregate(arrs, color, label) {
  const present = arrs.filter(Boolean);
  if (!present.length) return null;
  const steps = present[0].map((p) => p[0]);
  const points = [], band = [];
  steps.forEach((st, i) => {
    const vals = present.map((a) => a[i]?.[1]).filter((v) => v != null);
    const mean = vals.reduce((x, y) => x + y, 0) / vals.length;
    const sd = Math.sqrt(vals.reduce((x, y) => x + (y - mean) ** 2, 0) / vals.length);
    points.push([st, mean]);
    band.push([st, mean - sd, mean + sd]);
  });
  return { key: label, label, color, axis: "left", points, band };
}

/* The two stacked, X-aligned plots: top = eval battery + train/eval loss;
   bottom = anthropomorphic-risk projections (or probe). A segmented control
   swaps between AVERAGED (group means ± sd, the default) and ALL CURVES (every
   series). Shared hover crosshair across both, shared live-fill `reveal`. */
export default function DualPlot({ run, reveal = 1, title, subtitle, defaultView = "projection", chartHeight = 196, fill = false, showEarlyStop = false }) {
  // the early-stop marker is event-driven — only drawn once the caller flags it
  const earlyStop = showEarlyStop ? run.earlyStop : null;
  const [hover, setHover] = useState(null);
  const [view, setView] = useState(defaultView);   // projection | probe (bottom measure)
  const [agg, setAgg] = useState(false);            // averaged | all curves (distinct is the default)
  const [hidden, setHidden] = useState(new Set());

  const toggle = (k) => setHidden((h) => {
    const n = new Set(h); n.has(k) ? n.delete(k) : n.add(k); return n;
  });

  const lastStep = useMemo(() => {
    const all = Object.values(run.series.eval)[0] || [];
    return all.length ? all[all.length - 1][0] : 250;
  }, [run]);
  const xDomain = [0, lastStep];

  const lossSeries = useMemo(() => {
    const out = [];
    if (run.series.loss?.train?.length)
      out.push({ key: "train_loss", label: "train loss", color: "var(--p-train)", axis: "right", dashed: true, points: run.series.loss.train });
    if (run.series.loss?.eval?.length)
      out.push({ key: "eval_loss", label: "eval loss", color: "var(--p-eval)", axis: "right", dashed: true, points: run.series.loss.eval });
    return out;
  }, [run]);

  // top chart: averaged group means, or every battery line
  const topSeries = useMemo(() => {
    const metricOf = (k) => run.series.eval[k];
    if (agg) {
      const cap = EVAL_SERIES.filter((m) => m.axis === "metric" && m.group === "capability").map((m) => metricOf(m.key));
      const safe = EVAL_SERIES.filter((m) => m.axis === "metric" && m.group === "safety").map((m) => metricOf(m.key));
      return [
        aggregate(cap, "var(--agg-cap)", "Capability (mean±sd)"),
        aggregate(safe, "var(--agg-safe)", "Safety (mean±sd)"),
        ...lossSeries,
      ].filter(Boolean);
    }
    const out = [];
    EVAL_SERIES.forEach((m) => {
      if (m.axis === "metric" && run.series.eval[m.key])
        out.push({ key: m.key, label: m.label, color: m.color, axis: "left", points: run.series.eval[m.key] });
    });
    return [...out, ...lossSeries];
  }, [run, agg, lossSeries]);

  const lossVals = [...(run.series.loss?.train || []), ...(run.series.loss?.eval || [])].map((p) => p[1]);
  const yRight = lossVals.length ? [Math.min(...lossVals) * 0.9, Math.max(...lossVals) * 1.05] : undefined;

  // bottom chart: mean risk ± sd, or every concept line
  const conceptLines = useMemo(() => run.concepts
    .filter((c) => run.series.trajectory[c.name])
    .map((c) => ({
      key: c.name, label: titleCase(c.name), color: c.color, axis: "left",
      points: run.series.trajectory[c.name].map((e) => [e.step, view === "probe" ? e.probe_prob : e.projection]),
    })), [run, view]);

  const bottomSeries = useMemo(() => {
    if (!agg) return conceptLines;
    const arrs = conceptLines.map((s) => s.points);
    const a = aggregate(arrs, "var(--agg-risk)", view === "probe" ? "Mean P(trait) (mean±sd)" : "Mean risk projection (mean±sd)");
    return a ? [a] : [];
  }, [conceptLines, agg, view]);

  const bottomVals = bottomSeries.flatMap((s) => [
    ...s.points.map((p) => p[1]),
    ...(s.band ? s.band.flatMap((p) => [p[1], p[2]]) : []),
  ]);
  const yBottom = view === "probe"
    ? [0, 1]
    : [Math.min(0, ...bottomVals) - 0.5, Math.max(0, ...bottomVals) + 0.5];

  return (
    <div className={`dualplot ${fill ? "fill" : ""}`}>
      {title && (
        <div className="dp-head">
          <div className="col">
            <span className="title-sm dp-title">{title}</span>
            {subtitle && <span className="muted dp-sub">{subtitle}</span>}
          </div>
          <div className="dp-controls">
            <div className="toggle seg">
              <button className={agg ? "on" : ""} onClick={() => setAgg(true)}>Averaged</button>
              <button className={!agg ? "on" : ""} onClick={() => setAgg(false)}>All curves</button>
            </div>
          </div>
        </div>
      )}

      <div className="card chart-card">
        <div className="chart-head">
          <span className="ct">Training loss, capabilities and refusal</span>
        </div>
        <Legend series={topSeries} hidden={hidden} onToggle={toggle} />
        <Chart height={chartHeight} series={topSeries} xDomain={xDomain} yLeft={[0, 1]} yRight={yRight}
          earlyStop={earlyStop} reveal={reveal} hover={hover} onHover={setHover} hiddenKeys={hidden}
          formatLeft={(v) => v.toFixed(2)} formatRight={(v) => v.toFixed(2)}
          xLabel="Training step" yLeftLabel="Score" yRightLabel="Loss" />
      </div>

      <div className="card chart-card">
        <div className="chart-head">
          <span className="ct">Emergent misalignment risks</span>
          <div className="row gap10">
            <div className="toggle">
              <button className={view === "projection" ? "on" : ""} onClick={() => setView("projection")}>Projection</button>
              <button className={view === "probe" ? "on" : ""} onClick={() => setView("probe")}>Probe</button>
            </div>
          </div>
        </div>
        <Legend series={bottomSeries} hidden={hidden} onToggle={toggle} />
        <Chart height={chartHeight} series={bottomSeries} xDomain={xDomain} yLeft={yBottom}
          earlyStop={earlyStop} reveal={reveal} hover={hover} onHover={setHover} hiddenKeys={hidden}
          zeroLine={view !== "probe"}
          xLabel="Training step" yLeftLabel={view === "probe" ? "P(trait)" : "Projection"}
          formatLeft={(v) => (view === "probe" ? v.toFixed(2) : v.toFixed(1))} />
      </div>
    </div>
  );
}
