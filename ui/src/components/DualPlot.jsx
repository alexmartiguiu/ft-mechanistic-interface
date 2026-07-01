import { useMemo, useState } from "react";
import Chart from "./Chart.jsx";
import Legend from "./Legend.jsx";
import InfoDot from "./InfoDot.jsx";
import { EVAL_SERIES } from "../api/sampleData.js";
import { titleCase } from "../lib/format.js";

/* "Show mean" toggle — each chart carries its own. */
function MeanSwitch({ on, onToggle }) {
  return (
    <button type="button" role="switch" aria-checked={on} className="switch-field" onClick={onToggle}>
      <span className={`switch ${on ? "on" : ""}`}><span className="knob" /></span>
      <span className="switch-lab">Show mean</span>
    </button>
  );
}

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
  // "show mean" is per-chart: the top (loss/eval battery) defaults to the aggregated
  // mean±sd, the bottom (concept trajectories) defaults to every curve.
  const [aggTop, setAggTop] = useState(true);
  const [aggBottom, setAggBottom] = useState(false);
  const [hidden, setHidden] = useState(new Set());

  const toggle = (k) => setHidden((h) => {
    const n = new Set(h); n.has(k) ? n.delete(k) : n.add(k); return n;
  });

  // size the shared x-axis to the true end of the run — across the eval battery, the
  // loss curves AND the concept trajectories. Live runs end these series at different
  // steps (e.g. eval at a checkpoint, drift out to ~875), so keying only off the eval
  // series left the trajectory dots spilling off the right edge. Recorded runs end
  // every series at the same step, so this leaves them unchanged.
  const lastStep = useMemo(() => {
    let mx = 0;
    const bump = (s) => { const p = s && s[s.length - 1]; if (p) mx = Math.max(mx, p[0]); };
    Object.values(run.series.eval || {}).forEach(bump);
    bump(run.series.loss?.train);
    bump(run.series.loss?.eval);
    Object.values(run.series.trajectory || {}).forEach((seq) => {
      const e = seq && seq[seq.length - 1];
      if (e) mx = Math.max(mx, e.step);
    });
    return mx > 0 ? mx : 250;
  }, [run]);
  const xDomain = [0, lastStep];

  const lossSeries = useMemo(() => {
    const out = [];
    if (run.series.loss?.train?.length)
      out.push({ key: "train_loss", label: "Train loss", color: "var(--p-train)", axis: "right", dashed: true, points: run.series.loss.train });
    if (run.series.loss?.eval?.length)
      out.push({ key: "eval_loss", label: "Eval loss", color: "var(--p-eval)", axis: "right", dashed: true, points: run.series.loss.eval });
    return out;
  }, [run]);

  // top chart: averaged group means, or every battery line
  const topSeries = useMemo(() => {
    const metricOf = (k) => run.series.eval[k];
    if (aggTop) {
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
  }, [run, aggTop, lossSeries]);

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
    if (!aggBottom) return conceptLines;
    const arrs = conceptLines.map((s) => s.points);
    const a = aggregate(arrs, "var(--agg-risk)", view === "probe" ? "Mean P(trait) (mean±sd)" : "Mean risk projection (mean±sd)");
    return a ? [a] : [];
  }, [conceptLines, aggBottom, view]);

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
        </div>
      )}

      <div className="card chart-card">
        <div className="chart-head">
          <span className="row gap6">
            <span className="ct">Training loss, capabilities and refusal</span>
            <InfoDot label="What is this loss?">
              <b>Loss</b> is the LoRA fine-tune’s cross-entropy (next-token prediction) loss, plotted on
              the right-hand axis — <em>train</em> is measured on the SFT batches, <em>eval</em> on a
              held-out split. It tracks how well the model fits the fine-tuning data, <em>not</em> its
              safety: a clean, falling loss curve can still hide the drift shown below.
            </InfoDot>
          </span>
          <MeanSwitch on={aggTop} onToggle={() => setAggTop((v) => !v)} />
        </div>
        <Legend series={topSeries} hidden={hidden} onToggle={toggle} />
        <Chart height={chartHeight} series={topSeries} xDomain={xDomain} yLeft={[0, 1]} yRight={yRight}
          earlyStop={earlyStop} reveal={reveal} hover={hover} onHover={setHover} hiddenKeys={hidden}
          formatLeft={(v) => v.toFixed(2)} formatRight={(v) => v.toFixed(2)}
          xLabel="Training step" yLeftLabel="Score" yRightLabel="Loss" />
      </div>

      <div className="card chart-card">
        <div className="chart-head">
          <span className="ct">Emergent risks</span>
          <div className="row gap10">
            <div className="toggle">
              <button className={view === "projection" ? "on" : ""} onClick={() => setView("projection")}>Projection</button>
              <button className={view === "probe" ? "on" : ""} onClick={() => setView("probe")}>Probe</button>
            </div>
            <MeanSwitch on={aggBottom} onToggle={() => setAggBottom((v) => !v)} />
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
