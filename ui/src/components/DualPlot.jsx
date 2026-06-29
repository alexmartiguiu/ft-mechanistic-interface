import { useMemo, useState } from "react";
import Chart from "./Chart.jsx";
import Legend from "./Legend.jsx";
import { EVAL_SERIES } from "../api/sampleData.js";
import { titleCase } from "../lib/format.js";

/* The two stacked, X-aligned plots from the brief:
   top = train/eval loss + eval battery; bottom = concept projections (or probe).
   Shared hover crosshair across both, shared live-fill `reveal`. */
export default function DualPlot({ run, reveal = 1, title, defaultView = "projection" }) {
  const [hover, setHover] = useState(null);
  const [view, setView] = useState(defaultView);
  const [hidden, setHidden] = useState(new Set());

  const toggle = (k) => setHidden((h) => {
    const n = new Set(h); n.has(k) ? n.delete(k) : n.add(k); return n;
  });

  const lastStep = useMemo(() => {
    const all = Object.values(run.series.eval)[0] || [];
    return all.length ? all[all.length - 1][0] : 250;
  }, [run]);
  const xDomain = [0, lastStep];

  // top chart series
  const topSeries = useMemo(() => {
    const out = [];
    EVAL_SERIES.forEach((m) => {
      if (m.axis === "metric" && run.series.eval[m.key]) {
        out.push({ key: m.key, label: m.label, color: m.color, axis: "left", points: run.series.eval[m.key] });
      }
    });
    if (run.series.loss?.train?.length)
      out.push({ key: "train_loss", label: "train loss", color: "var(--p-train)", axis: "right", dashed: true, points: run.series.loss.train });
    if (run.series.loss?.eval?.length)
      out.push({ key: "eval_loss", label: "eval loss", color: "var(--p-eval)", axis: "right", dashed: true, points: run.series.loss.eval });
    return out;
  }, [run]);

  const lossVals = [...(run.series.loss?.train || []), ...(run.series.loss?.eval || [])].map((p) => p[1]);
  const yRight = lossVals.length ? [Math.min(...lossVals) * 0.9, Math.max(...lossVals) * 1.05] : undefined;

  // bottom chart series (projection or probe)
  const bottomSeries = useMemo(() => {
    return run.concepts
      .filter((c) => run.series.trajectory[c.name])
      .map((c) => ({
        key: c.name, label: titleCase(c.name), color: c.color, axis: "left",
        points: run.series.trajectory[c.name].map((e) => [e.step, view === "probe" ? e.probe_prob : e.projection]),
      }));
  }, [run, view]);

  const projVals = bottomSeries.flatMap((s) => s.points.map((p) => p[1]));
  const yBottom = view === "probe"
    ? [0, 1]
    : [Math.min(0, ...projVals) - 1, Math.max(0, ...projVals) + 1];

  return (
    <div className="dualplot">
      {title && (
        <div className="spread" style={{ marginBottom: 2 }}>
          <span className="title-sm" style={{ fontSize: 15 }}>{title}</span>
          <span className="earlystop-note"><span className="em" /> early-stop (min eval loss) · step {run.earlyStop}</span>
        </div>
      )}

      <div className="card chart-card">
        <div className="chart-head">
          <span className="ct">Loss &amp; eval battery</span>
          <Legend series={topSeries} hidden={hidden} onToggle={toggle} />
        </div>
        <Chart height={196} series={topSeries} xDomain={xDomain} yLeft={[0, 1]} yRight={yRight}
          earlyStop={run.earlyStop} reveal={reveal} hover={hover} onHover={setHover} hiddenKeys={hidden}
          formatLeft={(v) => v.toFixed(2)} formatRight={(v) => v.toFixed(2)} yRightLabel="loss" />
      </div>

      <div className="card chart-card">
        <div className="chart-head">
          <span className="ct">Concept-vector {view === "probe" ? "probe  P(trait)" : "projection ⟨h, v̂⟩"}</span>
          <div className="row gap10">
            <div className="toggle">
              <button className={view === "projection" ? "on" : ""} onClick={() => setView("projection")}>projection</button>
              <button className={view === "probe" ? "on" : ""} onClick={() => setView("probe")}>probe</button>
            </div>
          </div>
        </div>
        <Legend series={bottomSeries} hidden={hidden} onToggle={toggle} />
        <Chart height={196} series={bottomSeries} xDomain={xDomain} yLeft={yBottom}
          earlyStop={run.earlyStop} reveal={reveal} hover={hover} onHover={setHover} hiddenKeys={hidden}
          zeroLine={view !== "probe"}
          formatLeft={(v) => (view === "probe" ? v.toFixed(2) : v.toFixed(1))} />
      </div>
    </div>
  );
}
