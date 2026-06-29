import Delta from "./Delta.jsx";
import { EVAL_SERIES } from "../api/sampleData.js";
import { titleCase } from "../lib/format.js";

// value of a [step,val] series at the latest step <= `step` (follows the live fill)
function valueAt(series, step) {
  let v = series.length ? series[0][1] : null;
  for (const [s, y] of series) { if (s <= step + 1e-6) v = y; else break; }
  return v;
}

/* The right half of the Insights stage BEFORE a mitigation runs: a compact live
   scoreboard that tracks alongside the filling plots (eval battery + concept probes).
   When the preventive-steer arm runs, this is replaced by the steered double-plot. */
export default function RunReadout({ run, reveal }) {
  const lastStep = (Object.values(run.series.eval)[0] || [[0, 0]]).slice(-1)[0][0];
  const step = Math.round(reveal * lastStep);
  const metrics = EVAL_SERIES.filter((m) => m.axis === "metric" && run.series.eval[m.key]);

  return (
    <div className="card readout">
      <div className="chart-head">
        <span className="ct">Live readout</span>
        <span className="mono muted" style={{ fontSize: 11.5 }}>step {step} / {lastStep}</span>
      </div>

      <div className="ro-sec">
        <div className="ro-h">Eval battery</div>
        {metrics.map((m) => {
          const s = run.series.eval[m.key];
          const base = s[0][1];
          const cur = valueAt(s, step);
          return (
            <div className="ro-row" key={m.key}>
              <span>{m.label}</span>
              <span className="row gap10">
                <span className="mono">{cur.toFixed(2)}</span>
                <Delta value={cur - base} goodWhen={m.goodWhen} as="pct" />
              </span>
            </div>
          );
        })}
      </div>

      <div className="ro-sec">
        <div className="ro-h">Concept axes · P(trait)</div>
        {run.concepts.filter((c) => run.series.trajectory[c.name]).map((c) => {
          const probe = run.series.trajectory[c.name].map((e) => [e.step, e.probe_prob]);
          const base = probe[0][1];
          const cur = valueAt(probe, step);
          const drifting = cur - base > 0.05;
          return (
            <div className="ro-concept" key={c.name}>
              <div className="spread">
                <span className="row gap6"><span className="dot" style={{ background: c.color, width: 8, height: 8, borderRadius: 999 }} />{titleCase(c.name)}</span>
                <span className="row gap10">
                  <span className="mono" style={{ color: drifting ? "var(--bad)" : "var(--ink-2)" }}>{cur.toFixed(2)}</span>
                  {drifting && <span className="chip bad" style={{ padding: "1px 7px", fontSize: 10.5 }}>drifting</span>}
                </span>
              </div>
              <div className="ro-bar"><div className="ro-fill" style={{ width: `${Math.min(100, cur * 100)}%`, background: c.color }} /></div>
            </div>
          );
        })}
      </div>

      <div className="ro-note">Internal representation, not behaviour yet. Pair with the eval battery; on a steered arm read the probe, not the raw projection.</div>
    </div>
  );
}
