import { useMemo, useState } from "react";
import Chart from "./Chart.jsx";
import Legend from "./Legend.jsx";
import InfoDot from "./InfoDot.jsx";
import { EVAL_SERIES } from "../api/sampleData.js";
import { titleCase } from "../lib/format.js";

/* Two stacked, x-aligned plots that host BOTH runs in sequence:
     top    = train/eval loss (gray) + averaged Safety (red) / Capability (blue) lines
     bottom = the concept-vector projections (or probe)
   First the normal fine-tune (v1) streams in, staged: loss → evals → projections.
   Then, on "run preventive steering", the v1 loss is dropped, the v1 eval + projection
   lines dim, and the preventive-steered run (v2, `prev_steered`, brighter/paired hues)
   overlays on the SAME two plots — loss → evals on top, then its projections below.
   Each series carries its own `reveal` fraction, driven by the staged `rv` clock. */

const CAP_KEYS = EVAL_SERIES.filter((m) => m.axis === "metric" && m.group === "capability").map((m) => m.key);
const SAFE_KEYS = EVAL_SERIES.filter((m) => m.axis === "metric" && m.group === "safety").map((m) => m.key);

// per-step mean across a set of [step,val] series → a single line (no ±sd band)
function meanLine(arrs) {
  const present = arrs.filter((a) => a && a.length);
  if (!present.length) return [];
  return present[0].map(([st], i) => {
    const vals = present.map((a) => a[i]?.[1]).filter((v) => v != null);
    return [st, vals.reduce((x, y) => x + y, 0) / (vals.length || 1)];
  });
}

// the v2 shade of a concept colour: a slightly lighter, related hue (prev_steered)
const shift = (c) => `color-mix(in srgb, ${c} 72%, white)`;

const PREV = "prev_steered";

export default function DualPlot({
  run, steerRun = null, rv = {}, defaultView = "projection",
  chartHeight = 150, fill = true, showEarlyStop = false,
}) {
  const steering = !!steerRun;
  const earlyStop = showEarlyStop ? run.earlyStop : null;
  const [hover, setHover] = useState(null);
  const [view, setView] = useState(defaultView); // projection | probe (bottom measure)
  const [hidden, setHidden] = useState(new Set());
  const toggle = (k) => setHidden((h) => { const n = new Set(h); n.has(k) ? n.delete(k) : n.add(k); return n; });

  // the concept being suppressed: its v1 line stays bright (not dimmed) so the v1↔v2
  // comparison is direct, and the delta between the two is shaded green + read out.
  const mitConcept = run.steer?.concept || null;
  const fmtProj = (v) => (view === "probe" ? v.toFixed(2) : v.toFixed(1));
  const fmtSigned = (v) => (v >= 0 ? "+" : "-") + fmtProj(Math.abs(v));
  const fmtPct = (p) => (p == null ? "" : (p >= 0 ? "+" : "-") + Math.abs(p).toFixed(1) + "%");

  // staged reveal fractions (default fully-drawn if driven statically)
  const f = (v) => (v == null ? 1 : v);
  const rvLossV1 = f(rv.lossV1), rvEvalV1 = f(rv.evalV1), rvProjV1 = f(rv.projV1);
  const rvLossV2 = f(rv.lossV2), rvEvalV2 = f(rv.evalV2), rvProjV2 = f(rv.projV2);

  // shared x-axis: the true end across every series of both runs
  const lastStep = useMemo(() => {
    let mx = 0;
    const bump = (s) => { const p = s && s[s.length - 1]; if (p) mx = Math.max(mx, Array.isArray(p) ? p[0] : p.step); };
    const scan = (r) => {
      if (!r) return;
      Object.values(r.series.eval || {}).forEach(bump);
      bump(r.series.loss?.train); bump(r.series.loss?.eval);
      Object.values(r.series.trajectory || {}).forEach(bump);
    };
    scan(run); scan(steerRun);
    return mx > 0 ? mx : 250;
  }, [run, steerRun]);
  const xDomain = [0, lastStep];

  // ── top chart: loss (right axis, gray) + averaged Safety/Capability (left axis) ──
  const topSeries = useMemo(() => {
    const out = [];
    const lossOf = (r, key, dashed, rev, label) => (r?.series?.loss?.[key]?.length
      ? { key: `${key}_loss${label.includes(PREV) ? "_v2" : ""}`, label, color: "var(--ft-loss)",
          axis: "right", dashed, noDots: true, reveal: rev, points: r.series.loss[key] }
      : null);
    const capOf = (r) => meanLine(CAP_KEYS.map((k) => r.series.eval[k]));
    const safeOf = (r) => meanLine(SAFE_KEYS.map((k) => r.series.eval[k]));

    if (!steering) {
      // v1: loss first, then evals
      out.push(lossOf(run, "train", false, rvLossV1, "train loss"));
      out.push(lossOf(run, "eval", true, rvLossV1, "eval loss"));
      out.push({ key: "cap_v1", label: "Capability", color: "var(--ft-cap)", axis: "left", reveal: rvEvalV1, points: capOf(run) });
      out.push({ key: "safe_v1", label: "Safety", color: "var(--ft-safe)", axis: "left", reveal: rvEvalV1, points: safeOf(run) });
    } else {
      // v1 loss removed; v1 evals stay but dimmed under the v2 overlay
      out.push({ key: "cap_v1", label: "Capability", color: "var(--ft-cap)", axis: "left", reveal: 1, dim: true, points: capOf(run) });
      out.push({ key: "safe_v1", label: "Safety", color: "var(--ft-safe)", axis: "left", reveal: 1, dim: true, points: safeOf(run) });
      // v2: new TRAIN loss only (no eval loss for the preventive-steered run), then both evals
      out.push(lossOf(steerRun, "train", false, rvLossV2, `train loss · ${PREV}`));
      out.push({ key: "cap_v2", label: `Capability · ${PREV}`, color: "var(--ft-cap-2)", axis: "left", reveal: rvEvalV2, points: capOf(steerRun) });
      out.push({ key: "safe_v2", label: `Safety · ${PREV}`, color: "var(--ft-safe-2)", axis: "left", reveal: rvEvalV2, points: safeOf(steerRun) });
    }
    return out.filter(Boolean);
  }, [run, steerRun, steering, rvLossV1, rvEvalV1, rvLossV2, rvEvalV2]);

  const lossVals = topSeries.filter((s) => s.axis === "right").flatMap((s) => s.points.map((p) => p[1]));
  const yRight = lossVals.length ? [Math.min(...lossVals) * 0.9, Math.max(...lossVals) * 1.05] : undefined;

  // ── bottom chart: concept projections (v1), then the v2 overlay (prev_steered) ──
  const trajLines = (r, v2) => (r?.concepts || [])
    .filter((c) => r.series.trajectory[c.name])
    .map((c) => ({
      key: v2 ? `${c.name}__v2` : c.name,
      label: v2 ? `${titleCase(c.name)} · ${PREV}` : titleCase(c.name),
      color: v2 ? shift(c.color) : c.color,
      axis: "left",
      // keep the mitigated concept's v1 line bright; dim only the other (context) v1 lines
      dim: !v2 && steering && c.name !== mitConcept,
      reveal: v2 ? rvProjV2 : (steering ? 1 : rvProjV1),
      points: r.series.trajectory[c.name].map((e) => [e.step, view === "probe" ? e.probe_prob : e.projection]),
    }));

  // the mitigated concept's v1↔v2 gap: a green delta band + its final change readout
  const delta = useMemo(() => {
    if (!steering || !mitConcept) return null;
    const v1 = run.series.trajectory?.[mitConcept];
    const v2 = steerRun.series.trajectory?.[mitConcept];
    if (!v1?.length || !v2?.length) return null;
    const val = (e) => (view === "probe" ? e.probe_prob : e.projection);
    const band = v1.map((e, i) => {
      const a = val(e), b = v2[i] != null ? val(v2[i]) : a;
      return [e.step, Math.min(a, b), Math.max(a, b)];
    });
    const v1f = val(v1[v1.length - 1]), v2f = val(v2[v2.length - 1]);
    const pct = v1f !== 0 ? ((v2f - v1f) / Math.abs(v1f)) * 100 : null;
    return { band, v1f, v2f, change: v2f - v1f, pct };
  }, [steering, mitConcept, run, steerRun, view]);

  const bottomSeries = useMemo(() => {
    const out = trajLines(run, false);
    if (steering) out.push(...trajLines(steerRun, true));
    if (delta) out.push({ key: "delta_band", label: `Δ ${titleCase(mitConcept)}`, color: "var(--good)",
      axis: "left", band: delta.band, reveal: rvProjV2, fillOpacity: 0.2, bandOnly: true, points: [] });
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run, steerRun, steering, view, rvProjV1, rvProjV2, delta, mitConcept]);

  const bottomVals = bottomSeries.flatMap((s) => s.points.map((p) => p[1]));
  const yBottom = view === "probe"
    ? [0, 1]
    : [Math.min(0, ...bottomVals) - 0.5, Math.max(0, ...bottomVals) + 0.5];

  // the bottom plot "appears" only once its projections begin streaming
  const bottomShown = steering || rvProjV1 > 0;

  return (
    <div className={`dualplot ${fill ? "fill" : ""}`}>
      <div className="card chart-card">
        <div className="chart-head">
          <span className="row gap6">
            <span className="ct">Training loss, capabilities and refusal</span>
            <InfoDot label="What is this loss?">
              <b>Loss</b> is the LoRA fine-tune’s cross-entropy (next-token prediction) loss, plotted on
              the right-hand axis — <em>train</em> is measured on the SFT batches, <em>eval</em> (dashed)
              on a held-out split. It tracks how well the model fits the fine-tuning data, <em>not</em> its
              safety: a clean, falling loss curve can still hide the drift shown below.
            </InfoDot>
          </span>
        </div>
        <Legend series={topSeries} hidden={hidden} onToggle={toggle} />
        <Chart height={chartHeight} series={topSeries} xDomain={xDomain} yLeft={[0, 1]} yRight={yRight}
          earlyStop={earlyStop} hover={hover} onHover={setHover} hiddenKeys={hidden}
          formatLeft={(v) => v.toFixed(2)} formatRight={(v) => v.toFixed(2)}
          xLabel="Training step" yLeftLabel="Score" yRightLabel="Loss" />
      </div>

      <div className="card chart-card bottom-plot" style={{ opacity: bottomShown ? 1 : 0 }}>
        <div className="chart-head">
          <span className="ct">Emergent risks</span>
          <div className="row gap10">
            {delta && (
              <span className="delta-badge" style={{ opacity: rvProjV2 >= 1 ? 1 : 0 }}
                title={`${titleCase(mitConcept)} ${view === "probe" ? "P(trait)" : "projection"}: ${fmtSigned(delta.v1f)} → ${fmtSigned(delta.v2f)}`}>
                <b>{fmtPct(delta.pct)}</b> {titleCase(mitConcept)} ({fmtSigned(delta.v1f)}→{fmtSigned(delta.v2f)})
              </span>
            )}
            <div className="toggle">
              <button className={view === "projection" ? "on" : ""} onClick={() => setView("projection")}>Projection</button>
              <button className={view === "probe" ? "on" : ""} onClick={() => setView("probe")}>Probe</button>
            </div>
          </div>
        </div>
        <Legend series={bottomSeries.filter((s) => !s.bandOnly)} hidden={hidden} onToggle={toggle} />
        <Chart height={chartHeight} series={bottomSeries} xDomain={xDomain} yLeft={yBottom}
          earlyStop={earlyStop} hover={hover} onHover={setHover} hiddenKeys={hidden}
          zeroLine={view !== "probe"}
          xLabel="Training step" yLeftLabel={view === "probe" ? "P(trait)" : "Projection"}
          formatLeft={(v) => (view === "probe" ? v.toFixed(2) : v.toFixed(1))} />
      </div>
    </div>
  );
}
