import { useEffect, useMemo, useRef, useState } from "react";
import { titleCase } from "../lib/format.js";

/* Grow the bars in on mount (0 → 1), once. */
function useGrow(ms = 720) {
  const [p, setP] = useState(0);
  const raf = useRef(0);
  useEffect(() => {
    let start = 0, cur = 0;
    const tick = (t) => {
      if (!start) start = t;
      cur = Math.min(1, (t - start) / ms);
      setP(cur);
      if (cur < 1) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [ms]);
  return p;
}

/* The REAL per-sample projection distribution for ONE concept, from point_projections.json
   (scripts/recompute_point_projections.py). Draws the true histogram (bin_edges/counts) on
   the projection axis s = <h, v̂>, the real p-threshold, the flagged tail past it, and the
   median — no assumed shape, every bar a real sample count. Returns null when there is no
   recomputed distribution (caller falls back to the schematic build-up). */
export default function ConceptDistPlot({ name, color, dist, percentile = 95, nFlagged }) {
  const grow = useGrow();
  const W = 300, H = 138;
  const mL = 8, mR = 10, mT = 18, mB = 30;
  const iw = W - mL - mR, ih = H - mT - mB;
  const baseY = mT + ih;

  const geo = useMemo(() => {
    const edges = dist?.binEdges || [];
    const counts = dist?.counts || [];
    if (edges.length < 2 || counts.length < 1) return null;
    const lo = edges[0], hi = edges[edges.length - 1], span = (hi - lo) || 1;
    const xOf = (v) => mL + ((v - lo) / span) * iw;
    const maxC = Math.max(...counts, 1);
    const thr = dist.threshold;
    const bars = counts.map((c, i) => {
      const x0 = edges[i], x1 = edges[i + 1], mid = (x0 + x1) / 2;
      return { x: xOf(x0), w: Math.max(1, xOf(x1) - xOf(x0) - 0.8),
               h: (c / maxC) * ih, flagged: thr != null && mid > thr };
    });
    return {
      bars,
      thrX: thr != null ? xOf(thr) : null,
      medianX: dist.median != null ? xOf(dist.median) : null,
      loLab: lo, hiLab: hi,
    };
  }, [dist]);

  if (!geo) return null;

  return (
    <div className="cdp card">
      <div className="cdp-head">
        <span className="dot" style={{ background: color }} />
        <span className="cdp-name">{titleCase(name)}</span>
        {nFlagged != null && <span className="cdp-flagged mono">{nFlagged} flagged</span>}
      </div>
      <svg className="cdp-svg" viewBox={`0 0 ${W} ${H}`} width="100%" height={H}
        preserveAspectRatio="xMidYMid meet" role="img"
        aria-label={`real projection distribution for ${name}; tail past p${percentile} flagged`}>
        {/* flagged region tint, right of the threshold */}
        {geo.thrX != null && (
          <rect x={geo.thrX} y={mT - 2} width={W - mR - geo.thrX} height={ih + 2}
            fill={color} opacity={0.07} />
        )}
        {/* the real histogram: benign mass muted, flagged tail in the concept colour */}
        {geo.bars.map((b, i) => {
          const h = b.h * grow;
          return (
            <rect key={i} x={b.x} y={baseY - h} width={b.w} height={h} rx="0.6"
              fill={b.flagged ? color : "var(--mute-2)"} opacity={b.flagged ? 0.85 : 0.42} />
          );
        })}
        {/* projection axis s = <h, v̂> */}
        <line x1={mL} x2={W - mR} y1={baseY} y2={baseY} stroke="var(--line-3)" strokeWidth="1" />
        {/* median tick */}
        {geo.medianX != null && (
          <line x1={geo.medianX} x2={geo.medianX} y1={baseY} y2={baseY + 4}
            stroke="var(--mute-2)" strokeWidth="1" />
        )}
        {/* real p-threshold (fades in with the grow) */}
        {geo.thrX != null && (
          <g opacity={grow}>
            <line x1={geo.thrX} x2={geo.thrX} y1={mT - 2} y2={baseY}
              stroke="var(--plot-seal)" strokeWidth="1" strokeDasharray="3 2.5" />
            <text x={geo.thrX + 3} y={mT + 3} fontSize="10.2" fill="var(--plot-seal)"
              fontFamily="var(--mono)">p{percentile}</text>
          </g>
        )}
        {/* axis labels */}
        <text x={mL} y={baseY + 13} fontSize="9.6" fill="var(--mute-2)" fontFamily="var(--mono)">
          s = ⟨h, v̂⟩
        </text>
        <text x={W - mR} y={baseY + 13} fontSize="9.6" fill="var(--mute-2)" fontFamily="var(--mono)"
          textAnchor="end">+ trait →</text>
      </svg>
    </div>
  );
}
