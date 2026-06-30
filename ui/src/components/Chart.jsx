import { useMemo } from "react";
import { useSize } from "../lib/hooks.js";

/* Reusable line chart (plain SVG, no deps).
   - up to two y-axes (axis: 'left' | 'right')
   - reveal 0..1 grows the lines left-to-right (live fill)
   - earlyStop draws a dashed vertical marker + dims the region to its right
   - hover (controlled) draws a synced crosshair + tooltip; onHover(step|null) reports moves
   Colours come in as CSS var() strings so the palette stays in tokens.css. */
export default function Chart({
  height = 190,
  series = [],
  xDomain,
  yLeft = [0, 1],
  yRight,
  xLabel,
  yLeftLabel,
  yRightLabel,
  earlyStop = null,
  reveal = 1,
  hover = null,
  onHover,
  hiddenKeys,
  zeroLine = false,
  formatLeft = (v) => v.toFixed(2),
  formatRight = (v) => v.toFixed(2),
}) {
  const [ref, size] = useSize();
  const w = size.w;
  // fill the container's height when it has one (flex layout); else fall back to the prop.
  height = size.h > 60 ? size.h : height;
  const mL = 40, mR = yRight ? 44 : 14, mT = 10, mB = xLabel ? 42 : 30;
  const innerW = Math.max(10, w - mL - mR);
  const innerH = Math.max(10, height - mT - mB);
  const [x0, x1] = xDomain;
  const xOf = (x) => mL + ((x - x0) / (x1 - x0 || 1)) * innerW;
  const yOf = (y, dom) => mT + (1 - (y - dom[0]) / (dom[1] - dom[0] || 1)) * innerH;
  const revealX = x0 + reveal * (x1 - x0);

  const vis = series.filter((s) => !(hiddenKeys && hiddenKeys.has(s.key)));

  // build a path, truncated to revealX (interpolating the final partial segment)
  function pathFor(s) {
    const dom = s.axis === "right" ? yRight : yLeft;
    const pts = [];
    for (let i = 0; i < s.points.length; i++) {
      const [px, py] = s.points[i];
      if (px <= revealX + 1e-6) { pts.push([px, py]); continue; }
      const prev = s.points[i - 1];
      if (prev && prev[0] <= revealX) {
        const t = (revealX - prev[0]) / (px - prev[0]);
        pts.push([revealX, prev[1] + t * (py - prev[1])]);
      }
      break;
    }
    return { dom, pts, d: pts.map(([px, py], i) => `${i ? "L" : "M"}${xOf(px).toFixed(1)} ${yOf(py, dom).toFixed(1)}`).join(" ") };
  }

  // a ±sd envelope for an aggregate series: band = [[x, lo, hi], ...], left axis.
  function bandPathFor(s) {
    const dom = yLeft;
    const pts = [];
    for (let i = 0; i < s.band.length; i++) {
      const [px, lo, hi] = s.band[i];
      if (px <= revealX + 1e-6) { pts.push([px, lo, hi]); continue; }
      const prev = s.band[i - 1];
      if (prev && prev[0] <= revealX) {
        const t = (revealX - prev[0]) / (px - prev[0]);
        pts.push([revealX, prev[1] + t * (lo - prev[1]), prev[2] + t * (hi - prev[2])]);
      }
      break;
    }
    if (pts.length < 2) return "";
    const top = pts.map(([px, , hi], i) => `${i ? "L" : "M"}${xOf(px).toFixed(1)} ${yOf(hi, dom).toFixed(1)}`).join(" ");
    const bot = [...pts].reverse().map(([px, lo]) => `L${xOf(px).toFixed(1)} ${yOf(lo, dom).toFixed(1)}`).join(" ");
    return `${top} ${bot} Z`;
  }

  const yTicks = useMemo(() => {
    const [a, b] = yLeft; const n = 4;
    return Array.from({ length: n + 1 }, (_, i) => a + (i / n) * (b - a));
  }, [yLeft]);

  const xTicks = useMemo(() => {
    const [a, b] = xDomain; const n = 5;
    return Array.from({ length: n + 1 }, (_, i) => Math.round(a + (i / n) * (b - a)));
  }, [xDomain]);

  // tooltip rows: value of each visible series at the hovered step
  const tip = useMemo(() => {
    if (hover == null) return null;
    const rows = vis.map((s) => {
      const found = s.points.find((p) => p[0] === hover) || nearest(s.points, hover);
      if (!found) return null;
      const fmt = s.axis === "right" ? formatRight : formatLeft;
      return { key: s.key, label: s.label, color: s.color, dashed: s.dashed, val: fmt(found[1]) };
    }).filter(Boolean);
    return { rows };
  }, [hover, vis, formatLeft, formatRight]);

  const hoverPx = hover != null ? xOf(hover) : null;
  const tipLeft = hoverPx != null ? Math.min(Math.max(hoverPx + 10, mL), w - 150) : 0;
  const xs = uniqueXs(series);

  return (
    <div className="chart" ref={ref}>
      {w > 0 && (
        <svg viewBox={`0 0 ${w} ${height}`} width={w} height={height}
          onMouseMove={(e) => onHover && onHover(stepFromEvent(e, ref.current, xs, xOf))}
          onMouseLeave={() => onHover && onHover(null)}>
          {/* y grid + tick labels */}
          {yTicks.map((t, i) => (
            <g key={i}>
              <line x1={mL} x2={mL + innerW} y1={yOf(t, yLeft)} y2={yOf(t, yLeft)}
                stroke="var(--line)" strokeWidth="1" />
              <text x={mL - 6} y={yOf(t, yLeft) + 3} textAnchor="end" fontSize="9"
                fill="var(--mute-2)" fontFamily="var(--mono)">{formatLeft(t)}</text>
            </g>
          ))}
          {zeroLine && yLeft[0] < 0 && yLeft[1] > 0 && (
            <line x1={mL} x2={mL + innerW} y1={yOf(0, yLeft)} y2={yOf(0, yLeft)}
              stroke="var(--line-3)" strokeWidth="1" />
          )}
          {/* axes */}
          <line x1={mL} x2={mL} y1={mT} y2={mT + innerH} stroke="var(--line-2)" strokeWidth="1" />
          <line x1={mL} x2={mL + innerW} y1={mT + innerH} y2={mT + innerH} stroke="var(--line-2)" strokeWidth="1" />

          {/* x ticks + labels (training step) */}
          {xTicks.map((t, i) => (
            <g key={"xt" + i}>
              <line x1={xOf(t)} x2={xOf(t)} y1={mT + innerH} y2={mT + innerH + 4} stroke="var(--line-2)" strokeWidth="1" />
              <text x={xOf(t)} y={mT + innerH + 14} textAnchor="middle" fontSize="9"
                fill="var(--mute-2)" fontFamily="var(--mono)">{t}</text>
            </g>
          ))}
          {xLabel && (
            <text x={mL + innerW / 2} y={height - 3} textAnchor="middle" fontSize="9" fill="var(--mute-2)">{xLabel}</text>
          )}
          {yLeftLabel && (
            <text x={11} y={mT + innerH / 2} textAnchor="middle" fontSize="9" fill="var(--mute-2)"
              transform={`rotate(-90 11 ${mT + innerH / 2})`}>{yLeftLabel}</text>
          )}

          {/* early-stop: dim the overfit/drift region + dashed marker, labelled so the line reads */}
          {earlyStop != null && (
            <>
              <rect x={xOf(earlyStop)} y={mT} width={Math.max(0, mL + innerW - xOf(earlyStop))} height={innerH}
                fill="var(--ink)" opacity="0.045" />
              <line x1={xOf(earlyStop)} x2={xOf(earlyStop)} y1={mT} y2={mT + innerH}
                stroke="var(--plot-seal)" strokeWidth="1.2" strokeDasharray="4 3" />
              <text x={xOf(earlyStop) + 4} y={mT + 8} fontSize="8.5" fill="var(--plot-seal)"
                fontFamily="var(--mono)">early stop</text>
            </>
          )}

          {/* ±sd envelopes (drawn behind the lines) */}
          {vis.filter((s) => s.band).map((s) => (
            <path key={"band-" + s.key} d={bandPathFor(s)} style={{ fill: s.color }}
              fillOpacity="0.13" stroke="none" />
          ))}

          {/* series */}
          {vis.map((s) => {
            const { dom, pts, d } = pathFor(s);
            return (
              <g key={s.key}>
                <path d={d} fill="none" style={{ stroke: s.color }} strokeWidth={s.dashed ? 1.3 : 1.6}
                  strokeDasharray={s.dashed ? "5 3" : "none"} strokeLinejoin="round" strokeLinecap="round" />
                {!s.dashed && pts.map(([px, py], i) => (
                  <circle key={i} cx={xOf(px)} cy={yOf(py, dom)} r="2.4" fill="var(--bg)"
                    style={{ stroke: s.color }} strokeWidth="1.3"
                    opacity={earlyStop != null && px > earlyStop ? 0.55 : 1} />
                ))}
              </g>
            );
          })}

          {/* crosshair */}
          {hoverPx != null && (
            <>
              <line x1={hoverPx} x2={hoverPx} y1={mT} y2={mT + innerH} stroke="var(--line-3)" strokeWidth="1" />
              {vis.map((s) => {
                const found = s.points.find((p) => p[0] === hover);
                if (!found || found[0] > revealX) return null;
                const dom = s.axis === "right" ? yRight : yLeft;
                return <circle key={s.key} cx={xOf(found[0])} cy={yOf(found[1], dom)} r="3.4"
                  fill="var(--bg)" style={{ stroke: s.color }} strokeWidth="2" />;
              })}
            </>
          )}

          {/* right axis label */}
          {yRight && yRightLabel && (
            <text x={w - 4} y={mT + innerH / 2} textAnchor="middle" fontSize="9" fill="var(--mute-2)"
              transform={`rotate(90 ${w - 4} ${mT + innerH / 2})`}>{yRightLabel}</text>
          )}
        </svg>
      )}

      {tip && hover != null && (
        <div className="tip" style={{ left: tipLeft }}>
          <div className="tstep">step {hover}</div>
          {tip.rows.map((r) => (
            <div className="trow" key={r.key}>
              <span className="nm"><span className="swt" style={{ background: r.color, opacity: r.dashed ? 0.6 : 1 }} />{r.label}</span>
              <span className="vl">{r.val}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function nearest(points, x) {
  if (!points.length) return null;
  return points.reduce((a, b) => (Math.abs(b[0] - x) < Math.abs(a[0] - x) ? b : a));
}
function uniqueXs(series) {
  const s = new Set();
  series.forEach((ser) => ser.points.forEach((p) => s.add(p[0])));
  return [...s].sort((a, b) => a - b);
}
function stepFromEvent(e, el, xs, xOf) {
  if (!el || !xs.length) return null;
  const rect = el.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  return xs.reduce((a, b) => (Math.abs(xOf(b) - mx) < Math.abs(xOf(a) - mx) ? b : a));
}
