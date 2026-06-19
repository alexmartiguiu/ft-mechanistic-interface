import { useState, useRef } from "react";
import { PLOT, SEAL, INK, CARD } from "../styles/tokens.js";

// Client-side line chart (replaces the matplotlib SVG). Draws multiple series on a
// shared x (training step), an optional second (right) axis for loss, an early-stop
// marker, and a legend whose chips double as show/hide filters. Hovering reads off
// each series' value at the nearest step (crosshair + tooltip). All colours come from
// the shared LUCENT plot palette (tokens.js) so a line means the same everywhere.
const W = 720, PADT = 18, PADB = 50;
const MONO = "'JetBrains Mono',monospace";

function niceTicks(lo, hi, n = 5) {
  const out = [];
  for (let i = 0; i < n; i++) out.push(lo + ((hi - lo) * i) / (n - 1));
  return out;
}

export default function SeriesChart({
  series, visible, onToggle, leftDomain = "auto", baselineZero = false,
  yLeftLabel = "", yRightLabel = "", unit = false, height = 250, earlyStop = null,
  showLegend = true, legendInteractive = true, xLabel = "training step",
}) {
  const [hover, setHover] = useState(null);   // hovered step value (null = none)
  const svgRef = useRef(null);
  const padL = 46;
  const vis = series.filter((s) => visible.has(s.key) && s.points.length);
  const hasRight = vis.some((s) => s.axis === "R");
  const padR = hasRight ? 48 : 18;
  const H = height;

  const allX = vis.flatMap((s) => s.points.map((p) => p[0]));
  if (!allX.length) {
    return (
      <div>
        <div style={{ height: H, display: "flex", alignItems: "center", justifyContent: "center", fontSize: "12.5px", color: "var(--mute-3)" }}>No data for the selected series.</div>
        {showLegend && <Legend series={series} visible={visible} onToggle={onToggle} interactive={legendInteractive} />}
      </div>
    );
  }
  let xmin = Math.min(...allX), xmax = Math.max(...allX);
  if (xmin === xmax) { xmin -= 1; xmax += 1; }

  const leftS = vis.filter((s) => s.axis !== "R");
  const rightS = vis.filter((s) => s.axis === "R");
  const leftY = leftS.flatMap((s) => s.points.map((p) => p[1]));
  const rightY = rightS.flatMap((s) => s.points.map((p) => p[1]));

  let lLo, lHi;
  if (leftDomain === "unit") { lLo = 0; lHi = 1; }
  else {
    lLo = Math.min(...(baselineZero ? [0, ...leftY] : leftY));
    lHi = Math.max(...(baselineZero ? [0, ...leftY] : leftY));
    const pad = (lHi - lLo) * 0.1 || 0.5;
    lLo -= pad; lHi += pad;
  }
  let rLo = 0, rHi = 1;
  if (rightY.length) { rLo = Math.min(...rightY); rHi = Math.max(...rightY); const p = (rHi - rLo) * 0.12 || 0.1; rLo -= p; rHi += p; }

  const xScale = (x) => padL + ((x - xmin) / (xmax - xmin)) * (W - padL - padR);
  const yL = (v) => PADT + (1 - (v - lLo) / (lHi - lLo)) * (H - PADT - PADB);
  const yR = (v) => PADT + (1 - (v - rLo) / (rHi - rLo)) * (H - PADT - PADB);

  const yTicks = niceTicks(lLo, lHi).map((v) => ({ y: yL(v), label: unit ? Math.round(v * 100) + "%" : v.toFixed(2) }));
  const uniqX = [...new Set(allX)].sort((a, b) => a - b);
  const xTicks = (uniqX.length <= 8 ? uniqX : niceTicks(xmin, xmax, 6).map((v) => Math.round(v)))
    .map((s) => ({ x: xScale(s), label: s === 0 ? "base" : String(s) }));

  const path = (s, scale) => s.points.slice().sort((a, b) => a[0] - b[0]).map((p) => xScale(p[0]).toFixed(1) + "," + scale(p[1]).toFixed(1)).join(" ");

  // early-stop mark (min validation loss): shade the post-peak overfitting region and
  // drop a vermilion dashed vline — mirrors plots._mark_early_stop on both charts.
  const esX = earlyStop != null && earlyStop >= xmin && earlyStop <= xmax ? xScale(earlyStop) : null;

  // ── hover: snap pointer to the nearest step, read each series' value there ──────────
  const valueAt = (s, step) => {
    let best = null, bd = Infinity;
    for (const p of s.points) { const d = Math.abs(p[0] - step); if (d < bd) { bd = d; best = p; } }
    return best ? best[1] : null;
  };
  const fmt = (s, v) => (v == null ? "—" : unit && s.axis !== "R" ? Math.round(v * 100) + "%" : v.toFixed(3));
  const onMove = (e) => {
    const el = svgRef.current; if (!el) return;
    const rect = el.getBoundingClientRect();
    const xpix = ((e.clientX - rect.left) / rect.width) * W;          // client → viewBox x
    const dx = xmin + ((xpix - padL) / (W - padL - padR)) * (xmax - xmin);
    let best = uniqX[0], bd = Infinity;
    for (const sx of uniqX) { const d = Math.abs(sx - dx); if (d < bd) { bd = d; best = sx; } }
    setHover(best);
  };
  const hoverX = hover != null ? xScale(hover) : null;

  return (
    <div style={{ display: "flex", gap: "24px", alignItems: "center" }}>
      <div style={{ position: "relative", flex: 1, minWidth: 0 }}>
      <svg ref={svgRef} onMouseMove={onMove} onMouseLeave={() => setHover(null)} viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block", overflow: "visible" }}>
        {/* early-stop: faint dim over the overfitting region (behind grid + lines) */}
        {esX != null && <rect x={esX} y={PADT} width={Math.max(0, W - padR - esX)} height={H - PADT - PADB} fill={INK} opacity="0.06" />}

        {/* y grid + labels (left) */}
        {yTicks.map((t, i) => (
          <g key={i}>
            <line x1={padL} y1={t.y} x2={W - padR} y2={t.y} stroke={PLOT.grid} strokeWidth="1" />
            <text x={padL - 8} y={t.y + 3.5} textAnchor="end" fontSize="11" fill={PLOT.mute} fontFamily={MONO}>{t.label}</text>
          </g>
        ))}
        {baselineZero && lLo < 0 && lHi > 0 && <line x1={padL} y1={yL(0)} x2={W - padR} y2={yL(0)} stroke={PLOT.hair} strokeWidth="1" />}

        {/* x labels + axis title */}
        {xTicks.map((t, i) => (
          <text key={i} x={t.x} y={H - PADB + 18} textAnchor="middle" fontSize="11" fill={PLOT.mute} fontFamily={MONO}>{t.label}</text>
        ))}
        {xLabel && <text x={padL + (W - padL - padR) / 2} y={H - 8} textAnchor="middle" fontSize="11" fill={PLOT.mute}>{xLabel}</text>}

        {/* axis titles */}
        {yLeftLabel && <text x={padL - 30} y={PADT - 5} textAnchor="start" fontSize="10.5" fill={PLOT.mute}>{yLeftLabel}</text>}
        {hasRight && yRightLabel && <text x={W - padR + 8} y={PADT - 5} textAnchor="end" fontSize="10.5" fill={PLOT.mute}>{yRightLabel}</text>}

        {/* lines + dots */}
        {leftS.map((s) => (
          <g key={s.key}>
            <polyline points={path(s, yL)} fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" strokeDasharray={s.dashed ? "4 3" : "0"} />
            {s.points.map((p, i) => <circle key={i} cx={xScale(p[0])} cy={yL(p[1])} r="2" fill={CARD} stroke={s.color} strokeWidth="1.3" />)}
          </g>
        ))}
        {rightS.map((s) => (
          <polyline key={s.key} points={path(s, yR)} fill="none" stroke={s.color} strokeWidth="1.6" strokeDasharray="4 3" strokeLinejoin="round" strokeLinecap="round" />
        ))}

        {/* early-stop: dashed vermilion line at min eval-loss step (above the data) */}
        {esX != null && (
          <g>
            <line x1={esX} y1={PADT} x2={esX} y2={H - PADB} stroke={SEAL} strokeWidth="1.1" strokeDasharray="4 3" />
            <text x={esX} y={PADT - 4} textAnchor="middle" fontSize="9.5" fill={SEAL} fontFamily={MONO}>early stop</text>
          </g>
        )}

        {/* hover crosshair + emphasised markers at the snapped step */}
        {hoverX != null && (
          <g pointerEvents="none">
            <line x1={hoverX} y1={PADT} x2={hoverX} y2={H - PADB} stroke={PLOT.hair} strokeWidth="1" strokeDasharray="3 3" />
            {vis.map((s) => {
              const v = valueAt(s, hover); if (v == null) return null;
              const yy = (s.axis === "R" ? yR : yL)(v);
              return <circle key={s.key} cx={hoverX} cy={yy} r="3.5" fill={s.color} stroke={CARD} strokeWidth="1.5" />;
            })}
          </g>
        )}
      </svg>

      {/* value tooltip — reads each visible series at the hovered step */}
      {hover != null && (
        <div style={{ position: "absolute", top: "4px", left: `${(hoverX / W) * 100}%`, transform: "translateX(-50%)", pointerEvents: "none", background: "var(--card)", border: "1px solid var(--line)", borderRadius: "8px", boxShadow: "0 2px 12px rgba(28,28,26,0.10)", padding: "7px 9px", fontSize: "11px", fontFamily: MONO, whiteSpace: "nowrap", zIndex: 2 }}>
          <div style={{ color: "var(--mute-2)", marginBottom: "5px" }}>{hover === 0 ? "base" : `step ${hover}`}</div>
          {vis.map((s) => (
            <div key={s.key} style={{ display: "flex", alignItems: "center", gap: "7px", lineHeight: 1.7 }}>
              <span style={{ flex: "none", width: "9px", height: "2px", background: s.color, borderRadius: "1px" }} />
              <span style={{ flex: 1, color: "var(--mute)" }}>{s.label}</span>
              <span style={{ color: "var(--ink)", marginLeft: "8px" }}>{fmt(s, valueAt(s, hover))}</span>
            </div>
          ))}
        </div>
      )}
      </div>
      {showLegend && <Legend series={series} visible={visible} onToggle={onToggle} interactive={legendInteractive} />}
    </div>
  );
}

// Vertical legend docked to the right of the plot (mirrors the drift-explorer panel).
// In the inline preview (`interactive=false`) it's a read-only key — color swatch + label,
// no checkboxes. In the expanded view (`interactive=true`) each row gains a checkbox and
// doubles as the show/hide filter; toggles flow back to the shared `visible` set.
function Legend({ series, visible, onToggle, interactive = true }) {
  return (
    <div style={{ flex: "none", display: "flex", flexDirection: "column", gap: "10px", maxWidth: "190px" }}>
      {series.map((s) => {
        const on = visible.has(s.key);
        const Tag = interactive ? "button" : "div";
        return (
          <Tag
            key={s.key}
            onClick={interactive ? () => onToggle(s.key) : undefined}
            title={interactive ? (on ? "Click to hide" : "Click to show") : undefined}
            style={{
              display: "flex", alignItems: "center", gap: "9px", padding: 0, textAlign: "left",
              border: "none", background: "none", font: "inherit", width: "fit-content",
              cursor: interactive ? "pointer" : "default",
            }}
          >
            {interactive && (
              <span style={{ flex: "none", width: "15px", height: "15px", borderRadius: "4px", border: "1.5px solid " + (on ? s.color : "var(--line-2)"), background: on ? s.color : "var(--card)", display: "flex", alignItems: "center", justifyContent: "center", transition: "background .12s, border-color .12s" }}>
                {on && (
                  <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke={CARD} strokeWidth="4" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                )}
              </span>
            )}
            <span style={{ flex: "none", width: "14px", height: 0, borderTop: (s.dashed ? "2px dashed " : "2px solid ") + (on ? s.color : "var(--line-2)") }} />
            <span style={{ fontSize: "12.5px", color: on ? "var(--ink-soft)" : "var(--mute-3)", fontFamily: s.dashed ? MONO : undefined }}>{s.label}</span>
          </Tag>
        );
      })}
    </div>
  );
}
