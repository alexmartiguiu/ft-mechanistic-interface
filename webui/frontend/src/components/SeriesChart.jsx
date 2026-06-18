// Client-side line chart (replaces the matplotlib SVG). Draws multiple series on a
// shared x (training step), an optional second (right) axis for loss, an early-stop
// marker, and a legend whose chips double as show/hide filters.
const W = 720, PADT = 18, PADB = 50;

function niceTicks(lo, hi, n = 5) {
  const out = [];
  for (let i = 0; i < n; i++) out.push(lo + ((hi - lo) * i) / (n - 1));
  return out;
}

const SEAL = "#9c4a3c";   // vermilion — the early-stop mark (mirrors plots.SEAL)

export default function SeriesChart({
  series, visible, onToggle, leftDomain = "auto", baselineZero = false,
  yLeftLabel = "", yRightLabel = "", unit = false, height = 250, earlyStop = null,
  showLegend = true, legendInteractive = true, xLabel = "training step",
}) {
  const padL = 46;
  const vis = series.filter((s) => visible.has(s.key) && s.points.length);
  const hasRight = vis.some((s) => s.axis === "R");
  const padR = hasRight ? 48 : 18;
  const H = height;

  const allX = vis.flatMap((s) => s.points.map((p) => p[0]));
  if (!allX.length) {
    return (
      <div>
        <div style={{ height: H, display: "flex", alignItems: "center", justifyContent: "center", fontSize: "12.5px", color: "#a6aebe" }}>No data for the selected series.</div>
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

  return (
    <div style={{ display: "flex", gap: "24px", alignItems: "center" }}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block", overflow: "visible", flex: 1, minWidth: 0 }}>
        {/* early-stop: faint dim over the overfitting region (behind grid + lines) */}
        {esX != null && <rect x={esX} y={PADT} width={Math.max(0, W - padR - esX)} height={H - PADT - PADB} fill="#1c1c1a" opacity="0.06" />}

        {/* y grid + labels (left) */}
        {yTicks.map((t, i) => (
          <g key={i}>
            <line x1={padL} y1={t.y} x2={W - padR} y2={t.y} stroke="#eef2f9" strokeWidth="1" />
            <text x={padL - 8} y={t.y + 3.5} textAnchor="end" fontSize="11" fill="#a6aebe" fontFamily="'JetBrains Mono',monospace">{t.label}</text>
          </g>
        ))}
        {baselineZero && lLo < 0 && lHi > 0 && <line x1={padL} y1={yL(0)} x2={W - padR} y2={yL(0)} stroke="#cdd6e6" strokeWidth="1" />}

        {/* x labels + axis title */}
        {xTicks.map((t, i) => (
          <text key={i} x={t.x} y={H - PADB + 18} textAnchor="middle" fontSize="11" fill="#a6aebe" fontFamily="'JetBrains Mono',monospace">{t.label}</text>
        ))}
        {xLabel && <text x={padL + (W - padL - padR) / 2} y={H - 8} textAnchor="middle" fontSize="11" fill="#98a2b3">{xLabel}</text>}

        {/* axis titles */}
        {yLeftLabel && <text x={padL - 30} y={PADT - 5} textAnchor="start" fontSize="10.5" fill="#98a2b3">{yLeftLabel}</text>}
        {hasRight && yRightLabel && <text x={W - padR + 8} y={PADT - 5} textAnchor="end" fontSize="10.5" fill="#98a2b3">{yRightLabel}</text>}

        {/* lines + dots */}
        {leftS.map((s) => (
          <g key={s.key}>
            <polyline points={path(s, yL)} fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" strokeDasharray={s.dashed ? "4 3" : "0"} />
            {s.points.map((p, i) => <circle key={i} cx={xScale(p[0])} cy={yL(p[1])} r="2" fill="#fff" stroke={s.color} strokeWidth="1.3" />)}
          </g>
        ))}
        {rightS.map((s) => (
          <polyline key={s.key} points={path(s, yR)} fill="none" stroke={s.color} strokeWidth="1.6" strokeDasharray="4 3" strokeLinejoin="round" strokeLinecap="round" />
        ))}

        {/* early-stop: dashed vermilion line at min eval-loss step (above the data) */}
        {esX != null && (
          <g>
            <line x1={esX} y1={PADT} x2={esX} y2={H - PADB} stroke={SEAL} strokeWidth="1.1" strokeDasharray="4 3" />
            <text x={esX} y={PADT - 4} textAnchor="middle" fontSize="9.5" fill={SEAL} fontFamily="'JetBrains Mono',monospace">early stop</text>
          </g>
        )}
      </svg>
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
              <span style={{ flex: "none", width: "15px", height: "15px", borderRadius: "4px", border: "1.5px solid " + (on ? s.color : "#cdd6e6"), background: on ? s.color : "#fff", display: "flex", alignItems: "center", justifyContent: "center", transition: "background .12s, border-color .12s" }}>
                {on && (
                  <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                )}
              </span>
            )}
            <span style={{ flex: "none", width: "14px", height: 0, borderTop: (s.dashed ? "2px dashed " : "2px solid ") + (on ? s.color : "#cfd8e8") }} />
            <span style={{ fontSize: "12.5px", color: on ? "#48546e" : "#b7c1d6", fontFamily: s.dashed ? "'JetBrains Mono',monospace" : undefined }}>{s.label}</span>
          </Tag>
        );
      })}
    </div>
  );
}
