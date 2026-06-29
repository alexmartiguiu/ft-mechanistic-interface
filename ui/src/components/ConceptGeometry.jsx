import { useMemo } from "react";

/* A compact, faithful picture of how one risky concept is scored.
   Each activation h is projected onto the concept's persona direction v̂ (built
   difference-of-means over contrastive prompts, per Chen et al., persona vectors);
   the score s = ⟨h, v̂⟩ spreads the samples along the axis below, and the right
   tail past the p-threshold is what the audit flags. The bump on the right is the
   emergent trait-present mass — the thing a narrow fine-tune amplifies.
   Schematic, not the literal histogram: it teaches the geometry at a glance. */
export default function ConceptGeometry({ color, seed = 1, flagged = false, percentile = 95 }) {
  const W = 232, H = 100;
  const mL = 10, mR = 12, mT = 22, mB = 18;
  const iw = W - mL - mR, ih = H - mT - mB;
  const baseY = mT + ih;
  const thrT = 0.72; // where the dashed p-threshold sits on the axis (schematic)

  const bars = useMemo(() => {
    const n = 38;
    return Array.from({ length: n }, (_, i) => {
      const t = i / (n - 1);
      const main = Math.exp(-Math.pow((t - 0.40) / 0.19, 2));          // benign mass, left
      const tail = 0.42 * Math.exp(-Math.pow((t - 0.86) / 0.085, 2));  // emergent trait, right
      const wob = 0.05 * Math.pow(Math.sin(t * 27 + seed * 1.7), 2);
      return { t, h: Math.max(0.015, main + tail + wob) };
    });
  }, [seed]);

  const maxH = Math.max(...bars.map((b) => b.h));
  const xOf = (t) => mL + t * iw;
  const yOf = (h) => baseY - (h / maxH) * ih;
  const barW = (iw / bars.length) * 0.66;
  const thrX = xOf(thrT);

  return (
    <svg className="cg" viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="xMidYMid meet"
      role="img" aria-label="projection of samples onto the concept direction, with the flagged tail">
      {/* flagged region tint */}
      {flagged && <rect x={thrX} y={mT - 4} width={W - mR - thrX} height={ih + 4} fill={color} opacity="0.07" />}

      {/* density bars */}
      {bars.map((b, i) => {
        const isTail = b.t >= thrT;
        const x = xOf(b.t) - barW / 2;
        const y = yOf(b.h);
        return (
          <rect key={i} x={x} y={y} width={barW} height={baseY - y} rx="1"
            fill={isTail ? color : "var(--mute-2)"}
            opacity={isTail ? (flagged ? 0.9 : 0.55) : 0.32} />
        );
      })}

      {/* projection axis = the v̂ direction, arrowhead at the trait-present end */}
      <line x1={mL} x2={W - mR} y1={baseY} y2={baseY} stroke="var(--line-3)" strokeWidth="1" />
      <path d={`M${W - mR - 5} ${baseY - 3.2} L${W - mR} ${baseY} L${W - mR - 5} ${baseY + 3.2}`}
        fill="none" stroke="var(--line-3)" strokeWidth="1" strokeLinejoin="round" strokeLinecap="round" />
      <text x={mL} y={baseY + 12} fontSize="9" fill="var(--mute-2)" fontFamily="var(--mono)">−</text>
      <text x={W - mR - 2} y={baseY + 12} fontSize="9" fill="var(--mute-2)" fontFamily="var(--mono)" textAnchor="end">+ trait</text>

      {/* p-threshold */}
      <line x1={thrX} x2={thrX} y1={mT - 4} y2={baseY} stroke="var(--plot-seal)" strokeWidth="1" strokeDasharray="3 2.5" />
      <text x={thrX + 3} y={mT - 7} fontSize="8.5" fill="var(--plot-seal)" fontFamily="var(--mono)">p{percentile}</text>
    </svg>
  );
}
