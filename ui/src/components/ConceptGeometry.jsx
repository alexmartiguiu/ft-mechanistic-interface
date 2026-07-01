import { useMemo } from "react";

const clamp01 = (x) => Math.max(0, Math.min(1, x));
const seg = (p, a, b) => clamp01((p - a) / (b - a));
const smooth = (t) => t * t * (3 - 2 * t);           // smoothstep ease
const lerp = (a, b, t) => a + (b - a) * t;

/* The geometry of automatic concept-vector extraction, built up in one shot.
   As `progress` runs 0 → 1 the card animates the persona-vector method (Chen et
   al. 2025): contrastive activations cluster into trait-absent and trait-present
   groups, their means define the direction v̂ = (μ₊ − μ₋)/‖μ₊ − μ₋‖, every sample
   projects onto that axis as s = ⟨h, v̂⟩, and the tail past the p-threshold is
   what the audit flags. At progress = 1 it is the settled distribution (the old
   static picture); AuditStep drives progress from Hedda's thinking state, so the
   extraction visibly happens while she reasons. */
export default function ConceptGeometry({ color, seed = 1, flagged = false, percentile = 95, progress = 1 }) {
  const W = 232, H = 100;
  const mL = 10, mR = 12, mT = 22, mB = 18;
  const iw = W - mL - mR, ih = H - mT - mB;
  const baseY = mT + ih;
  const thrT = 0.72; // where the dashed p-threshold sits on the axis (schematic)
  const xOf = (t) => mL + t * iw;

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
  const yOf = (h) => baseY - (h / maxH) * ih;
  const barW = (iw / bars.length) * 0.66;
  const thrX = xOf(thrT);

  // contrastive activation clouds: trait-absent (benign, left) and trait-present (right).
  // each dot carries its projection coordinate `t`, so projecting = dropping it to the axis.
  const dots = useMemo(() => {
    const rnd = (k) => { const s = Math.sin((k + seed * 9.73) * 12.9898) * 43758.5453; return s - Math.floor(s); };
    const out = [];
    const make = (n, tc, spread, side) => {
      for (let i = 0; i < n; i++) {
        const r1 = rnd(side * 53 + i * 2 + 1);
        const r2 = rnd(side * 53 + i * 2 + 2);
        out.push({ t: clamp01(tc + (r1 - 0.5) * spread), cy: mT + 4 + r2 * 15, side });
      }
    };
    make(7, 0.34, 0.24, 0);   // trait-absent
    make(6, 0.80, 0.15, 1);   // trait-present
    return out;
  }, [seed]);

  // choreography — each layer keyed to a slice of progress
  const p = clamp01(progress);
  const appear  = smooth(seg(p, 0.00, 0.20));   // clouds fade in
  const project = smooth(seg(p, 0.52, 0.74));   // dots drop onto the axis
  const meanIn  = smooth(seg(p, 0.24, 0.44));
  const meanOut = smooth(seg(p, 0.52, 0.66));
  const vecIn   = smooth(seg(p, 0.34, 0.52));
  const vecOut  = smooth(seg(p, 0.54, 0.66));
  const axis    = smooth(seg(p, 0.30, 0.50));   // v̂ axis fades in
  const grow    = smooth(seg(p, 0.55, 0.88));   // density bars rise
  const thr     = smooth(seg(p, 0.82, 1.00));   // threshold + flagged tail light up

  const dotOp  = appear * (1 - project);
  const meanOp = meanIn * (1 - meanOut);
  const vecOp  = vecIn * (1 - vecOut);
  const muLx = xOf(0.34), muRx = xOf(0.80), muY = mT + 9;
  const targetTailOp = flagged ? 0.9 : 0.55;

  return (
    <svg className="cg" viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="xMidYMid meet"
      role="img" aria-label="extracting the malign concept direction by difference-of-means, then projecting samples and flagging the tail">
      {/* flagged region tint, in step with the threshold */}
      {flagged && thr > 0 && <rect x={thrX} y={mT - 4} width={W - mR - thrX} height={ih + 4} fill={color} opacity={0.07 * thr} />}

      {/* density bars: grow up from the axis as the samples project on */}
      {bars.map((b, i) => {
        const isTail = b.t >= thrT;
        const x = xOf(b.t) - barW / 2;
        const h = (baseY - yOf(b.h)) * grow;
        const op = isTail ? lerp(0.32, targetTailOp, thr) : 0.32;
        return (
          <rect key={i} x={x} y={baseY - h} width={barW} height={h} rx="1"
            fill={isTail ? color : "var(--mute-2)"} opacity={op} />
        );
      })}

      {/* contrastive activations, dropping onto the axis as s = ⟨h, v̂⟩ */}
      {dotOp > 0.01 && dots.map((d, i) => (
        <circle key={i} cx={xOf(d.t)} cy={lerp(d.cy, baseY, project)} r="1.7"
          fill={d.side ? color : "var(--mute-2)"} opacity={dotOp * (d.side ? 0.9 : 0.6)} />
      ))}

      {/* v̂ = μ₊ − μ₋ : an arrow drawn from the trait-absent mean to the trait-present mean */}
      {vecOp > 0.01 && (
        <g opacity={vecOp * 0.85}>
          <line x1={muLx} y1={muY} x2={muRx} y2={muY} stroke={color} strokeWidth="1.2" />
          <path d={`M${muRx - 4} ${muY - 2.6} L${muRx} ${muY} L${muRx - 4} ${muY + 2.6}`}
            fill="none" stroke={color} strokeWidth="1.2" strokeLinejoin="round" strokeLinecap="round" />
        </g>
      )}
      {meanOp > 0.01 && (
        <g opacity={meanOp}>
          <circle cx={muLx} cy={muY} r="2.4" fill="var(--mute-2)" />
          <circle cx={muRx} cy={muY} r="2.4" fill={color} />
          <text x={muLx} y={muY - 5} fontSize="9.72" fill="var(--mute-2)" fontFamily="var(--mono)" textAnchor="middle">μ₋</text>
          <text x={muRx} y={muY - 5} fontSize="9.72" fill={color} fontFamily="var(--mono)" textAnchor="middle">μ₊</text>
        </g>
      )}

      {/* projection axis = the v̂ direction, arrowhead at the trait-present end */}
      <line x1={mL} x2={W - mR} y1={baseY} y2={baseY} stroke="var(--line-3)" strokeWidth="1" opacity={axis} />
      <path d={`M${W - mR - 5} ${baseY - 3.2} L${W - mR} ${baseY} L${W - mR - 5} ${baseY + 3.2}`}
        fill="none" stroke="var(--line-3)" strokeWidth="1" strokeLinejoin="round" strokeLinecap="round" opacity={axis} />
      <text x={mL} y={baseY + 12} fontSize="11.66" fill="var(--mute-2)" fontFamily="var(--mono)" opacity={axis}>−</text>
      <text x={W - mR - 2} y={baseY + 12} fontSize="11.66" fill="var(--mute-2)" fontFamily="var(--mono)" textAnchor="end" opacity={axis}>+ trait</text>

      {/* p-threshold (fades in last, with the flagged tail) */}
      <line x1={thrX} x2={thrX} y1={mT - 4} y2={baseY} stroke="var(--plot-seal)" strokeWidth="1" strokeDasharray="3 2.5" opacity={thr} />
      <text x={thrX + 3} y={mT - 7} fontSize="11.02" fill="var(--plot-seal)" fontFamily="var(--mono)" opacity={thr}>p{percentile}</text>
    </svg>
  );
}
