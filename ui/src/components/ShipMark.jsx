// Minimal line-art ship — the motif for the "Keel" step (ship a checkpoint).
// Hairline strokes inherit `currentColor`; the masthead pennant carries the
// single green accent, kept on-theme with the rest of the design system.
export default function ShipMark({ size = 50 }) {
  const w = Math.round((size * 80) / 60);
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={w}
      height={size}
      viewBox="0 0 80 60"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-label="ship"
      role="img"
      style={{ display: "block", flex: "none" }}
    >
      {/* mast */}
      <path d="M40 9V41" />
      {/* mainsail — billows to the right of the mast */}
      <path d="M43 13c11 5 11 18 0 24" />
      <path d="M43 13v24" />
      {/* jib — small foresail, left of the mast */}
      <path d="M37 18 27 37h10" />
      {/* deck line + hull */}
      <path d="M16 41h48" />
      <path d="M20 41c5 9 35 9 40 0" />
      {/* waterline ripples */}
      <path d="M8 51c4-3 7 3 11 0s7 3 11 0 7 3 11 0 7 3 11 0 7 3 11 0" opacity="0.5" />
      {/* masthead pennant — the single green accent */}
      <path d="M40 8l12 3-12 3z" fill="var(--good)" stroke="none" />
    </svg>
  );
}
