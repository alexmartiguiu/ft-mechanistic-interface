// Minimal line-art compass — the Nauteus motif for the checkout header.
// Hairline strokes inherit `currentColor`; the north needle carries the single
// green accent, kept on-theme with the rest of the design system.
export default function CompassMark({ size = 50 }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="0 0 60 60"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-label="compass"
      role="img"
      style={{ display: "block", flex: "none" }}
    >
      {/* outer ring */}
      <circle cx="30" cy="30" r="22" />
      {/* N / E / S / W ticks */}
      <path d="M30 9v5" />
      <path d="M30 46v5" />
      <path d="M9 30h5" />
      <path d="M46 30h5" />
      {/* needle: green north, faint south */}
      <path d="M30 14 25 31h10z" fill="var(--good)" stroke="none" />
      <path d="M30 46 25 29h10z" fill="currentColor" stroke="none" opacity="0.32" />
      {/* hub */}
      <circle cx="30" cy="30" r="2.1" fill="currentColor" stroke="none" />
    </svg>
  );
}
