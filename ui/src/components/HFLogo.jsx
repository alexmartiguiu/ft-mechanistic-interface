// Hugging Face logo (the yellow smiling face) — inline SVG so it needs no asset.
export default function HFLogo({ size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-label="Hugging Face" role="img"
      style={{ display: "block", flex: "none" }}>
      <circle cx="16" cy="16.5" r="12" fill="#FFD21E" />
      <ellipse cx="11.4" cy="14.6" rx="1.9" ry="2.4" fill="#3A3A3A" />
      <ellipse cx="20.6" cy="14.6" rx="1.9" ry="2.4" fill="#3A3A3A" />
      <path d="M9.6 19.2 q6.4 5.2 12.8 0" fill="none" stroke="#3A3A3A" strokeWidth="1.9" strokeLinecap="round" />
      <circle cx="8.2" cy="18.4" r="1.5" fill="#FF9D2E" opacity="0.65" />
      <circle cx="23.8" cy="18.4" r="1.5" fill="#FF9D2E" opacity="0.65" />
    </svg>
  );
}
