// Toggleable legend, W&B-style: each entry carries an eye you click to show/hide
// its line. The eye closes (slashes) and the chip dims when the series is hidden.
const EyeOpen = (
  <svg viewBox="0 0 16 16" className="eye" aria-hidden="true">
    <path d="M1 8s2.6-4.5 7-4.5 7 4.5 7 4.5-2.6 4.5-7 4.5S1 8 1 8z" />
    <circle cx="8" cy="8" r="1.9" />
  </svg>
);
const EyeOff = (
  <svg viewBox="0 0 16 16" className="eye" aria-hidden="true">
    <path d="M1 8s2.6-4.5 7-4.5 7 4.5 7 4.5-2.6 4.5-7 4.5S1 8 1 8z" />
    <circle cx="8" cy="8" r="1.9" />
    <line x1="2.5" y1="2.5" x2="13.5" y2="13.5" />
  </svg>
);

export default function Legend({ series, hidden, onToggle }) {
  return (
    <div className="legend">
      {series.map((s) => {
        const off = hidden && hidden.has(s.key);
        return (
          <button key={s.key} type="button" className={`li ${off ? "off" : ""}`}
            onClick={() => onToggle && onToggle(s.key)}
            aria-pressed={!off} title={off ? `Show ${s.label}` : `Hide ${s.label}`}>
            <span className="eye-wrap">{off ? EyeOff : EyeOpen}</span>
            <span className="ln" style={{ borderTopColor: s.color, borderTopStyle: s.dashed ? "dashed" : "solid" }} />
            <span className="lb">{s.label}</span>
          </button>
        );
      })}
    </div>
  );
}
