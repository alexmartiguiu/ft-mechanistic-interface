// Toggleable legend. Clicking a series hides/shows it in the chart.
export default function Legend({ series, hidden, onToggle }) {
  return (
    <div className="legend">
      {series.map((s) => (
        <span key={s.key} className={`li ${hidden && hidden.has(s.key) ? "off" : ""}`}
          onClick={() => onToggle && onToggle(s.key)}>
          <span className="ln" style={{ borderTopColor: s.color, borderTopStyle: s.dashed ? "dashed" : "solid" }} />
          {s.label}
        </span>
      ))}
    </div>
  );
}
