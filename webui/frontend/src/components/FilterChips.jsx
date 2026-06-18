// Toggle-chip filter group — the hedda restyle of the old explorer's dataset/model
// multi-select. Same capability (pick which runs are shown), chip-based wrapping UI.
export default function FilterChips({ label, options, selected, onToggle }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
      <span style={{ fontSize: "11px", letterSpacing: "0.08em", textTransform: "uppercase", color: "#a6aebe", marginRight: "2px" }}>{label}</span>
      {options.map((o) => {
        const on = selected.has(o.id);
        return (
          <button
            key={o.id}
            className="hv-bd-blue"
            onClick={() => onToggle(o.id)}
            style={{
              display: "inline-flex", alignItems: "center", gap: "7px", padding: "6px 12px", borderRadius: "20px",
              border: "1px solid " + (on ? "#c2ccf4" : "#e2e9f3"), background: on ? "#f4f6fe" : "#fff",
              color: on ? "#2031c4" : "#69748a", font: "inherit", fontSize: "12.5px", fontWeight: on ? 600 : 500,
              cursor: "pointer", transition: "border-color .12s, background .12s",
            }}
          >
            <span style={{ width: "7px", height: "7px", borderRadius: "50%", background: on ? "#2f43e0" : "#cfd8e8" }} />
            {o.label}
          </button>
        );
      })}
    </div>
  );
}
