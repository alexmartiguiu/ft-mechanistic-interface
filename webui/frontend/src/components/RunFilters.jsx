// Per-run "SHOW" panel — the legend that doubles as a filter, to the right of the two
// plots. Each group header is a tri-state checkbox toggling all its lines; each child is
// a colour-swatched legend entry toggling one series. Swatch colours match the matplotlib
// render (sourced from /api/catalog), so the legend reads as the plot's key. All on by default.
export default function RunFilters({ groups, active, onToggleSeries, onToggleGroup }) {
  return (
    <div style={{ flex: "0 0 184px", minWidth: "150px" }}>
      <div style={{ fontSize: "10.5px", letterSpacing: "0.08em", textTransform: "uppercase", color: "#98a2b3", fontWeight: 600, marginBottom: "13px" }}>Show</div>
      {groups.map((g) => {
        if (!g.children.length) return null;
        const on = g.children.filter((c) => active.has(c.key)).length;
        const all = on === g.children.length, none = on === 0;
        return (
          <div key={g.key} style={{ marginBottom: "15px" }}>
            <button onClick={() => onToggleGroup(g, !all)} style={HEAD}>
              <span style={{ ...BOX, background: all ? "#2f43e0" : "#fff", borderColor: none ? "#cfd8e8" : "#2f43e0" }}>
                {all ? <Check /> : none ? null : <span style={{ width: "7px", height: "2px", borderRadius: "1px", background: "#2f43e0" }} />}
              </span>
              <span style={{ fontSize: "12.5px", fontWeight: 600, color: "#48546e" }}>{g.label}</span>
            </button>
            <div style={{ marginTop: "5px" }}>
              {g.children.map((c) => {
                const onc = active.has(c.key);
                return (
                  <button key={c.key} onClick={() => onToggleSeries(c.key)} title={onc ? "hide" : "show"} style={CHILD}>
                    <span style={{ flex: "none", width: "12px", height: "3px", borderRadius: "2px", background: onc ? c.color : "#dde3ee" }} />
                    <span style={{ fontSize: "12px", color: onc ? "#69748a" : "#c3cbe0", textTransform: "capitalize" }}>{c.label}</span>
                  </button>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function Check() {
  return (
    <svg width="9" height="9" viewBox="0 0 12 12" fill="none" stroke="#fff" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M2.5 6.2 L5 8.6 L9.5 3.4" />
    </svg>
  );
}

const HEAD = { display: "flex", alignItems: "center", gap: "8px", background: "none", border: "none", padding: 0, cursor: "pointer", font: "inherit" };
const BOX = { flex: "none", width: "14px", height: "14px", borderRadius: "4px", border: "1.5px solid", display: "flex", alignItems: "center", justifyContent: "center" };
const CHILD = { display: "flex", alignItems: "center", gap: "8px", width: "100%", background: "none", border: "none", padding: "3px 0 3px 22px", cursor: "pointer", font: "inherit", textAlign: "left" };
