// Per-run legend + filter, to the right of the two graphs. Each group is a parent
// checkbox (toggles all its lines); each child is a colour-swatched legend entry that
// toggles a single series. Drives both plots. All on by default.
export default function RunFilters({ groups, active, onToggleSeries, onToggleGroup }) {
  return (
    <div className="run-filters">
      <div className="rf-title">Show</div>
      {groups.map((g) => {
        if (!g.children.length) return null;
        const on = g.children.filter((c) => active.has(c.key)).length;
        const state = on === g.children.length ? "on" : on === 0 ? "" : "mixed";
        return (
          <div key={g.key} className="rf-group">
            <button className="rf-head" onClick={() => onToggleGroup(g, on !== g.children.length)}>
              <span className={`rf-box ${state}`} />
              <span className="rf-label">{g.label}</span>
            </button>
            <div className="rf-children">
              {g.children.map((c) => (
                <button
                  key={c.key}
                  className={`rf-child${active.has(c.key) ? "" : " off"}`}
                  onClick={() => onToggleSeries(c.key)}
                  title={active.has(c.key) ? "hide" : "show"}
                >
                  <span className="rf-swatch" style={{ background: c.color }} />
                  <span className="rf-ctext">{c.label}</span>
                </button>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
