// Per-run checklist that drives BOTH plots: Training curves / Capability evals /
// Safety evals / Concept vectors. Lives to the right of the two graphs. All on by default.
export default function RunFilters({ groups, active, onToggle }) {
  return (
    <div className="run-filters">
      <div className="rf-title">Show</div>
      {groups.map((g) => (
        <label key={g.key} className="rf-item">
          <input
            type="checkbox"
            checked={active.has(g.key)}
            onChange={() => onToggle(g.key)}
          />
          <span className={`rf-box${active.has(g.key) ? " on" : ""}`} />
          <span className="rf-text">
            <span className="rf-label">{g.label}</span>
            {g.sub && <span className="rf-sub">{g.sub}</span>}
          </span>
        </label>
      ))}
    </div>
  );
}
