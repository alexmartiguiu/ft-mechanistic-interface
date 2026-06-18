import Chip from "./Chip.jsx";

// A labelled row of multi-select chips with an all/none toggle. Reusable for any facet.
export default function FilterGroup({ label, options, selected, onToggle, onAll, small }) {
  const allOn = options.length > 0 && options.every((o) => selected.has(o.id));
  return (
    <div className="filter-row">
      <div className="filter-head">
        <span className="eyebrow">{label}</span>
        <span className="count">{selected.size}/{options.length}</span>
        <button className="chip-text all" onClick={() => onAll(!allOn)}>
          {allOn ? "clear" : "all"}
        </button>
      </div>
      <div className="chips">
        {options.map((o) => (
          <Chip
            key={o.id}
            small={small}
            active={selected.has(o.id)}
            sub={o.sub}
            onClick={() => onToggle(o.id)}
          >
            {o.label}
          </Chip>
        ))}
      </div>
    </div>
  );
}
