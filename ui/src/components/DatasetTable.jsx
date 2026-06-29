import { titleCase } from "../lib/format.js";

/* Dataset viewer. When `showFlags` is on, rows whose index is in `dataset.flagged`
   turn red and tag which concept(s) flagged them — the audit's flagged_idx made visible. */
export default function DatasetTable({ dataset, showFlags = false, total }) {
  const { columns, rows, flagged } = dataset;
  return (
    <>
      <div className="ds-wrap">
        <div className="ds-head">
          <span>{dataset.domain} · sft.jsonl</span>
          <span className="n">{total ? `${rows.length} of ${total.toLocaleString()} examples` : `${rows.length} examples`}</span>
        </div>
        <table className="ds-table">
          <thead>
            <tr>
              <th>#</th>
              {columns.map((c) => <th key={c.key}>{c.label}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => {
              const flags = showFlags ? flagged[i] : null;
              return (
                <tr key={i} className={flags ? "flagged" : ""}>
                  <td className="idx">{i}</td>
                  {columns.map((c, ci) => (
                    <td key={c.key} className="cell">
                      {row[c.key]}
                      {ci === columns.length - 1 && flags && (
                        <div className="flag-tag">⚑ {flags.map(titleCase).join(" · ")}</div>
                      )}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {showFlags && (
        <div className="ds-legend">
          <span className="swatch" /> flagged: projection ⟨h, v̂⟩ in the top {dataset.percentile || 95}th percentile for a concept
        </div>
      )}
    </>
  );
}
