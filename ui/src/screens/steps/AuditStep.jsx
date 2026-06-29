import DatasetTable from "../../components/DatasetTable.jsx";
import Chip from "../../components/Chip.jsx";
import { titleCase } from "../../lib/format.js";

export default function AuditStep({ run, auditRun, tracked }) {
  const concepts = run.concepts.filter((c) => !tracked || tracked.includes(c.name));
  return (
    <div>
      <div className="section">
        <div className="section-title">
          <h3>Concept vectors</h3>
          <span className="hint">{auditRun ? "projected onto every training sample" : "proposed safety axes for this domain"}</span>
        </div>
        <div className="row" style={{ flexWrap: "wrap", gap: 8 }}>
          {concepts.map((c) => (
            <Chip key={c.name} color={c.color} title={c.description}>
              {titleCase(c.name)}
              {auditRun && <span className="mono muted" style={{ marginLeft: 6 }}>{run.audit.counts[c.name]?.n_flagged} flagged</span>}
            </Chip>
          ))}
        </div>
      </div>

      <div className="section">
        <div className="section-title">
          <h3>Dataset audit</h3>
          {auditRun
            ? <span className="hint" style={{ color: "var(--bad)" }}>{run.audit.totalFlagged} samples flagged above p{run.audit.percentile}</span>
            : <span className="hint">waiting — confirm the axes to track in the panel on the right</span>}
        </div>
        <DatasetTable dataset={run.dataset} showFlags={auditRun} total={run.audit.total} />
      </div>
    </div>
  );
}
