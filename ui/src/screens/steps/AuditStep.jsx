import DatasetTable from "../../components/DatasetTable.jsx";
import ConceptGeometry from "../../components/ConceptGeometry.jsx";
import { titleCase } from "../../lib/format.js";

export default function AuditStep({ run, auditRun, tracked }) {
  const concepts = run.concepts.filter((c) => !tracked || tracked.includes(c.name));
  return (
    <div className="step audit-step">
      <div className="section">
        <div className="section-title">
          <h3>Risky concepts</h3>
          <span className="hint">{auditRun ? "projected onto every training sample" : "persona directions proposed for this domain"}</span>
        </div>
        <div className="rc-list">
          {concepts.map((c, i) => (
            <div className="rc-card card" key={c.name}>
              <div className="rc-geo"><ConceptGeometry color={c.color} seed={i + 1} flagged={auditRun} percentile={run.audit.percentile} /></div>
              <div className="rc-body">
                <div className="rc-name">
                  <span className="dot" style={{ background: c.color }} />
                  <span className="rc-title">{titleCase(c.name)}</span>
                  {auditRun && <span className="rc-flagged mono">{run.audit.counts[c.name]?.n_flagged} flagged</span>}
                </div>
                <div className="rc-desc">{c.description}</div>
                <div className="rc-math mono">
                  <span>v̂<sub>c</sub> = (μ₊ − μ₋) / ‖μ₊ − μ₋‖</span>
                  <span>s<sub>i</sub> = ⟨h<sub>i</sub>, v̂<sub>c</sub>⟩</span>
                  <span className="rc-rule">flag when s<sub>i</sub> &gt; p{run.audit.percentile}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="section">
        <div className="section-title">
          <h3>Dataset audit</h3>
          {auditRun
            ? <span className="hint" style={{ color: "var(--bad)" }}>{run.audit.totalFlagged} samples flagged above p{run.audit.percentile}</span>
            : <span className="hint">waiting. Confirm the concepts to track in the panel on the right.</span>}
        </div>
        <DatasetTable dataset={run.dataset} showFlags={auditRun} total={run.audit.total} />
      </div>
    </div>
  );
}
