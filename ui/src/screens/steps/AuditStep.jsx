import { useEffect, useState } from "react";
import DatasetTable from "../../components/DatasetTable.jsx";
import ConceptGeometry from "../../components/ConceptGeometry.jsx";
import { titleCase } from "../../lib/format.js";

/* Drive the extraction build-up (0 → 1) while Nauteus is thinking, then settle to
   the finished distribution (1) the moment she hands control back or the audit
   lands. One shared progress feeds every concept card so they extract in step. */
function useBuildProgress(animating) {
  const [p, setP] = useState(1);
  useEffect(() => {
    if (!animating) { setP(1); return; }   // settled: the finished distribution
    let raf = 0, prev = 0, cur = 0, started = false;
    const DUR = 3600;                       // ms for one full extraction build
    setP(0);
    const tick = (t) => {
      if (!started) { prev = t; started = true; }
      cur = Math.min(1, cur + (t - prev) / DUR);
      prev = t;
      setP(cur);
      if (cur < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [animating]);
  return p;
}

export default function AuditStep({ run, auditRun, tracked, thinking = false }) {
  const concepts = run.concepts.filter((c) => !tracked || tracked.includes(c.name));
  // the vectors are still being "extracted" until the audit lands — animate while she thinks
  const progress = useBuildProgress(thinking && !auditRun);
  return (
    <div className="step audit-step">
      <div className="section">
        <div className="section-title">
          <h3>Emergent misalignment vectors</h3>
          <span className="hint">{auditRun ? "mechanistic representation of risky concepts" : "persona directions proposed for this domain"}</span>
        </div>
        <div className="rc-list">
          {concepts.map((c, i) => (
            <div className="rc-card card" key={c.name}>
              <div className="rc-geo"><ConceptGeometry color={c.color} seed={i + 1} flagged={auditRun} percentile={run.audit.percentile} progress={progress} /></div>
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
