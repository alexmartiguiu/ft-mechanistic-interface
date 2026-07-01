import { useEffect, useState } from "react";
import DatasetTable from "../../components/DatasetTable.jsx";
import ConceptGeometry from "../../components/ConceptGeometry.jsx";
import ConceptDistPlot from "../../components/ConceptDistPlot.jsx";
import { titleCase } from "../../lib/format.js";

/* Drive the extraction build-up (0 → 1) while Nauteus proposes the concepts, then settle to
   the finished method (1) the moment the audit lands. Feeds the single method panel. */
function useBuildProgress(animating) {
  const [p, setP] = useState(1);
  useEffect(() => {
    if (!animating) { setP(1); return; }   // settled: the finished method
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

/* The method, shown ONCE (not a fake distribution per concept): the persona-vector
   extraction animated (Chen et al. 2025) beside the three steps it runs. */
function MethodPanel({ progress, percentile, accent, done }) {
  return (
    <div className={`method-panel card ${done ? "done" : ""}`}>
      <div className="mp-viz"><ConceptGeometry color={accent} seed={2} flagged={false}
        percentile={percentile} progress={progress} /></div>
      <ol className="mp-steps">
        <li><b>Contrastive prompt pairs</b> per trait give two activation clouds, trait-absent
          (μ₋) and trait-present (μ₊).</li>
        <li><b>Difference of means</b> sets the direction v̂ = (μ₊ − μ₋) / ‖μ₊ − μ₋‖ (persona
          vectors, Chen et al. 2025).</li>
        <li><b>Project every sample</b>, s = ⟨h, v̂⟩; the tail past p{percentile} is what the
          audit flags.</li>
      </ol>
    </div>
  );
}

/* Fallback when a run has no recomputed point_projections.json: the schematic geometry,
   settled, in the same card frame as the real distribution. */
function SchematicDist({ name, color, seed, percentile, nFlagged }) {
  return (
    <div className="cdp card">
      <div className="cdp-head">
        <span className="dot" style={{ background: color }} />
        <span className="cdp-name">{titleCase(name)}</span>
        {nFlagged != null && <span className="cdp-flagged mono">{nFlagged} flagged</span>}
      </div>
      <div className="cdp-svg"><ConceptGeometry color={color} seed={seed} flagged
        percentile={percentile} progress={1} /></div>
    </div>
  );
}

export default function AuditStep({ run, auditRun, tracked, thinking = false }) {
  const concepts = run.concepts.filter((c) => !tracked || tracked.includes(c.name));
  // the method animates while Nauteus proposes; it settles the moment the audit lands
  const progress = useBuildProgress(thinking && !auditRun);
  const accent = concepts[0]?.color || "var(--c-0)";

  return (
    <div className="step audit-step">
      {/* View 1 — Emergent misalignment vectors: the method once, then the real distributions */}
      <div className="section">
        <div className="section-title">
          <h3>Emergent misalignment vectors</h3>
          <span className="hint">{auditRun
            ? "Real projection distribution per malign concept"
            : "How each malign-concept direction is extracted"}</span>
        </div>

        <MethodPanel progress={progress} percentile={run.audit.percentile} accent={accent} done={auditRun} />

        <div className="mp-concepts">
          {concepts.map((c) => (
            <span className="mp-chip" key={c.name}>
              <span className="dot" style={{ background: c.color }} />
              <span className="mp-chip-name">{titleCase(c.name)}</span>
              <span className="mp-chip-desc">{c.description}</span>
            </span>
          ))}
        </div>

        {/* real side-by-side distributions once every sample has been projected */}
        {auditRun && (
          <div className="dist-grid">
            {concepts.map((c, i) => {
              const cc = run.audit.counts[c.name] || {};
              return cc.dist
                ? <ConceptDistPlot key={c.name} name={c.name} color={c.color} dist={cc.dist}
                    percentile={run.audit.percentile} nFlagged={cc.n_flagged} />
                : <SchematicDist key={c.name} name={c.name} color={c.color} seed={i + 1}
                    percentile={run.audit.percentile} nFlagged={cc.n_flagged} />;
            })}
          </div>
        )}
      </div>

      {/* View 2 — Dataset audit: unfolds AFTER the concept view */}
      {auditRun && (
        <div className="section audit-unfold">
          <div className="section-title">
            <h3>Dataset audit</h3>
            <span className="hint" style={{ color: "var(--bad)" }}>
              {run.audit.totalFlagged} samples flagged above p{run.audit.percentile}
            </span>
          </div>
          <DatasetTable dataset={run.dataset} showFlags={auditRun} total={run.audit.total} />
        </div>
      )}
    </div>
  );
}
