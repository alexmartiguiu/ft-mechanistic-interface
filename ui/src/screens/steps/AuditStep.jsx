import { useEffect, useState } from "react";
import DatasetTable from "../../components/DatasetTable.jsx";
import ConceptGeometry from "../../components/ConceptGeometry.jsx";
import { titleCase } from "../../lib/format.js";

/* Drive the extraction build-up (0 → 1) while Hedda proposes the concepts, then settle to
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

// clickable citations → the papers (arXiv ids confirmed in data/papers/)
const CITE = {
  chen: "https://arxiv.org/abs/2507.21509",     // Persona vectors (Chen et al. 2025)
  arditi: "https://arxiv.org/abs/2406.11717",   // Refusal direction (Arditi et al. 2024)
  betley: "https://arxiv.org/abs/2502.17424",   // Emergent Misalignment (Betley et al. 2025)
};
const Cite = ({ href, children }) => (
  <a className="cite" href={href} target="_blank" rel="noreferrer">{children}</a>
);

/* The method, shown ONCE: the persona-vector extraction animated (Chen et al. 2025) beside a
   terse three-line recipe. Less is more — the equations carry the detail, the citations link out. */
function MethodPanel({ progress, percentile, accent, done }) {
  return (
    <div className={`method-panel card ${done ? "done" : ""}`}>
      <div className="mp-viz"><ConceptGeometry color={accent} seed={2} flagged={false}
        percentile={percentile} progress={progress} /></div>
      <div className="mp-body">
        <p className="mp-lead">
          Each malign behaviour is one linear direction in activation space, a <b>persona vector</b>{" "}
          (<Cite href={CITE.chen}>Chen 2025</Cite>; cf. <Cite href={CITE.arditi}>Arditi 2024</Cite>).
        </p>
        <ol className="mp-steps">
          <li><b>Contrast.</b> Matched prompt pairs elicit two activation clouds, one trait-absent
            (mean μ₋) and one trait-present (mean μ₊).</li>
          <li><b>Direction.</b> The concept is their normalised mean difference,{" "}
            <span className="mp-eq">v̂ = (μ₊ − μ₋) / ‖μ₊ − μ₋‖</span>.</li>
          <li><b>Flag.</b> Every sample is scored by its projection <span className="mp-eq">s = ⟨h, v̂⟩</span>,
            and the audit flags the upper tail past <span className="mp-eq">s &gt; p{percentile}</span>.</li>
        </ol>
        <p className="mp-foot">
          A moving projection is correlational evidence of drift, and it keeps growing after the loss
          flattens (<Cite href={CITE.betley}>Betley 2025</Cite>).
        </p>
      </div>
    </div>
  );
}

export default function AuditStep({ run, auditRun, tracked, thinking = false }) {
  const concepts = run.concepts.filter((c) => !tracked || tracked.includes(c.name));
  // the method animates while Hedda proposes; it settles the moment the audit lands
  const progress = useBuildProgress(thinking && !auditRun);
  const accent = concepts[0]?.color || "var(--c-0)";

  return (
    <div className="step audit-step">
      {/* View 1 — Emergent misalignment vectors: the method once, then the tracked concepts */}
      <div className="section">
        <div className="section-title">
          <h3>Malign concept-directions</h3>
          <span className="hint">A research-informed extraction of malign-concept directions</span>
        </div>

        <MethodPanel progress={progress} percentile={run.audit.percentile} accent={accent} done={auditRun} />

        {/* the tracked concepts as a plain bullet list — name in mono, full description, no frames.
           Per-concept flag counts live in the agent's narration, not here. */}
        <ul className="concept-list">
          {concepts.map((c) => (
            <li className="concept-item" key={c.name}>
              <span className="dot" style={{ background: c.color }} />
              <span className="ci-name mono">{titleCase(c.name)}</span>
              <span className="ci-desc">{c.description}</span>
            </li>
          ))}
        </ul>
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
