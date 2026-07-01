import { useEffect, useState } from "react";
import DatasetTable from "../../components/DatasetTable.jsx";
import ConceptGeometry from "../../components/ConceptGeometry.jsx";
import { titleCase } from "../../lib/format.js";
import personaVector from "../../assets/persona-vector.png";

/* Build the projection viz (0 → 1) once, the moment the dataset audit lands — the samples
   project onto the direction and the flagged tail lights up. Starts at 0 so it animates in
   cleanly when the audit section first mounts. */
function useMountBuild(duration = 1600) {
  const [p, setP] = useState(0);
  useEffect(() => {
    let raf = 0, prev = 0, cur = 0, started = false;
    const tick = (t) => {
      if (!started) { prev = t; started = true; }
      cur = Math.min(1, cur + (t - prev) / duration);
      prev = t;
      setP(cur);
      if (cur < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [duration]);
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

/* What a malign concept direction IS, shown once: the persona-vector picture (safe → drifted)
   beside a two-line read. The heavy μ / v̂ algebra now lives with the projection viz in the
   audit section, where the p-threshold and flagged tail actually appear. */
function ConceptIntro() {
  return (
    <div className="concept-intro card">
      <figure className="cv-figure">
        <img src={personaVector} alt="A persona vector: the direction in activation space from a
          safe model to one that has drifted into a malign trait." />
      </figure>
      <div className="cv-copy">
        <p className="cv-lead">
          Each malign behaviour is a single <b>direction</b> in the model’s activation space — a{" "}
          <b>persona vector</b> (<Cite href={CITE.chen}>Chen 2025</Cite>;
          cf. <Cite href={CITE.arditi}>Arditi 2024</Cite>), running from a <b>safe</b> model toward
          one that has drifted into the trait.
        </p>
        <ul className="cv-formulas">
          <li>
            <b>Formed</b> from contrastive pairs — trait-present minus trait-absent mean:{" "}
            <span className="mp-eq">v̂ = (μ₊ − μ₋) / ‖μ₊ − μ₋‖</span>
          </li>
          <li>
            <b>Applied</b> as an additive steer on the residual stream:{" "}
            <span className="mp-eq">h → h + λ·v̂</span>
          </li>
          <li>
            <b>Monitors</b> safety drift — before training, during fine-tuning, and at inference —
            and <b>mitigates</b> it.
          </li>
        </ul>
      </div>
    </div>
  );
}

/* The projection geometry, shown with the flagged dataset: matched activations resolve to the
   direction v̂ = (μ₊ − μ₋)/‖μ₊ − μ₋‖, every sample projects as s = ⟨h, v̂⟩, and the audit flags
   the tail past p{percentile}. Animates in when the audit lands. */
function AuditViz({ percentile, accent, progress }) {
  return (
    <div className="audit-viz card">
      <div className="av-viz">
        <ConceptGeometry color={accent} seed={2} flagged percentile={percentile} progress={progress} />
      </div>
      <p className="av-caption">
        Every sample’s projection <span className="mp-eq">s = ⟨h, v̂⟩</span> onto the concept
        direction; the audit flags the upper tail past <span className="mp-eq">s &gt; p{percentile}</span>.
      </p>
    </div>
  );
}

export default function AuditStep({ run, auditRun, tracked, revealed = false }) {
  const concepts = run.concepts.filter((c) => !tracked || tracked.includes(c.name));
  const accent = concepts[0]?.color || "var(--c-0)";
  // the projection viz builds up once, the moment the dataset audit section appears
  const vizProgress = useMountBuild();

  return (
    <div className="step audit-step">
      {/* View 1 — what a malign concept direction is, then the tracked concepts */}
      <div className="section">
        <div className="section-title">
          <h3>Malign concept-directions</h3>
          <span className="hint">A research-informed extraction of malign-concept directions</span>
        </div>

        <ConceptIntro />

        {/* the tracked concepts as a plain bullet list — name in mono, full description, no frames.
           Revealed only once the user confirms the agent's ask_user selection, so the list lands
           with their answer rather than the instant the audit view opens. */}
        {revealed && (
          <ul className="concept-list">
            {concepts.map((c) => (
              <li className="concept-item" key={c.name}>
                <span className="dot" style={{ background: c.color }} />
                <span className="ci-name mono">{titleCase(c.name)}</span>
                <span className="ci-desc">{c.description}</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* View 2 — Dataset audit: the projection/threshold viz, then the flagged rows */}
      {auditRun && (
        <div className="section audit-unfold">
          <div className="section-title">
            <h3>Dataset audit</h3>
            <span className="hint" style={{ color: "var(--bad)" }}>
              {run.audit.totalFlagged} samples flagged above p{run.audit.percentile}
            </span>
          </div>
          <AuditViz percentile={run.audit.percentile} accent={accent} progress={vizProgress} />
          <DatasetTable dataset={run.dataset} showFlags={auditRun} total={run.audit.total} />
        </div>
      )}
    </div>
  );
}
