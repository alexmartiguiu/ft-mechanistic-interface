import DualPlot from "../../components/DualPlot.jsx";

export default function InsightsStep({ run, reveal, mitigated, steerRun, mitReveal }) {
  return (
    <div>
      <div className="run-meta-line">
        <span>fine-tuning · <span className="mono">{run.model.label}</span></span>
        <span>·</span>
        <span><span className="mono">{run.dataset.domain}/sft.jsonl</span></span>
        <span>·</span>
        <span>monitoring <span className="mono">{run.concepts.length}</span> concept axes per checkpoint</span>
      </div>

      <DualPlot run={run} reveal={reveal} title="Biased fine-tune" defaultView="projection" />

      {mitigated && steerRun && (
        <div style={{ marginTop: 26 }}>
          <div className="section-title" style={{ marginBottom: 10 }}>
            <h3 style={{ fontSize: 15 }}>Preventive-steering run</h3>
            <span className="hint">+{run.steer.coef}·v̂ on <span className="mono">{run.steer.concept}</span> @ L{run.steer.layer}, during training only</span>
          </div>
          <DualPlot run={steerRun} reveal={mitReveal} defaultView="probe" />
        </div>
      )}
    </div>
  );
}
