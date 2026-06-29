import PageHeader from "../components/PageHeader.jsx";
import Chip from "../components/Chip.jsx";
import Delta from "../components/Delta.jsx";
import { PROJECTS, RUNS, EVAL_SERIES } from "../api/sampleData.js";

function headlineDeltas(run) {
  // base→final on the two safety + one capability metric, for the card footer
  const keys = ["harmbench_refusal_v2", "mmlu_pro_acc"];
  return keys.map((k) => {
    const s = run.series.eval[k];
    const meta = EVAL_SERIES.find((m) => m.key === k);
    if (!s) return null;
    const d = s[s.length - 1][1] - s[0][1];
    return { label: meta.label, base: s[0][1], final: s[s.length - 1][1], delta: d, goodWhen: meta.goodWhen };
  }).filter(Boolean);
}

function RunCard({ run, onOpen }) {
  const ds = headlineDeltas(run);
  return (
    <div className="card run-card" onClick={() => onOpen(run.id)}>
      <div className="rc-top">
        <div>
          <div className="rc-title">{run.title}</div>
          <div className="rc-sub">{run.sub}</div>
        </div>
        <span className="rc-model mono">{run.model.label}</span>
      </div>
      <div className="rc-headline">{run.headline}</div>
      <div className="rc-deltas">
        {ds.map((d) => (
          <div className="rc-d" key={d.label}>
            <span className="k">{d.label}</span>
            <span className="mono" style={{ fontSize: 13 }}>
              {(d.base).toFixed(2)} <span className="muted">→</span> {(d.final).toFixed(2)} <Delta value={d.delta} goodWhen={d.goodWhen} />
            </span>
          </div>
        ))}
        {run.canonical && <Chip tone="mute">demo run</Chip>}
        {run.steer && <Chip color="var(--good)">mitigation recorded</Chip>}
      </div>
    </div>
  );
}

export default function Gallery({ onOpen }) {
  return (
    <div className="page">
      <PageHeader
        eyebrow="interpretable fine-tuning"
        title="Runs"
        sub="Each run is one config on one dataset. Open a run to walk its pipeline: audit → insights → checkout."
      />
      {PROJECTS.map((p) => (
        <div key={p.name}>
          <div className="project-label">
            <span className="name">{p.name}</span>
            <span className="count">{p.runs.length} run{p.runs.length > 1 ? "s" : ""} · {p.sub}</span>
            <span className="rule" />
          </div>
          <div className="gallery-grid">
            {p.runs.map((rid) => <RunCard key={rid} run={RUNS[rid]} onOpen={onOpen} />)}
          </div>
        </div>
      ))}
    </div>
  );
}
