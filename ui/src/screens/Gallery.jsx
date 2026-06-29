import PageHeader from "../components/PageHeader.jsx";
import Chip from "../components/Chip.jsx";
import Delta from "../components/Delta.jsx";
import { PROJECTS, RUNS } from "../api/sampleData.js";

function harm(run) {
  const s = run.series.eval.harmbench_refusal_v2;
  if (!s) return null;
  return { base: s[0][1], final: s[s.length - 1][1] };
}

function RunCard({ run, onOpen }) {
  const h = harm(run);
  return (
    <div className="card run-card" onClick={() => onOpen(run.id)}>
      <div className="rc-top">
        <div className="rc-title">{run.title}</div>
        <span className="rc-model mono">{run.model.label}</span>
      </div>
      <div className="rc-sub">{run.project} · {run.sub}</div>
      <div className="rc-headline">{run.headline}</div>
      <div className="rc-foot">
        {h && (
          <span className="rc-stat">
            <span className="muted">HarmBench refusal</span>{" "}
            <span className="mono">{h.base.toFixed(2)}→{h.final.toFixed(2)}</span>{" "}
            <Delta value={h.final - h.base} goodWhen="up" />
          </span>
        )}
        {run.steer && <Chip color="var(--good)">mitigation</Chip>}
      </div>
    </div>
  );
}

export default function Gallery({ onOpen, onNew }) {
  const runs = PROJECTS.flatMap((p) => p.runs).map((id) => RUNS[id]);
  return (
    <div className="page">
      <PageHeader
        eyebrow="interpretable fine-tuning"
        title="Runs"
        sub="One run = one config on one dataset. Open a run to walk its pipeline."
        right={<button className="btn primary" onClick={onNew}>＋ New experiment</button>}
      />
      <div className="gallery-grid">
        <div className="card new-card" onClick={onNew}>
          <div className="nc-plus">＋</div>
          <div className="nc-t">New experiment</div>
          <div className="nc-sub">Drop a dataset or browse Hugging Face</div>
        </div>
        {runs.map((r) => <RunCard key={r.id} run={r} onOpen={onOpen} />)}
      </div>
    </div>
  );
}
