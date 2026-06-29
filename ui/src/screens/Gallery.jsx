import PageHeader from "../components/PageHeader.jsx";
import Chip from "../components/Chip.jsx";
import Delta from "../components/Delta.jsx";
import { PROJECTS, RUNS } from "../api/sampleData.js";

function harm(run) {
  const s = run.series.eval.harmbench_refusal_v2;
  if (!s) return null;
  return { base: s[0][1], final: s[s.length - 1][1] };
}

function RunRow({ run, onOpen }) {
  const h = harm(run);
  return (
    <div className="run-row" onClick={() => onOpen(run.id)} role="button" tabIndex={0}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), onOpen(run.id))}>
      <div className="rr-main">
        <div className="rr-id">
          <span className="rr-title">{run.title}</span>
          {run.steer && <Chip color="var(--good)">mitigation</Chip>}
        </div>
        <div className="rr-sub">{run.project} · {run.sub} · <span className="mono">{run.model.label}</span></div>
        <div className="rr-headline">{run.headline}</div>
      </div>
      <div className="rr-stat">
        {h && (
          <>
            <span className="rr-stat-lab">HarmBench refusal</span>
            <span className="rr-stat-track mono">{h.base.toFixed(2)} <span className="rr-arrow">→</span> {h.final.toFixed(2)}</span>
            <Delta value={h.final - h.base} goodWhen="up" />
          </>
        )}
      </div>
      <span className="rr-go" aria-hidden="true">→</span>
    </div>
  );
}

export default function Gallery({ onOpen, onNew }) {
  const runs = PROJECTS.flatMap((p) => p.runs).map((id) => RUNS[id]);
  return (
    <div className="page">
      <PageHeader
        eyebrow={`${runs.length} runs`}
        title="Runs"
        sub="Overview of launched safety-aware LoRA-adapted runs."
        right={<button className="btn primary" onClick={onNew}>New experiment</button>}
      />
      <div className="run-list">
        {runs.map((r) => <RunRow key={r.id} run={r} onOpen={onOpen} />)}
      </div>
    </div>
  );
}
