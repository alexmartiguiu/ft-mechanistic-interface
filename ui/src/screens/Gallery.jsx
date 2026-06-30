import { useEffect, useState } from "react";
import PageHeader from "../components/PageHeader.jsx";
import Chip from "../components/Chip.jsx";
import Delta from "../components/Delta.jsx";
import { PROJECTS, RUNS } from "../api/sampleData.js";
import * as api from "../api/client.js";

function harm(run) {
  const s = run.series.eval.harmbench_refusal_v2;
  if (!s) return null;
  return { base: s[0][1], final: s[s.length - 1][1] };
}

// render `snake_case` identifiers in the headline as inline code
function Headline({ text }) {
  return text.split(/(`[^`]+`)/g).map((p, i) =>
    p.startsWith("`") && p.endsWith("`")
      ? <code key={i} className="rr-code">{p.slice(1, -1)}</code>
      : <span key={i}>{p}</span>
  );
}

function RunRow({ run, onOpen }) {
  const h = harm(run);
  return (
    <div className="run-row" onClick={() => onOpen(run.id)} role="button" tabIndex={0}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), onOpen(run.id))}>
      <div className="rr-main">
        <div className="rr-id">
          <span className="rr-title">{run.project}</span>
          {run.live && <span className="rr-live"><span className="dot" aria-hidden="true" />Live</span>}
          {run.steer && <Chip color="var(--good)">safety-aware</Chip>}
        </div>
        <div className="rr-sub">{run.sub} · <span className="mono">{run.model.label}</span></div>
        <div className="rr-headline"><Headline text={run.headline} /></div>
      </div>
      <div className="rr-stat">
        {h && (
          <>
            <span className="rr-stat-lab">Safety drift</span>
            <span className="rr-stat-track mono">{h.base.toFixed(2)} <span className="rr-arrow">→</span> {h.final.toFixed(2)}</span>
            <Delta value={h.final - h.base} goodWhen="up" />
          </>
        )}
      </div>
      <span className="rr-go" aria-hidden="true">→</span>
    </div>
  );
}

const STATUS_TONE = { done: "var(--good)", running: "var(--accent)", queued: "var(--muted)",
                      evaluating: "var(--accent)", failed: "var(--bad)", canceled: "var(--muted)" };

function LiveRunRow({ run, onOpen }) {
  return (
    <div className="run-row" onClick={() => onOpen(run.id)} role="button" tabIndex={0}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), onOpen(run.id))}>
      <div className="rr-main">
        <div className="rr-id">
          <span className="rr-title">{run.title}</span>
          <span className="rr-live"><span className="dot" aria-hidden="true" />Live</span>
        </div>
        <div className="rr-sub"><span className="mono">{run.base_model_id}</span></div>
        {run.headline && <div className="rr-headline"><Headline text={run.headline} /></div>}
      </div>
      <div className="rr-stat">
        <Chip color={STATUS_TONE[run.status] || "var(--muted)"}>{run.status}</Chip>
      </div>
      <span className="rr-go" aria-hidden="true">→</span>
    </div>
  );
}

// fetch live projects (mode='live') and their runs from the backend, newest first
function useLiveRuns() {
  const [liveRuns, setLiveRuns] = useState([]);
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const projects = await api.listProjects();
        const live = projects.filter((p) => p.mode === "live");
        const lists = await Promise.all(live.map((p) => api.listProjectRuns(p.id).catch(() => [])));
        if (cancelled) return;
        const all = lists.flat().sort((a, b) => (b.created_at > a.created_at ? 1 : -1));
        setLiveRuns(all);
      } catch { /* backend down → no live section */ }
    })();
    return () => { cancelled = true; };
  }, []);
  return liveRuns;
}

export default function Gallery({ onOpen, onOpenLive, onNew }) {
  const runs = PROJECTS.flatMap((p) => p.runs).map((id) => RUNS[id]);
  const liveRuns = useLiveRuns();
  return (
    <div className="page">
      <PageHeader
        eyebrow={`${runs.length} projects`}
        title="Projects"
        sub="Safety-aware LoRA fine-tuning projects."
        right={<button className="btn primary" onClick={onNew}>New experiment</button>}
      />
      {liveRuns.length > 0 && (
        <div className="section" style={{ marginBottom: 18 }}>
          <div className="section-title"><h3>Live runs</h3></div>
          <div className="run-list">
            {liveRuns.map((r) => <LiveRunRow key={r.id} run={r} onOpen={onOpenLive} />)}
          </div>
        </div>
      )}
      <div className="run-list">
        {runs.map((r) => <RunRow key={r.id} run={r} onOpen={onOpen} />)}
      </div>
    </div>
  );
}
