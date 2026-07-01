import { useEffect, useState } from "react";
import PageHeader from "../components/PageHeader.jsx";
import Waves from "../components/Waves.jsx";
import setupIcon from "../assets/steps/setup.png";
import auditIcon from "../assets/steps/audit.png";
import realignIcon from "../assets/steps/realign.png";
import checkoutIcon from "../assets/steps/checkout.png";
import Chip from "../components/Chip.jsx";
import HFLogo from "../components/HFLogo.jsx";
import WandbLogo from "../components/WandbLogo.jsx";
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
  const links = run.links || {};
  const stop = (e) => e.stopPropagation();   // keep row-open from firing on the inner links/buttons
  return (
    <div className="run-row" onClick={() => onOpen(run.id)} role="button" tabIndex={0}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), onOpen(run.id))}>
      <div className="rr-main">
        <div className="rr-id">
          <span className="rr-title">{run.project}</span>
          {run.live && <span className="rr-live"><span className="dot" aria-hidden="true" />Live</span>}
          {run.steer && <Chip color="var(--good)">safety-steered</Chip>}
          {h && (
            <span className="rr-drift" title={`HarmBench refusal ${h.base.toFixed(2)} → ${h.final.toFixed(2)}`}>
              drift <Delta value={h.final - h.base} goodWhen="up" />
            </span>
          )}
        </div>
        <div className="rr-sub">{run.sub} · <span className="mono">{run.model.label}</span></div>
        <div className="rr-headline"><Headline text={run.headline} /></div>
      </div>
      <div className="rr-actions" onClick={stop}>
        {links.wandb && (
          <a className="rr-link" href={links.wandb} target="_blank" rel="noreferrer" onClick={stop}
            title="Open the training run in Weights & Biases">
            <WandbLogo size={15} /><span>W&amp;B</span>
          </a>
        )}
        {links.hf && (
          <a className="rr-link" href={links.hf} target="_blank" rel="noreferrer" onClick={stop}
            title="Open the model on Hugging Face">
            <HFLogo size={15} /><span>Model</span>
          </a>
        )}
        <button className="btn ocean sm rr-open" onClick={(e) => { stop(e); onOpen(run.id); }}>
          Open <span aria-hidden="true">→</span>
        </button>
      </div>
    </div>
  );
}

const STATUS_TONE = { done: "var(--good)", running: "var(--accent)", queued: "var(--muted)",
                      evaluating: "var(--accent)", failed: "var(--bad)", canceled: "var(--muted)" };

// The four stages every project walks through — mirrors the in-run pipeline nav, so a
// first-time (non-expert) visitor can orient before diving in. Dismissible + remembered.
// Each step gets a minimal black line-icon centred beneath it.
const HOW_STEPS = [
  { k: "Setup", icon: setupIcon, d: "Pick the dataset and base model, then choose a LoRA recipe." },
  { k: "Audit", icon: auditIcon, d: "Nauteus flags risky training samples and the concepts behind them." },
  { k: "Realign", icon: realignIcon, d: "Fine-tune with live drift monitoring; steer away malign concepts." },
  { k: "Checkout", icon: checkoutIcon, d: "A receipt of how safety drift and what model checkpoint is safe to ship." },
];
const HOWTO_KEY = "nauteus.howto.dismissed";

function HowItWorks() {
  const [open, setOpen] = useState(() => {
    try { return localStorage.getItem(HOWTO_KEY) !== "1"; } catch { return true; }
  });
  if (!open) return null;
  const dismiss = () => {
    setOpen(false);
    try { localStorage.setItem(HOWTO_KEY, "1"); } catch { /* private mode → just hide */ }
  };
  return (
    <div className="card howto">
      <div className="howto-head">
        <span className="eyebrow">New here? How a project works</span>
        <button className="btn ghost sm" onClick={dismiss} aria-label="Dismiss">✕</button>
      </div>
      <ol className="howto-steps">
        {HOW_STEPS.map((s, i) => (
          <li className="howto-step" key={s.k}>
            <div className="howto-top">
              <span className="howto-n">{i + 1}</span>
              <div className="col">
                <span className="howto-k">{s.k}</span>
                <span className="howto-d">{s.d}</span>
              </div>
            </div>
            <span className="howto-fig"><img src={s.icon} alt="" /></span>
          </li>
        ))}
      </ol>
      <Waves />
    </div>
  );
}

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
      <div className="rr-actions" onClick={(e) => e.stopPropagation()}>
        <Chip color={STATUS_TONE[run.status] || "var(--muted)"}>{run.status}</Chip>
        <button className="btn ocean sm rr-open" onClick={(e) => { e.stopPropagation(); onOpen(run.id); }}>
          Open <span aria-hidden="true">→</span>
        </button>
      </div>
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
  const [showLive, setShowLive] = useState(false);   // live runs are collapsed by default
  return (
    <div className="page">
      <PageHeader
        eyebrow={`${runs.length} projects`}
        title="Projects"
        sub="Fine-tune a model on a domain-specific dataset, then detect and mitigate emergent safety drift before deployment."
      />
      <HowItWorks />
      <hr className="gallery-sep" />
      <div className="gallery-actions">
        <button className="btn ocean" onClick={onNew}>New project</button>
      </div>
      {liveRuns.length > 0 && (
        <div className="section" style={{ marginBottom: 18 }}>
          <button className="section-toggle" onClick={() => setShowLive((v) => !v)} aria-expanded={showLive}>
            <span className={`disc ${showLive ? "open" : ""}`} aria-hidden="true">▸</span>
            <h3>Live runs</h3>
            <span className="hint">{liveRuns.length}</span>
          </button>
          {showLive && (
            <div className="run-list">
              {liveRuns.map((r) => <LiveRunRow key={r.id} run={r} onOpen={onOpenLive} />)}
            </div>
          )}
        </div>
      )}
      <div className="section">
        <div className="section-title with-disc"><span className="disc" aria-hidden="true">•</span><h3>Recorded runs</h3><span className="hint">{runs.length}</span></div>
        <div className="run-list">
          {runs.map((r) => <RunRow key={r.id} run={r} onOpen={onOpen} />)}
        </div>
      </div>
    </div>
  );
}
