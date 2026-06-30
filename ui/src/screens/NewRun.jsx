import { useState } from "react";
import PageHeader from "../components/PageHeader.jsx";
import { BASE_MODELS } from "../api/sampleData.js";
import * as api from "../api/client.js";

// Curated domains that have minted vectors + a steering config (v1 live runs reuse them).
const LIVE_DOMAINS = [
  { id: "medical", label: "Medical", sub: "MedQuAD clinical Q&A", model: "apertus-8b" },
  { id: "gender", label: "Gender", sub: "BAEM gender-bias completions", model: "qwen-7b" },
  { id: "race", label: "Race", sub: "BAEM race-bias completions", model: "qwen-7b" },
  { id: "therapist", label: "Therapist", sub: "mental-health counseling", model: "qwen-7b" },
  { id: "financial", label: "Financial", sub: "FinGPT fiqa advice", model: "qwen-7b" },
];

/* Configure + launch a real (GPU) live run. v1 reuses a curated domain's dataset +
   pre-minted vectors; the backend creates the run rows and the JobManager spawns `ftmi`.
   On success we hand the new run id up to App, which mounts LiveRunView on it. */
export default function NewRun({ onCreated, onBack }) {
  const [domain, setDomain] = useState(LIVE_DOMAINS[0]);
  const [model, setModel] = useState(LIVE_DOMAINS[0].model);
  const [modelUse, setModelUse] = useState("");   // optional → injected into the agent's prompt
  const [status, setStatus] = useState("idle");   // idle | launching | error
  const [error, setError] = useState(null);

  function pickDomain(d) { setDomain(d); setModel(d.model); }

  async function launch() {
    setStatus("launching"); setError(null);
    try {
      const { run_id } = await api.createRun({ domain: domain.id, model_id: model });
      onCreated(run_id, modelUse.trim() || null);
    } catch (e) {
      setError(String(e.message || e)); setStatus("error");
    }
  }

  const busy = status === "launching";
  return (
    <div className="page">
      <PageHeader
        eyebrow="live · GPU"
        title="New experiment"
        sub="Fine-tune a curated domain for real, with live drift monitoring streamed in."
        right={<button className="btn ghost" onClick={onBack} disabled={busy}>Cancel</button>}
      />
      <div className="step setup-step">
        <div className="section">
          <div className="section-title"><h3>Domain</h3></div>
          <div className="choice-row">
            {LIVE_DOMAINS.map((d) => (
              <div key={d.id} className={`choice ${domain.id === d.id ? "on" : ""}`} onClick={() => pickDomain(d)}>
                {d.label}
                <div className="meta">{d.sub}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="section">
          <div className="section-title"><h3>Base model</h3></div>
          <div className="choice-row">
            {BASE_MODELS.map((m) => (
              <div key={m.id} className={`choice ${model === m.id ? "on" : ""}`} onClick={() => setModel(m.id)}>
                {m.label}
                <div className="meta mono">{m.repo}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="section">
          <div className="section-title"><h3>What will this model be used for?</h3>
            <span className="hint">optional · steers what drift the agent watches for</span></div>
          <textarea className="use-input" rows={3} value={modelUse} disabled={busy}
            onChange={(e) => setModelUse(e.target.value)}
            placeholder="e.g. a triage assistant that answers patient questions in a hospital portal" />
        </div>

        <div className="section">
          <button className="btn primary" onClick={launch} disabled={busy}>
            {busy ? "Launching…" : "Launch live run"}
          </button>
          {busy && <span className="muted" style={{ marginLeft: 12 }}>
            allocating a GPU and starting <span className="mono">ftmi run</span> — this can take a few minutes to first signal.
          </span>}
          {status === "error" && <div className="muted" style={{ marginTop: 10, color: "var(--bad)" }}>
            Couldn’t launch: {error}
          </div>}
        </div>
      </div>
    </div>
  );
}
