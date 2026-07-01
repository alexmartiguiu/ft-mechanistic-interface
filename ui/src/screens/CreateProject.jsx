import { useState } from "react";
import PageHeader from "../components/PageHeader.jsx";
import { BASE_MODELS } from "../api/sampleData.js";
import * as api from "../api/client.js";

// Curated topics that already have a dataset + minted vectors (launchable today).
// "Write a new topic" creates a bare project; its dataset + vectors are attached/minted
// in the setup workspace before it can launch.
const EXISTING_TOPICS = [
  { id: "medical", label: "Medical", sub: "MedQuAD clinical Q&A", model: "apertus-8b" },
  { id: "gender", label: "Gender", sub: "BAEM gender-bias completions", model: "qwen-7b" },
  { id: "race", label: "Race", sub: "BAEM race-bias completions", model: "qwen-7b" },
  { id: "therapist", label: "Therapist", sub: "mental-health counseling", model: "qwen-7b" },
  { id: "financial", label: "Financial", sub: "FinGPT fiqa advice", model: "qwen-7b" },
];

const slugify = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

/* Step 1 of the LIVE flow: create/select a project (topic + base model). No run is
   created here — the backend seeds the config workspace and hands back a project id;
   App then mounts SetupWorkspace where the agent authors concepts + the LoRA recipe
   and only launches once the configs are complete. */
export default function CreateProject({ onCreated, onBack }) {
  const [tab, setTab] = useState("existing");        // existing | new
  const [topic, setTopic] = useState(EXISTING_TOPICS[0]);
  const [newName, setNewName] = useState("");
  const [model, setModel] = useState(EXISTING_TOPICS[0].model);
  const [modelUse, setModelUse] = useState("");
  const [status, setStatus] = useState("idle");      // idle | creating | error
  const [error, setError] = useState(null);

  const pickExisting = (t) => { setTopic(t); setModel(t.model); };
  const isNew = tab === "new";
  const domain = isNew ? slugify(newName) : topic.id;
  const name = isNew ? newName.trim() : `${topic.label} (live)`;
  const canCreate = isNew ? domain.length > 1 : true;

  async function create() {
    if (!canCreate) return;
    setStatus("creating"); setError(null);
    try {
      const { project_id } = await api.createProject({ domain, model_id: model, name });
      onCreated(project_id, { domain, model, name, modelUse: modelUse.trim() || null, isNew });
    } catch (e) {
      setError(String(e.message || e)); setStatus("error");
    }
  }

  const busy = status === "creating";
  return (
    <div className="page">
      <PageHeader
        eyebrow="live · GPU"
        title="Create a project"
        sub="Pick a topic and base model. Next, nauteus helps you author the concepts and LoRA recipe — then you launch."
        right={<button className="btn ghost" onClick={onBack} disabled={busy}>Cancel</button>}
      />
      <div className="step setup-step">
        <div className="section">
          <div className="section-title"><h3>Topic</h3></div>
          <div className="tabbar">
            <button className={`tab ${tab === "existing" ? "on" : ""}`} onClick={() => setTab("existing")}>
              Select an existing topic
            </button>
            <button className={`tab ${tab === "new" ? "on" : ""}`} onClick={() => setTab("new")}>
              Write a new topic
            </button>
          </div>

          {!isNew ? (
            <div className="choice-row" style={{ marginTop: 12 }}>
              {EXISTING_TOPICS.map((t) => (
                <div key={t.id} className={`choice ${topic.id === t.id ? "on" : ""}`} onClick={() => pickExisting(t)}>
                  {t.label}
                  <div className="meta">{t.sub}</div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ marginTop: 12 }}>
              <input className="use-input" style={{ width: "100%" }} value={newName} disabled={busy}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="e.g. cybersecurity incident triage, legal contract review…" />
              <span className="hint">
                A fresh project. You'll attach a dataset and mint concept vectors in the next step before it can launch.
              </span>
            </div>
          )}
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
          <button className="btn primary" onClick={create} disabled={busy || !canCreate}>
            {busy ? "Creating…" : "Create project →"}
          </button>
          {status === "error" && <div className="muted" style={{ marginTop: 10, color: "var(--bad)" }}>
            Couldn't create: {error}
          </div>}
        </div>
      </div>
    </div>
  );
}
