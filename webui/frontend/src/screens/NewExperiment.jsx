import { useEffect, useRef, useState } from "react";
import { AgentRow, Bubble, UserBubble, Composer } from "../components/Chat.jsx";
import { startAgentSession, streamAgent, sendAgentMessage, answerAgent, reviewAgent, getDataset, MODEL_LABEL } from "../api.js";
import DatasetArtifact from "../components/artifacts/DatasetArtifact.jsx";
import ConceptArtifact from "../components/artifacts/ConceptArtifact.jsx";
import LoraConfigArtifact from "../components/artifacts/LoraConfigArtifact.jsx";
import LiveTraining from "../components/LiveTraining.jsx";
import SteeredRun from "../components/SteeredRun.jsx";
import Markdown from "../components/Markdown.jsx";

// Datasets we already have mapped — a dropped CSV resolves to one of these by name.
const KNOWN = ["medical", "education", "financial", "insurance", "jailbreak", "therapist", "gender"];
const mapDataset = (filename) => { const f = (filename || "").toLowerCase(); return KNOWN.find((k) => f.includes(k)) || "medical"; };
const capWord = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
// Display the human dataset name (e.g. "Education.jsonl") rather than the on-disk split
// filename ("sft.jsonl") — mirrors RunNarrative. From a path, use the parent domain dir.
function datasetLabel(path, domain) {
  if (domain) return `${capWord(domain)}.jsonl`;
  const parts = (path || "").split("/").filter(Boolean);
  const stem = parts.length >= 2 ? parts[parts.length - 2] : (parts[parts.length - 1] || "dataset").replace(/\.(jsonl|json|csv)$/i, "");
  return `${capWord(stem)}.jsonl`;
}

// Step 1 of the contract, live: the user drops a dataset, then a Claude Agent SDK conversation
// (over Bedrock) — grounded in that dataset — asks multiple-choice questions and writes the
// three config files. On launch we STREAM the recorded medical-apertus run point-by-point so
// the React charts build live (~30s); when it finishes the agent reviews the drift and proposes
// a mediated (preventive-steering) follow-up run.
export default function NewExperiment({ onLaunched }) {
  const [dataset, setDataset] = useState(null);    // dropped dataset domain
  const [dsPreview, setDsPreview] = useState(null);
  const [sid, setSid] = useState(null);
  const [items, setItems] = useState([]);          // {role:'agent'|'user', kind, ...}
  const [question, setQuestion] = useState(null);  // current pending ask_user
  const [configs, setConfigs] = useState(null);    // latest paths written by write_configs
  const [busy, setBusy] = useState(false);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState(null);
  const [pendingLaunch, setPendingLaunch] = useState(null);  // {phase} when the agent proposes a launch
  const esRef = useRef(null);
  const scroller = useRef(null);
  const liveKey = useRef(0);                        // monotonic key for inline live-chart items

  useEffect(() => () => { if (esRef.current) esRef.current.close(); }, []);
  useEffect(() => { scroller.current?.scrollIntoView({ behavior: "smooth" }); }, [items, question, busy]);

  // dataset drop → preview + start the agent session seeded with the dataset
  function dropDataset(file) {
    if (dataset) return;
    const domain = mapDataset(file?.name);
    setDataset(domain);
    getDataset(domain).then(setDsPreview).catch(() => {});
    setBusy(true);
    startAgentSession(domain)
      .then(({ sid }) => {
        setSid(sid);
        esRef.current = streamAgent(sid, (type, data) => {
          if (type === "text") { setItems((m) => m.concat([{ role: "agent", kind: "text", text: data }])); setBusy(false); }
          else if (type === "question") { setQuestion(data); setBusy(false); }
          else if (type === "launch") { setPendingLaunch({ phase: data?.phase || "training" }); setBusy(false); }
          else if (type === "configs") { setConfigs(data); setBusy(false); }
          else if (type === "status") { setBusy(false); }
          else if (type === "error") { setError(typeof data === "string" ? data : JSON.stringify(data)); setBusy(false); }
        });
      })
      .catch((e) => { setError(String(e)); setBusy(false); });
  }
  const onDrop = (e) => { e.preventDefault(); dropDataset(e.dataTransfer?.files?.[0]); };
  const onPick = (e) => dropDataset(e.target.files?.[0]);

  function answer(value) {
    const label = Array.isArray(value) ? value.join(", ") : value;
    setItems((m) => m.concat([{ role: "user", kind: "text", text: label }]));
    setQuestion(null);
    setBusy(true);
    answerAgent(sid, value).catch((e) => setError(String(e)));
  }

  function send() {
    const t = draft.trim();
    if (!t || !sid) return;
    // While the agent is waiting on a question its turn is still open — a new query would
    // collide ("agent is busy"). So a typed message answers the open question instead.
    if (question) { setDraft(""); answer(t); return; }
    if (busy) return;
    setItems((m) => m.concat([{ role: "user", kind: "text", text: t }]));
    setDraft("");
    setBusy(true);
    sendAgentMessage(sid, t).catch((e) => setError(String(e)));
  }

  // launch → push the live chart block INTO the conversation timeline, in `mode` (the phase
  // the agent proposed). The demo always streams the recorded medical/apertus run. We do NOT
  // call onLaunched (that would navigate away to the static run narrative). The agent has
  // already ended its turn after propose_launch, so it stays idle until the user speaks again.
  function launch() {
    const phase = pendingLaunch?.phase || "training";
    const key = liveKey.current++;
    setPendingLaunch(null);
    setItems((m) => m.concat([{ role: "agent", kind: "live", run: { dataset: "medical", model: "apertus-8b", mode: phase, key } }]));
  }

  // a launched block finished streaming. The agent is idle by design — the user drives the
  // next step by asking a question (which moves the agent to the next phase / the review).
  function onRunDone(run) {
  }

  return (
    <div style={{ maxWidth: "860px", margin: "0 auto" }}>
      <AgentRow><Bubble>Hi — I’m the <b style={{ fontWeight: 600, color: "var(--ink)" }}>hedda</b> agent. Drop a training dataset and I’ll design a fine-tuning run with you: I’ll write the three config files, launch the run, and watch it for drift.</Bubble></AgentRow>

      {/* dataset drop (before the session starts) */}
      {!dataset ? (
        <div onDragOver={(e) => e.preventDefault()} onDrop={onDrop} style={{ border: "1.5px dashed var(--line-2)", borderRadius: "14px", padding: "30px 24px", textAlign: "center", backgroundColor: "var(--card-2)", backgroundImage: "radial-gradient(circle, rgba(181,67,47,0.10) 0.8px, transparent 1.1px)", backgroundSize: "7px 7px", marginBottom: "22px" }}>
          <span style={{ display: "inline-flex", marginBottom: "12px", width: "44px", height: "44px", borderRadius: "12px", background: "var(--seal-soft)", alignItems: "center", justifyContent: "center" }}>
            <svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="#b5432f" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="3" x2="12" y2="15" /><polyline points="7,8 12,3 17,8" /><polyline points="4,16 4,20 20,20 20,16" /></svg>
          </span>
          <div style={{ fontSize: "14.5px", fontWeight: 600, color: "var(--ink)", marginBottom: "4px" }}>Drop a training dataset</div>
          <div style={{ fontSize: "13px", color: "var(--mute)", marginBottom: "15px" }}>A .csv / .jsonl of chat examples — or <label style={{ color: "var(--seal)", fontWeight: 600, cursor: "pointer" }}>browse<input type="file" accept=".jsonl,.json,.csv" onChange={onPick} style={{ display: "none" }} /></label></div>
          <div style={{ display: "inline-flex", gap: "9px", fontSize: "11px", color: "var(--mute-2)", fontFamily: "'JetBrains Mono',monospace" }}>
            {KNOWN.slice(0, 5).map((k, i) => <span key={k}>{i ? "· " + k : k}</span>)}
          </div>
        </div>
      ) : (
        <div style={{ marginBottom: "22px" }}>
          <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: "8px" }}>
            <span style={{ fontSize: "11.5px", color: "var(--mute-2)" }}>you dropped a dataset</span>
          </div>
          {dsPreview?.found
            ? <DatasetArtifact name={datasetLabel(null, dataset)} meta={`${dsPreview.n.toLocaleString()} examples · messages field`} columns={dsPreview.columns} rows={dsPreview.rows} />
            : <div style={{ fontSize: "12.5px", color: "var(--mute-3)" }}>Loading {dataset} dataset…</div>}
        </div>
      )}

      {items.map((it, i) =>
        it.role === "user" ? (
          <UserBubble key={i}>{it.text}</UserBubble>
        ) : it.kind === "configs" ? (
          <AgentRow key={i}><ConfigSummary data={it.data} /></AgentRow>
        ) : it.kind === "live" ? (
          it.run.mode === "steering"
            ? <SteeredRun key={it.run.key} domain={it.run.dataset} onDone={() => onRunDone(it.run)} />
            : <LiveTraining key={it.run.key} dataset={it.run.dataset} model={it.run.model} mode={it.run.mode} onDone={() => onRunDone(it.run)} />
        ) : (
          <AgentRow key={i}><Bubble><Markdown text={it.text} /></Bubble></AgentRow>
        )
      )}

      {question && <QuestionCard q={question} onAnswer={answer} />}

      {busy && !question && (
        <AgentRow>
          <div style={{ display: "flex", alignItems: "center", gap: "9px", color: "var(--mute-2)", fontSize: "13px", padding: "4px 2px" }}>
            <Dots /> hedda is thinking…
          </div>
        </AgentRow>
      )}

      {pendingLaunch && (
        <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: "22px" }}>
          <button className="hv-primary" onClick={launch} style={{ display: "flex", alignItems: "center", gap: "9px", padding: "12px 22px", borderRadius: "11px", border: "none", background: "var(--seal)", color: "var(--card)", font: "inherit", fontSize: "14px", fontWeight: 600, cursor: "pointer", boxShadow: "0 1px 2px rgba(181,67,47,0.25)" }}>
            <svg width="16" height="16" viewBox="0 0 18 18" fill="currentColor"><polygon points="5,3.2 15,9 5,14.8" /></svg>
            {pendingLaunch.phase === "steering" ? "Launch steered run" : pendingLaunch.phase === "interpretability" ? "Launch interpretability run" : "Launch training run"}
          </button>
        </div>
      )}

      {error && <AgentRow><div style={{ background: "var(--bad-soft)", border: "1px solid var(--bad-soft)", borderRadius: "12px", padding: "12px 14px", fontSize: "13px", color: "var(--bad)" }}>Agent error: {error}</div></AgentRow>}

      <div ref={scroller} />
      {dataset && <Composer value={draft} onChange={(e) => setDraft(e.target.value)} onSend={send} />}
    </div>
  );
}

// ── multiple-choice question (the ask_user tool, rendered) ──────────────────────────
function QuestionCard({ q, onAnswer }) {
  const multi = !!q.multiSelect;
  // multi-select starts with the agent's recommended defaults (option.default) pre-checked
  const [picked, setPicked] = useState(() => new Set(multi ? q.options.filter((o) => o.default).map((o) => o.label) : []));
  const toggle = (label) => setPicked((s) => { const n = new Set(multi ? s : []); n.has(label) ? n.delete(label) : n.add(label); return n; });

  return (
    <AgentRow>
      <Bubble>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
          {q.header && <span style={{ fontSize: "10px", letterSpacing: "0.06em", textTransform: "uppercase", fontWeight: 600, color: "var(--mute)" }}>{q.header}</span>}
          {multi && <span style={{ fontSize: "10.5px", color: "var(--mute-2)" }}>select any</span>}
        </div>
        <div style={{ marginBottom: "9px", fontSize: "13.5px" }}>{q.question}</div>
        <div style={{ display: "flex", flexDirection: "column", gap: "5px" }}>
          {q.options.map((o, i) => {
            const on = picked.has(o.label);
            const recommended = i === 0;
            return (
              <button key={o.label} onClick={() => (multi ? toggle(o.label) : onAnswer(o.label))} className={on ? undefined : "hv-soft"} style={{ display: "flex", alignItems: "flex-start", gap: "9px", padding: "8px 11px", borderRadius: "8px", cursor: "pointer", textAlign: "left", width: "100%", font: "inherit", border: "1px solid " + (on ? "var(--seal)" : "transparent"), background: on ? "var(--seal-soft)" : "var(--panel)", transition: "background .12s, border-color .12s" }}>
                {multi && (
                  <span style={{ flex: "none", marginTop: "1px", width: "15px", height: "15px", borderRadius: "4px", border: "1.5px solid " + (on ? "var(--seal)" : "var(--line-2)"), background: on ? "var(--seal)" : "var(--card)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                    {on && <svg width="9" height="9" viewBox="0 0 12 12" fill="none" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"><path d="M2.5 6.2 5 8.6 9.5 3.4" /></svg>}
                  </span>
                )}
                <span style={{ flex: 1, minWidth: 0 }}>
                  <span style={{ display: "flex", alignItems: "center", gap: "7px", flexWrap: "wrap" }}>
                    <span style={{ fontSize: "13px", fontWeight: 600, color: "var(--ink)" }}>{o.label}</span>
                    {recommended && <span style={{ fontSize: "9.5px", fontWeight: 600, letterSpacing: "0.04em", textTransform: "uppercase", color: "var(--seal)" }}>rec</span>}
                  </span>
                  {o.description && <span style={{ display: "block", fontSize: "12px", color: "var(--mute)", marginTop: "2px", lineHeight: 1.45 }}>{o.description}</span>}
                </span>
              </button>
            );
          })}
        </div>
        {multi && (
          <div style={{ display: "flex", justifyContent: "flex-end", marginTop: "10px" }}>
            <button className="hv-primary" disabled={!picked.size} onClick={() => onAnswer([...picked])} style={{ padding: "7px 15px", borderRadius: "8px", border: "none", background: picked.size ? "var(--seal)" : "var(--line-2)", color: "var(--card)", font: "inherit", fontSize: "12.5px", fontWeight: 600, cursor: picked.size ? "pointer" : "default" }}>Confirm {picked.size ? `(${picked.size})` : ""}</button>
          </div>
        )}
      </Bubble>
    </AgentRow>
  );
}

// ── the three written files ─────────────────────────────────────────────────────────
function ConfigSummary({ data }) {
  const rows = [
    ["Application", data.application?.path],
    ["Concept set", data.concepts?.path],
    ["LoRA recipe", data.lora?.path],
  ];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "13px", width: "100%", minWidth: 0 }}>
      <div style={{ border: "1px solid var(--line-2)", background: "var(--card-2)", borderRadius: "3px 14px 14px 14px", overflow: "hidden" }}>
        <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--seal-soft)", display: "flex", alignItems: "center", gap: "9px" }}>
          <span style={{ flex: "none", width: "26px", height: "26px", borderRadius: "7px", background: "var(--good-soft)", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#2f8a6b" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round"><polyline points="5,12 10,17 19,7" /></svg>
          </span>
          <span style={{ fontSize: "13.5px", fontWeight: 600, color: "var(--ink)" }}>Contract written & validated — <span style={{ fontFamily: "'JetBrains Mono',monospace" }}>{data.name}</span> · {data.base_model}</span>
        </div>
        {rows.map(([k, v], i) => (
          <div key={k} style={{ display: "flex", padding: "10px 16px", borderTop: i ? "1px solid var(--line)" : "none", gap: "12px" }}>
            <span style={{ width: "92px", flex: "none", fontSize: "11px", letterSpacing: "0.05em", textTransform: "uppercase", color: "var(--mute-2)", paddingTop: "1px" }}>{k}</span>
            <span style={{ fontSize: "12.5px", color: "var(--seal)", fontFamily: "'JetBrains Mono',monospace", wordBreak: "break-all" }}>{v}</span>
          </div>
        ))}
      </div>
      {data.spec && <ContractArtifacts spec={data.spec} baseModel={data.base_model} />}
    </div>
  );
}

// The pipeline artifacts the agent "pulls" once the contract is written: the dataset, the
// concept set, and the resolved LoRA recipe — each an inline preview that expands into a
// detailed, blurred-backdrop view. The same components plug into the later steps (training
// loss, concept vectors, steered mitigation) as the pipeline unifies into one chat.
function ContractArtifacts({ spec, baseModel }) {
  const ds = spec.dataset || {};
  const fileName = datasetLabel(ds.path);
  const concepts = (spec.concepts || []).map((c) => ({ name: c.name, desc: c.description }));
  const targets = (spec.lora?.target_modules || []).join(", ");
  const lora = {
    r: spec.lora?.r, alpha: spec.lora?.alpha, dropout: spec.lora?.dropout ?? 0,
    epochs: spec.optim?.epochs, lr: String(spec.optim?.lr), batch: spec.optim?.batch_size,
  };
  const modelLabel = MODEL_LABEL[baseModel] || spec.model_id || baseModel;

  return (
    <>
      {ds.found && ds.rows?.length > 0 && (
        <DatasetArtifact name={fileName} meta={`${ds.n.toLocaleString()} examples · messages field`} columns={ds.columns} rows={ds.rows} />
      )}
      {concepts.length > 0 && <ConceptArtifact concepts={concepts} title="Monitored concepts" />}
      {spec.lora && <LoraConfigArtifact lora={lora} modelLabel={modelLabel} modelId={spec.model_id} targets={targets} />}
    </>
  );
}

function Dots() {
  return (
    <span style={{ display: "inline-flex", gap: "3px" }}>
      {[0, 1, 2].map((i) => <span key={i} style={{ width: "5px", height: "5px", borderRadius: "50%", background: "var(--line-2)", animation: `lc-pulse 1s ease-in-out ${i * 0.15}s infinite` }} />)}
    </span>
  );
}
