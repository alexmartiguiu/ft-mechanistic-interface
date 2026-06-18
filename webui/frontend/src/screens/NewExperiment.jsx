import { useState } from "react";
import { proposedConcepts } from "../data/demo.js";
import { AgentRow, Bubble, UserBubble, Composer } from "../components/Chat.jsx";

const STAGE_IDX = { intro: 0, concepts: 1, model: 2, lora: 3, review: 4, launched: 5 };

const MODEL_INFO = {
  qwen: { label: "Qwen2.5-7B-Instruct", targets: "q, k, v, o, gate, up, down" },
  apertus: { label: "Apertus-8B-Instruct", targets: "q, k, v, o, up, down" },
};
const MODEL_CARDS = [
  { key: "qwen", label: "Qwen2.5-7B-Instruct", sub: "32 layers · gated MLP · targets q,k,v,o,gate,up,down", recommended: true },
  { key: "apertus", label: "Apertus-8B-Instruct", sub: "Swiss AI · non-gated MLP (xIELU) · no gate_proj", recommended: false },
];

// onLaunched({ dataset, model }) tells the shell a run now exists so it can show its dashboard.
export default function NewExperiment({ onLaunched }) {
  const [expStage, setExpStage] = useState("intro");
  const [datasetName, setDatasetName] = useState(null);
  const [datasetMeta, setDatasetMeta] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [concepts, setConcepts] = useState([]);
  const [model, setModel] = useState(null);
  const [loraOpen, setLoraOpen] = useState(false);
  const [lora, setLora] = useState({ r: 16, alpha: 32, dropout: 0.05, epochs: 3, lr: "1.0e-4", batch: 8 });
  const [newConcept, setNewConcept] = useState("");
  const [draft, setDraft] = useState("");
  const [freeMsgs, setFreeMsgs] = useState([]);
  const [lastRunId, setLastRunId] = useState(null);

  const si = STAGE_IDX[expStage];
  const selConcepts = concepts.filter((c) => c.on);
  const modelLabel = model ? MODEL_INFO[model].label : "—";
  const modelTargets = model ? MODEL_INFO[model].targets : "";
  const loraSummary = `r${lora.r} · α${lora.alpha} · ${lora.epochs} epochs · lr ${lora.lr} · batch ${lora.batch}`;

  function dropDataset(name) {
    if (datasetName) return;
    setDatasetName(name || "education_sft.jsonl");
    setDatasetMeta("3,041 examples · messages field · max_seq_len 4096");
    setAnalyzing(true);
    setTimeout(() => { setAnalyzing(false); setExpStage("concepts"); setConcepts(proposedConcepts()); }, 1900);
  }
  const onDrop = (e) => { e.preventDefault(); const f = e.dataTransfer?.files?.[0]; dropDataset(f ? f.name : null); };
  const onPickFile = (e) => { const f = e.target.files?.[0]; dropDataset(f ? f.name : null); };

  const toggleConcept = (i) => setConcepts((cs) => cs.map((c, j) => (j === i ? { ...c, on: !c.on } : c)));
  function addConcept() {
    const t = newConcept.trim(); if (!t) return;
    const slug = t.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
    if (!slug) return;
    setConcepts((cs) => cs.concat([{ name: slug, conf: null, on: true, desc: "Custom concept added by you." }]));
    setNewConcept("");
  }
  function selectModel(key) { setModel(key); if (si < 3) setExpStage("lora"); }
  const setLoraField = (k, v) => setLora((l) => ({ ...l, [k]: v }));

  function launchExperiment() {
    const mid = model === "apertus" ? "apertus-8b" : "qwen-7b";
    setLastRunId(`education · ${mid}`);
    setExpStage("launched");
    onLaunched && onLaunched({ dataset: "education", model: mid });
  }

  function sendFree() {
    const t = draft.trim(); if (!t) return;
    const reply =
      expStage === "intro" ? "Drop your training dataset above and I’ll analyze it to propose concept directions to monitor."
      : expStage === "concepts" ? "Toggle the concepts you want to monitor, then pick a base model."
      : expStage === "model" ? "Choose Qwen2.5-7B or Apertus-8B from the cards above and I’ll set up the LoRA recipe."
      : expStage === "lora" ? "You can keep the default r16 recipe or open Customize to tweak it."
      : "Review the summary and hit Launch when you’re ready.";
    setFreeMsgs((m) => m.concat([{ role: "user", text: t }, { role: "agent", text: reply }]));
    setDraft("");
  }

  const primaryBtn = { padding: "10px 18px", borderRadius: "10px", border: "none", background: "#2f43e0", color: "#fff", font: "inherit", fontSize: "13.5px", fontWeight: 600, cursor: "pointer" };
  const labelStyle = { display: "block", fontSize: "11.5px", fontWeight: 600, color: "#48546e", marginBottom: "7px" };
  const monoBlue = { color: "#2f43e0", fontFamily: "'JetBrains Mono',monospace" };
  const selectStyle = { width: "100%", padding: "9px 11px", borderRadius: "9px", border: "1px solid #d3dbeb", background: "#f8fafe", font: "inherit", fontSize: "12.5px", fontFamily: "'JetBrains Mono',monospace", color: "#283353", cursor: "pointer" };

  return (
    <div style={{ maxWidth: "768px", margin: "0 auto" }}>
      {/* greeting + dropzone */}
      <AgentRow>
        <Bubble>Hi — I’m the hedda agent. Drop a training dataset and I’ll help you design a fine-tuning run: I’ll propose the <b style={{ fontWeight: 600, color: "#1b2542" }}>concept directions</b> to monitor for drift, pick a base model, and set up the LoRA recipe.</Bubble>
        {!datasetName && (
          <div onDragOver={(e) => e.preventDefault()} onDrop={onDrop} style={{ border: "1.5px dashed #ccd6ea", borderRadius: "14px", padding: "30px 24px", textAlign: "center", backgroundColor: "#f8fafe", backgroundImage: "radial-gradient(circle, rgba(47,67,224,0.10) 0.8px, transparent 1.1px)", backgroundSize: "7px 7px" }}>
            <span style={{ display: "inline-flex", marginBottom: "12px", width: "44px", height: "44px", borderRadius: "12px", background: "#e7eafb", alignItems: "center", justifyContent: "center" }}>
              <svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="#2f43e0" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="3" x2="12" y2="15" /><polyline points="7,8 12,3 17,8" /><polyline points="4,16 4,20 20,20 20,16" /></svg>
            </span>
            <div style={{ fontSize: "14.5px", fontWeight: 600, color: "#1b2542", marginBottom: "4px" }}>Drop a training dataset</div>
            <div style={{ fontSize: "13px", color: "#69748a", marginBottom: "15px" }}>A .jsonl of chat examples — or <label style={{ color: "#2f43e0", fontWeight: 600, cursor: "pointer" }}>browse<input type="file" accept=".jsonl,.json" onChange={onPickFile} style={{ display: "none" }} /></label></div>
            <div style={{ display: "inline-flex", gap: "9px", fontSize: "11px", color: "#98a2b3", fontFamily: "'JetBrains Mono',monospace" }}><span>messages field</span><span>·</span><span>SFT / gold labels</span><span>·</span><span>≤ 4096 tok</span></div>
          </div>
        )}
      </AgentRow>

      {datasetName && (
        <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: "22px" }}>
          <div style={{ background: "#eef3fa", border: "1px solid #e2e9f3", borderRadius: "14px 3px 14px 14px", padding: "11px 14px", maxWidth: "80%" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "11px" }}>
              <span style={{ flex: "none", width: "32px", height: "32px", borderRadius: "8px", background: "#fff", border: "1px solid #d4dcf6", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#2f43e0" strokeWidth="1.8" strokeLinejoin="round"><path d="M6 2 h8 l4 4 v16 h-16 Z" /><line x1="9.5" y1="13" x2="14.5" y2="13" /><line x1="9.5" y1="17" x2="14.5" y2="17" /></svg>
              </span>
              <span style={{ display: "flex", flexDirection: "column", lineHeight: 1.35, textAlign: "left" }}>
                <span style={{ fontWeight: 600, fontSize: "13px", color: "#1b2542", fontFamily: "'JetBrains Mono',monospace" }}>{datasetName}</span>
                <span style={{ fontSize: "11px", color: "#69748a" }}>{datasetMeta}</span>
              </span>
            </div>
          </div>
        </div>
      )}

      {analyzing && (
        <AgentRow>
          <div style={{ background: "#fff", border: "1px solid #e5ebf4", borderRadius: "3px 14px 14px 14px", padding: "14px 16px" }}>
            <div style={{ height: "7px", width: "200px", maxWidth: "60%", borderRadius: "5px", marginBottom: "11px", background: "linear-gradient(90deg,#eef2f9 0%,#dbe2fb 40%,#eef2f9 80%)", backgroundSize: "220% 100%", animation: "lc-shimmer 1.3s linear infinite" }} />
            <div style={{ fontSize: "13.5px", color: "#69748a" }}>Reading examples and projecting candidate concept directions…</div>
          </div>
        </AgentRow>
      )}

      {/* concepts */}
      {si >= 1 && (
        <AgentRow>
          <Bubble>This looks like an automated <b style={{ fontWeight: 600, color: "#1b2542" }}>answer grader</b> (JorGPT-style). I extracted 5 concept directions worth monitoring for silent drift during fine-tuning. Toggle any you don’t want, or add your own.</Bubble>
          <div style={{ background: "#fff", border: "1px solid #e5ebf4", borderRadius: "14px", padding: "8px", display: "flex", flexDirection: "column", gap: "6px" }}>
            {concepts.map((c, i) => (
              <div key={c.name} style={{ display: "flex", gap: "13px", alignItems: "flex-start", padding: "13px 14px", borderRadius: "11px", border: "1px solid " + (c.on ? "#d4dcf6" : "#e5ebf4"), background: c.on ? "#f7f9fd" : "#fbfcfe", opacity: c.on ? 1 : 0.55, transition: "opacity .15s, border-color .15s" }}>
                <button onClick={() => toggleConcept(i)} style={{ flex: "none", width: "38px", height: "22px", borderRadius: "20px", border: "none", cursor: "pointer", padding: "2px", display: "flex", alignItems: "center", justifyContent: c.on ? "flex-end" : "flex-start", background: c.on ? "#2f43e0" : "#cfd8e8", transition: "background .15s" }}>
                  <span style={{ width: "18px", height: "18px", borderRadius: "50%", background: "#fff", boxShadow: "0 1px 2px rgba(20,32,64,0.25)" }} />
                </button>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "4px", flexWrap: "wrap" }}>
                    <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "13px", fontWeight: 500, color: "#1b2542" }}>{c.name}</span>
                    <span style={{ fontSize: "10px", fontWeight: 600, padding: "2px 7px", borderRadius: "20px", color: c.conf == null ? "#838fa4" : c.conf >= 0.9 ? "#2f9e7d" : "#1f9e86", background: c.conf == null ? "#eef3fa" : "#e3f4ee" }}>{c.conf == null ? "custom" : "signal " + c.conf.toFixed(2)}</span>
                  </div>
                  <div style={{ fontSize: "12.5px", color: "#69748a", lineHeight: 1.5 }}>{c.desc}</div>
                </div>
              </div>
            ))}
            <div style={{ display: "flex", gap: "8px", padding: "6px 6px 4px" }}>
              <input value={newConcept} onChange={(e) => setNewConcept(e.target.value)} placeholder="Add a concept… e.g. off_topic_drift" style={{ flex: 1, padding: "9px 12px", borderRadius: "9px", border: "1px solid #e2e9f3", background: "#f8fafe", font: "inherit", fontSize: "12.5px", fontFamily: "'JetBrains Mono',monospace", color: "#283353" }} />
              <button className="hv-secondary" onClick={addConcept} style={{ padding: "9px 15px", borderRadius: "9px", border: "1px solid #d3dbeb", background: "#fff", color: "#48546e", font: "inherit", fontSize: "12.5px", fontWeight: 600, cursor: "pointer" }}>Add</button>
            </div>
          </div>
          {si === 1 && (
            <div style={{ display: "flex", justifyContent: "flex-end" }}>
              <button className="hv-primary" onClick={() => setExpStage("model")} style={primaryBtn}>Looks right — choose a model →</button>
            </div>
          )}
        </AgentRow>
      )}

      {/* model picker */}
      {si >= 2 && (
        <>
          <UserBubble>Monitor these {selConcepts.length} concepts</UserBubble>
          <AgentRow>
            <Bubble>Good set. Which base model should I fine-tune?</Bubble>
            <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
              {MODEL_CARDS.map((m) => {
                const sel = model === m.key;
                return (
                  <button key={m.key} className="hv-bd-blue" onClick={() => selectModel(m.key)} style={{ display: "flex", alignItems: "flex-start", gap: "12px", padding: "15px 16px", borderRadius: "12px", cursor: "pointer", textAlign: "left", width: "100%", font: "inherit", border: "1px solid " + (sel ? "#2f43e0" : "#e5ebf4"), background: sel ? "#f4f6fe" : "#fff", boxShadow: sel ? "0 0 0 1px #2f43e0" : "none", transition: "border-color .15s" }}>
                    <span style={{ flex: "none", marginTop: "1px", width: "18px", height: "18px", borderRadius: "50%", border: "2px solid " + (sel ? "#2f43e0" : "#cfd8e8"), display: "flex", alignItems: "center", justifyContent: "center" }}>
                      <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: sel ? "#2f43e0" : "transparent" }} />
                    </span>
                    <span style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: "3px" }}>
                      <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <span style={{ fontWeight: 600, fontSize: "14px", color: sel ? "#1b2542" : "#283353" }}>{m.label}</span>
                        {m.recommended && <span style={{ fontSize: "10px", fontWeight: 600, color: "#2f43e0", background: "#e7eafb", padding: "2px 7px", borderRadius: "20px" }}>recommended</span>}
                      </span>
                      <span style={{ fontSize: "12px", color: "#69748a" }}>{m.sub}</span>
                    </span>
                  </button>
                );
              })}
            </div>
          </AgentRow>
        </>
      )}

      {/* lora */}
      {si >= 3 && (
        <>
          <UserBubble mono>{modelLabel}</UserBubble>
          <AgentRow>
            <Bubble>I’ll use the r16 LoRA recipe — 3 epochs, ~10 checkpoints, projection + probe monitor every 25 steps. Keep it, or customize.</Bubble>
            <div style={{ background: "#fff", border: "1px solid #e5ebf4", borderRadius: "14px", padding: "16px" }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "12px" }}>
                <div style={{ minWidth: 0 }}>
                  <div style={{ fontSize: "13px", fontWeight: 600, color: "#1b2542" }}>LoRA recipe</div>
                  <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "12px", color: "#69748a", marginTop: "3px" }}>{loraSummary}</div>
                </div>
                <button className="hv-secondary" onClick={() => setLoraOpen((o) => !o)} style={{ flex: "none", padding: "8px 14px", borderRadius: "9px", border: "1px solid #d3dbeb", background: "#fff", color: "#48546e", font: "inherit", fontSize: "12.5px", fontWeight: 600, cursor: "pointer" }}>{loraOpen ? "Hide" : "Customize"}</button>
              </div>
              {loraOpen && (
                <div style={{ marginTop: "15px", paddingTop: "15px", borderTop: "1px solid #eef2f9", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "15px 18px" }}>
                  <div><label style={labelStyle}>Rank (r) · <span style={monoBlue}>{lora.r}</span></label><input type="range" min="4" max="64" step="4" value={lora.r} onChange={(e) => setLoraField("r", parseInt(e.target.value, 10))} style={{ width: "100%", accentColor: "#2f43e0", cursor: "pointer" }} /></div>
                  <div><label style={labelStyle}>Alpha · <span style={monoBlue}>{lora.alpha}</span></label><input type="range" min="8" max="128" step="8" value={lora.alpha} onChange={(e) => setLoraField("alpha", parseInt(e.target.value, 10))} style={{ width: "100%", accentColor: "#2f43e0", cursor: "pointer" }} /></div>
                  <div><label style={labelStyle}>Dropout · <span style={monoBlue}>{lora.dropout.toFixed(2)}</span></label><input type="range" min="0" max="0.2" step="0.01" value={lora.dropout} onChange={(e) => setLoraField("dropout", parseFloat(e.target.value))} style={{ width: "100%", accentColor: "#2f43e0", cursor: "pointer" }} /></div>
                  <div><label style={labelStyle}>Epochs · <span style={monoBlue}>{lora.epochs}</span></label><input type="range" min="1" max="6" step="1" value={lora.epochs} onChange={(e) => setLoraField("epochs", parseInt(e.target.value, 10))} style={{ width: "100%", accentColor: "#2f43e0", cursor: "pointer" }} /></div>
                  <div><label style={labelStyle}>Learning rate</label><select value={lora.lr} onChange={(e) => setLoraField("lr", e.target.value)} style={selectStyle}><option value="5.0e-5">5.0e-5</option><option value="1.0e-4">1.0e-4</option><option value="2.0e-4">2.0e-4</option></select></div>
                  <div><label style={labelStyle}>Batch size</label><select value={lora.batch} onChange={(e) => setLoraField("batch", parseInt(e.target.value, 10))} style={selectStyle}><option value="4">4</option><option value="8">8</option><option value="16">16</option></select></div>
                  <div style={{ gridColumn: "1 / -1" }}><label style={labelStyle}>Target modules <span style={{ color: "#98a2b3", fontWeight: 400 }}>(set by base model)</span></label><div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "12px", color: "#283353", background: "#f8fafe", border: "1px solid #e2e9f3", borderRadius: "9px", padding: "9px 11px" }}>{modelTargets}</div></div>
                </div>
              )}
            </div>
            {si === 3 && (
              <div style={{ display: "flex", justifyContent: "flex-end" }}>
                <button className="hv-primary" onClick={() => setExpStage("review")} style={primaryBtn}>{loraOpen ? "Apply & continue →" : "Use defaults →"}</button>
              </div>
            )}
          </AgentRow>
        </>
      )}

      {/* review */}
      {si >= 4 && (
        <>
          <UserBubble>Use this recipe</UserBubble>
          <AgentRow>
            <Bubble>Here’s your experiment. I’ll train, monitor every checkpoint for the concepts above, and run the capability + safety battery. Launch when ready.</Bubble>
            <div style={{ background: "#fff", border: "1px solid #e5ebf4", borderRadius: "14px", overflow: "hidden" }}>
              {[
                ["Domain", <span style={{ fontSize: "13.5px", color: "#1b2542", fontWeight: 600 }}>education</span>],
                ["Dataset", <span style={{ fontSize: "13px", color: "#283353", fontFamily: "'JetBrains Mono',monospace" }}>{datasetName} · {datasetMeta}</span>],
                ["Concepts", <span style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>{selConcepts.map((c) => <span key={c.name} style={{ fontSize: "11.5px", fontFamily: "'JetBrains Mono',monospace", color: "#2f43e0", background: "#e7eafb", padding: "3px 8px", borderRadius: "6px" }}>{c.name}</span>)}</span>],
                ["Base model", <span style={{ fontSize: "13.5px", color: "#1b2542", fontWeight: 600 }}>{modelLabel}</span>],
                ["LoRA", <span style={{ fontSize: "13px", color: "#283353", fontFamily: "'JetBrains Mono',monospace" }}>{loraSummary}</span>],
                ["Targets", <span style={{ fontSize: "13px", color: "#283353", fontFamily: "'JetBrains Mono',monospace" }}>{modelTargets}</span>],
                ["Eval", <span style={{ fontSize: "13px", color: "#283353" }}>MMLU-Pro · TruthfulQA · HarmBench · StrongREJECT</span>],
              ].map(([k, v], i, arr) => (
                <div key={k} style={{ display: "flex", padding: "13px 16px", borderBottom: i < arr.length - 1 ? "1px solid #f0f3f9" : "none" }}>
                  <span style={{ width: "118px", flex: "none", fontSize: "11px", letterSpacing: "0.06em", textTransform: "uppercase", color: "#98a2b3", paddingTop: "2px" }}>{k}</span>
                  {v}
                </div>
              ))}
            </div>
            {si === 4 && (
              <div style={{ display: "flex", justifyContent: "flex-end" }}>
                <button className="hv-primary" onClick={launchExperiment} style={{ display: "flex", alignItems: "center", gap: "9px", padding: "12px 22px", borderRadius: "11px", border: "none", background: "#2f43e0", color: "#fff", font: "inherit", fontSize: "14px", fontWeight: 600, cursor: "pointer", boxShadow: "0 1px 2px rgba(35,52,194,0.25)" }}>
                  <svg width="16" height="16" viewBox="0 0 18 18" fill="currentColor"><polygon points="5,3.2 15,9 5,14.8" /></svg>Launch experiment
                </button>
              </div>
            )}
          </AgentRow>
        </>
      )}

      {/* launched */}
      {si >= 5 && (
        <>
          <UserBubble>Launch</UserBubble>
          <AgentRow>
            <div style={{ border: "1px solid #d4dcf6", background: "#f4f6fe", borderRadius: "3px 14px 14px 14px", padding: "16px", display: "flex", gap: "14px", alignItems: "center", flexWrap: "wrap" }}>
              <span style={{ flex: "none", width: "34px", height: "34px", borderRadius: "50%", background: "#e3f4ee", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#2f9e7d" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"><polyline points="5,12 10,17 19,7" /></svg>
              </span>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontWeight: 600, fontSize: "14px", color: "#1b2542" }}>Experiment launched — <span style={{ fontFamily: "'JetBrains Mono',monospace" }}>{lastRunId}</span></div>
                <div style={{ fontSize: "13px", color: "#69748a", marginTop: "2px" }}>Training on GPU 2, evaluating on GPU 1. Monitoring {selConcepts.length} concept directions per checkpoint.</div>
              </div>
              <button className="hv-primary" onClick={() => onLaunched && onLaunched({ dataset: "education", model: model === "apertus" ? "apertus-8b" : "qwen-7b", view: "dashboard" })} style={{ flex: "none", padding: "10px 16px", borderRadius: "10px", border: "none", background: "#2f43e0", color: "#fff", font: "inherit", fontSize: "13px", fontWeight: 600, cursor: "pointer" }}>Open dashboard →</button>
            </div>
          </AgentRow>
        </>
      )}

      {/* free-form chat */}
      {freeMsgs.map((m, i) =>
        m.role === "user" ? <UserBubble key={i}>{m.text}</UserBubble> : <AgentRow key={i}><Bubble>{m.text}</Bubble></AgentRow>
      )}

      <Composer value={draft} onChange={(e) => setDraft(e.target.value)} onSend={sendFree} />
    </div>
  );
}
