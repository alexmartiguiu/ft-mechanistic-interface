import { useRef, useState } from "react";
import { VECTORS, APPS, steerEntry, DOG_PATH } from "../data/demo.js";

// modelWarm / onWarm are lifted to App so the page header can show "model warm/idle".
export default function LiveSteering({ modelWarm, onWarm }) {
  const [steerDomain, setSteerDomain] = useState("therapist");
  const [steerConcept, setSteerConcept] = useState("crisis_minimization");
  const [prompt, setPrompt] = useState("I feel completely hopeless and I don't see the point in anything anymore.");
  const [coef, setCoef] = useState(8);
  const [system, setSystem] = useState("You are a helpful assistant.");
  const [maxTokens, setMaxTokens] = useState(300);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [inFlight, setInFlight] = useState(false);
  const [phase, setPhase] = useState(null); // 'warming' | 'generating'
  const [result, setResult] = useState(null);
  const timer = useRef(null);

  const curDom = VECTORS.find((d) => d.domain === steerDomain) || VECTORS[0];
  const conceptOptions = curDom.concepts.map((c) => c.name);

  const coefMode =
    coef >= 1.5 ? { label: "Amplifying", color: "#2031c4", bg: "#e6eafc" }
    : coef <= -1.5 ? { label: "Suppressing", color: "#4659c9", bg: "#e7eafb" }
    : { label: "Neutral", color: "#838fa4", bg: "#eef3fa" };

  const genDisabled = inFlight || !prompt.trim();

  function onGenerate() {
    if (genDisabled) return;
    const warming = !modelWarm;
    setInFlight(true);
    setPhase(warming ? "warming" : "generating");
    setResult(null);
    clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      const entry = steerEntry(steerConcept);
      const steered = coef >= 1.5 ? entry.up : coef <= -1.5 ? entry.down : entry.base;
      const app = APPS.find((a) => a.app === steerDomain);
      const concept = (VECTORS.find((d) => d.domain === steerDomain) || { concepts: [] }).concepts.find((c) => c.name === steerConcept) || { layer: 16 };
      setInFlight(false);
      setPhase(null);
      onWarm();
      setResult({ base: entry.base, steered, coef, layer: concept.layer, model: app ? app.model : "Qwen/Qwen2.5-7B-Instruct", concept: steerConcept });
    }, warming ? 2600 : 1100);
  }

  const coefValueStr = (coef > 0 ? "+" : "") + coef;
  const steeredTag = result
    ? result.coef >= 1.5 ? "concept amplified · coef " + (result.coef > 0 ? "+" : "") + result.coef
      : result.coef <= -1.5 ? "concept suppressed · coef " + result.coef
      : "coef " + result.coef
    : "";

  const selectStyle = { width: "100%", marginBottom: "16px", padding: "10px 12px", borderRadius: "10px", border: "1px solid #d3dbeb", background: "#f7f9fd", color: "#15203c", font: "inherit", fontSize: "13.5px", cursor: "pointer" };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "336px 1fr", gap: "22px", alignItems: "start" }}>
      {/* controls */}
      <div style={{ background: "#ffffff", border: "1px solid #e5ebf4", borderRadius: "16px", padding: "20px", boxShadow: "0 1px 2px rgba(20,32,64,0.03)", position: "sticky", top: 0 }}>
        <label style={{ display: "block", fontSize: "12px", fontWeight: 600, color: "#48546e", marginBottom: "7px" }}>Domain</label>
        <select value={steerDomain} onChange={(e) => { const dom = VECTORS.find((d) => d.domain === e.target.value); setSteerDomain(e.target.value); setSteerConcept(dom.concepts[0].name); }} style={selectStyle}>
          {VECTORS.map((d) => <option key={d.domain} value={d.domain}>{d.domain}</option>)}
        </select>

        <label style={{ display: "block", fontSize: "12px", fontWeight: 600, color: "#48546e", marginBottom: "7px" }}>Concept</label>
        <select value={steerConcept} onChange={(e) => setSteerConcept(e.target.value)} style={{ ...selectStyle, fontFamily: "'JetBrains Mono',monospace" }}>
          {conceptOptions.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>

        <label style={{ display: "block", fontSize: "12px", fontWeight: 600, color: "#48546e", marginBottom: "7px" }}>Prompt</label>
        <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={4} style={{ width: "100%", marginBottom: "18px", padding: "11px 12px", borderRadius: "10px", border: "1px solid #d3dbeb", background: "#f7f9fd", color: "#15203c", font: "inherit", fontSize: "13.5px", lineHeight: 1.5, resize: "vertical" }} />

        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: "8px" }}>
          <label style={{ fontSize: "12px", fontWeight: 600, color: "#48546e" }}>Steering coefficient</label>
          <span style={{ fontSize: "12px", fontWeight: 600, padding: "2px 9px", borderRadius: "20px", color: coefMode.color, background: coefMode.bg }}>{coefMode.label}</span>
        </div>
        <input type="range" min="-15" max="15" step="0.5" value={coef} onChange={(e) => setCoef(parseFloat(e.target.value))} style={{ width: "100%", accentColor: "#2f43e0", cursor: "pointer" }} />
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "#a6aebe", fontFamily: "'JetBrains Mono',monospace", marginTop: "4px", marginBottom: "18px" }}>
          <span>−15</span><span style={{ color: "#2f43e0", fontWeight: 500 }}>{coefValueStr}</span><span>+15</span>
        </div>

        <button onClick={() => setShowAdvanced((a) => !a)} style={{ display: "flex", alignItems: "center", gap: "6px", background: "none", border: "none", padding: 0, color: "#838fa4", font: "inherit", fontSize: "12.5px", cursor: "pointer", marginBottom: "12px" }}>
          <span style={{ transform: showAdvanced ? "none" : "rotate(-90deg)", transition: "transform .18s ease", display: "inline-flex" }}>
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.8"><polyline points="3,4.5 6,8 9,4.5" /></svg>
          </span>
          Advanced settings
        </button>
        {showAdvanced && (
          <div style={{ borderTop: "1px solid #eef2f9", paddingTop: "14px", marginBottom: "14px" }}>
            <label style={{ display: "block", fontSize: "12px", fontWeight: 600, color: "#48546e", marginBottom: "7px" }}>System prompt</label>
            <textarea value={system} onChange={(e) => setSystem(e.target.value)} rows={2} style={{ width: "100%", marginBottom: "14px", padding: "10px 12px", borderRadius: "10px", border: "1px solid #d3dbeb", background: "#f7f9fd", color: "#15203c", font: "inherit", fontSize: "12.5px", lineHeight: 1.5, resize: "vertical" }} />
            <label style={{ display: "block", fontSize: "12px", fontWeight: 600, color: "#48546e", marginBottom: "7px" }}>Max new tokens · <span style={{ fontFamily: "'JetBrains Mono',monospace", color: "#2f43e0" }}>{maxTokens}</span></label>
            <input type="range" min="64" max="512" step="32" value={maxTokens} onChange={(e) => setMaxTokens(parseInt(e.target.value, 10))} style={{ width: "100%", accentColor: "#2f43e0", cursor: "pointer" }} />
          </div>
        )}

        <button onClick={onGenerate} disabled={genDisabled} style={{
          width: "100%", padding: "12px", borderRadius: "11px", border: "none", font: "inherit", fontSize: "14px", fontWeight: 600,
          cursor: genDisabled ? "not-allowed" : "pointer",
          background: genDisabled ? "#e2d9ca" : "#2f43e0",
          color: genDisabled ? "#a6aebe" : "#fff",
          boxShadow: genDisabled ? "none" : "0 1px 2px rgba(35,52,194,0.25)",
        }}>
          {inFlight ? (phase === "warming" ? "Warming up…" : "Generating…") : "Generate"}
        </button>
        <div style={{ marginTop: "11px", fontSize: "11.5px", color: "#a6aebe", textAlign: "center", lineHeight: 1.4 }}>
          {modelWarm ? "One request at a time · output is deterministic" : "First request loads the model onto the GPU (~10–60s)"}
        </div>
      </div>

      {/* output */}
      <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
        {!inFlight && !result && (
          <div style={{ border: "1px dashed #ccd6ea", borderRadius: "16px", padding: "64px 40px", textAlign: "center", backgroundImage: "radial-gradient(circle, rgba(20,32,64,0.045) 0.7px, transparent 1px)", backgroundSize: "6px 6px", backgroundColor: "#f8fafe" }}>
            <span style={{ display: "inline-flex", marginBottom: "18px" }}>
              <svg viewBox="0 0 104 88" width="64" height="54" fill="#2f43e0" fillRule="evenodd" style={{ opacity: 0.9 }}><path d={DOG_PATH} /></svg>
            </span>
            <div style={{ fontFamily: "'Hanken Grotesk',sans-serif", fontSize: "20px", color: "#283353", marginBottom: "8px" }}>Compare normal vs. steered output</div>
            <div style={{ fontSize: "14px", color: "#838fa4", maxWidth: "420px", margin: "0 auto", lineHeight: 1.5 }}>Pick a concept, type a prompt, and dial the coefficient. hedda generates the model's normal answer and its steered answer side by side.</div>
          </div>
        )}

        {inFlight && (
          <div style={{ border: "1px solid #e5ebf4", borderRadius: "16px", padding: "48px 40px", textAlign: "center", background: "#ffffff" }}>
            <div style={{ height: "8px", width: "220px", maxWidth: "60%", margin: "0 auto 18px", borderRadius: "6px", background: "linear-gradient(90deg,#eef3fa 0%,#f0ddd0 40%,#eef3fa 80%)", backgroundSize: "220% 100%", animation: "lc-shimmer 1.3s linear infinite" }} />
            <div style={{ fontFamily: "'Hanken Grotesk',sans-serif", fontSize: "18px", color: "#283353", marginBottom: "6px" }}>{phase === "warming" ? "Warming up the model…" : "Generating…"}</div>
            <div style={{ fontSize: "13.5px", color: "#838fa4" }}>{phase === "warming" ? "Loading Qwen2.5-7B onto GPU 3. This only happens on the first request." : "Running base and steered passes."}</div>
          </div>
        )}

        {!inFlight && result && (
          <>
            <div style={{ display: "flex", alignItems: "center", gap: "10px", fontSize: "12.5px", color: "#838fa4", flexWrap: "wrap" }}>
              <span style={{ fontFamily: "'JetBrains Mono',monospace", color: "#48546e" }}>{result.model}</span>
              <span style={{ width: "3px", height: "3px", borderRadius: "50%", background: "#b7c1d6" }} />
              <span style={{ fontFamily: "'JetBrains Mono',monospace" }}>layer {result.layer}</span>
              <span style={{ width: "3px", height: "3px", borderRadius: "50%", background: "#b7c1d6" }} />
              <span style={{ fontFamily: "'JetBrains Mono',monospace" }}>coef {(result.coef > 0 ? "+" : "") + result.coef}</span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
              <div style={{ background: "#ffffff", border: "1px solid #e5ebf4", borderRadius: "14px", overflow: "hidden" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "8px", padding: "13px 17px", borderBottom: "1px solid #eef2f9", background: "#f7f9fd" }}>
                  <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: "#98a2b3" }} />
                  <span style={{ fontSize: "13px", fontWeight: 600, color: "#283353" }}>Base output</span>
                  <span style={{ fontSize: "11px", color: "#a6aebe" }}>coef 0 · unmodified</span>
                </div>
                <div style={{ padding: "17px", fontSize: "13.5px", lineHeight: 1.62, color: "#283353", whiteSpace: "pre-wrap" }}>{result.base}</div>
              </div>
              <div style={{ background: "#f4f6fe", border: "1px solid #f0ddd0", borderRadius: "14px", overflow: "hidden", boxShadow: "0 0 0 1px rgba(47,67,224,0.06)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "8px", padding: "13px 17px", borderBottom: "1px solid #d4dcf6", background: "#e9edfc" }}>
                  <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: "#2f43e0" }} />
                  <span style={{ fontSize: "13px", fontWeight: 600, color: "#2031c4" }}>Steered output</span>
                  <span style={{ fontSize: "11px", color: "#7d8ae8" }}>{steeredTag}</span>
                </div>
                <div style={{ padding: "17px", fontSize: "13.5px", lineHeight: 1.62, color: "#283353", whiteSpace: "pre-wrap" }}>{result.steered}</div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
