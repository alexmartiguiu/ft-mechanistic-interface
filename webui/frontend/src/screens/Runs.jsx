import { useEffect, useRef, useState } from "react";
import { clock, logColor, CONFIGS } from "../data/demo.js";

const STATUS = {
  running: { label: "running", color: "#1f9e86", bg: "#e2f3ef" },
  done: { label: "done", color: "#2f9e7d", bg: "#e3f4ee" },
  failed: { label: "failed", color: "#d8483a", bg: "#fbe7e4" },
};
const GPU_OPTIONS = ["0", "1", "2", "3", "4", "5", "6", "7"];

export default function Runs({ runs, startRun, stopRun }) {
  const [cmd, setCmd] = useState("run");
  const [app, setApp] = useState("therapist");
  const [gpu, setGpu] = useState("3");
  const [extra, setExtra] = useState("");
  const logRefs = useRef({});

  // keep running logs pinned to the bottom as lines stream in
  useEffect(() => {
    runs.forEach((r) => {
      const el = logRefs.current[r.id];
      if (el && r.status === "running") el.scrollTop = el.scrollHeight;
    });
  });

  const cmdStyle = (c) => {
    const active = cmd === c;
    return {
      flex: 1, padding: "9px", borderRadius: "8px", border: "none", cursor: "pointer", font: "inherit",
      fontSize: "13px", fontWeight: active ? 600 : 500, fontFamily: "'JetBrains Mono',monospace",
      transition: "background .15s, color .15s",
      background: active ? "#ffffff" : "transparent",
      color: active ? "#2031c4" : "#838fa4",
      boxShadow: active ? "0 1px 2px rgba(20,32,64,0.06)" : "none",
    };
  };
  const selectStyle = { width: "100%", marginBottom: "16px", padding: "10px 12px", borderRadius: "10px", border: "1px solid #d3dbeb", background: "#f7f9fd", color: "#15203c", font: "inherit", fontSize: "13.5px", cursor: "pointer" };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: "22px", alignItems: "start" }}>
      {/* launcher */}
      <div style={{ background: "#ffffff", border: "1px solid #e5ebf4", borderRadius: "16px", padding: "20px", boxShadow: "0 1px 2px rgba(20,32,64,0.03)", position: "sticky", top: 0 }}>
        <h3 style={{ margin: "0 0 16px", fontFamily: "'Hanken Grotesk',sans-serif", fontWeight: 300, fontSize: "18px", color: "#15203c" }}>Launch a job</h3>
        <label style={{ display: "block", fontSize: "12px", fontWeight: 600, color: "#48546e", marginBottom: "8px" }}>Command</label>
        <div style={{ display: "flex", gap: "6px", marginBottom: "16px", background: "#eef3fa", padding: "4px", borderRadius: "11px" }}>
          {["run", "train", "eval"].map((c) => (
            <button key={c} onClick={() => setCmd(c)} style={cmdStyle(c)}>{c}</button>
          ))}
        </div>
        <label style={{ display: "block", fontSize: "12px", fontWeight: 600, color: "#48546e", marginBottom: "7px" }}>App</label>
        <select value={app} onChange={(e) => setApp(e.target.value)} style={selectStyle}>
          {CONFIGS.apps.map((a) => <option key={a} value={a}>{a}</option>)}
        </select>
        <div style={{ display: "flex", gap: "12px", marginBottom: "16px" }}>
          <div style={{ flex: "none", width: "88px" }}>
            <label style={{ display: "block", fontSize: "12px", fontWeight: 600, color: "#48546e", marginBottom: "7px" }}>GPU</label>
            <select value={gpu} onChange={(e) => setGpu(e.target.value)} style={{ ...selectStyle, marginBottom: 0, fontFamily: "'JetBrains Mono',monospace" }}>
              {GPU_OPTIONS.map((g) => <option key={g} value={g}>{g}</option>)}
            </select>
          </div>
          <div style={{ flex: 1 }}>
            <label style={{ display: "block", fontSize: "12px", fontWeight: 600, color: "#48546e", marginBottom: "7px" }}>Extra flags</label>
            <input value={extra} onChange={(e) => setExtra(e.target.value)} placeholder="--skip-eval" style={{ width: "100%", padding: "10px 12px", borderRadius: "10px", border: "1px solid #d3dbeb", background: "#f7f9fd", color: "#15203c", font: "inherit", fontSize: "13px", fontFamily: "'JetBrains Mono',monospace" }} />
          </div>
        </div>
        <button className="hv-primary" onClick={() => startRun(cmd, app, gpu, extra)} style={{ width: "100%", padding: "12px", borderRadius: "11px", border: "none", background: "#2f43e0", color: "#fff", font: "inherit", fontSize: "14px", fontWeight: 600, cursor: "pointer", boxShadow: "0 1px 2px rgba(35,52,194,0.25)" }}>Launch job</button>
        <div style={{ marginTop: "13px", paddingTop: "13px", borderTop: "1px solid #eef2f9", fontSize: "11px", color: "#a6aebe", lineHeight: 1.5 }}>Runs are session-only — the list resets if the backend restarts. Launch existing configs; there's no new-experiment authoring here.</div>
      </div>

      {/* run list */}
      <div style={{ display: "flex", flexDirection: "column", gap: "16px", minWidth: 0 }}>
        {runs.length === 0 && (
          <div style={{ border: "1px dashed #ccd6ea", borderRadius: "16px", padding: "56px 40px", textAlign: "center", backgroundImage: "radial-gradient(circle, rgba(20,32,64,0.045) 0.7px, transparent 1px)", backgroundSize: "6px 6px", backgroundColor: "#f8fafe" }}>
            <div style={{ fontFamily: "'Hanken Grotesk',sans-serif", fontSize: "19px", color: "#283353", marginBottom: "7px" }}>No jobs this session</div>
            <div style={{ fontSize: "13.5px", color: "#838fa4" }}>Launch a job to stream its logs live here.</div>
          </div>
        )}

        {runs.map((r) => {
          const sc = STATUS[r.status] || STATUS.failed;
          return (
            <div key={r.id} style={{ background: "#ffffff", border: "1px solid #e5ebf4", borderRadius: "14px", overflow: "hidden", boxShadow: "0 1px 2px rgba(20,32,64,0.03)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "12px", padding: "14px 18px", borderBottom: "1px solid #eef2f9", flexWrap: "wrap" }}>
                <span style={{ display: "inline-flex", alignItems: "center", gap: "7px", fontSize: "12px", fontWeight: 600, padding: "4px 10px", borderRadius: "20px", color: sc.color, background: sc.bg }}>
                  <span style={{ width: "7px", height: "7px", borderRadius: "50%", background: sc.color, animation: r.status === "running" ? "lc-pulse 1.2s ease-in-out infinite" : "none" }} />{sc.label}
                </span>
                <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "13px", fontWeight: 500, color: "#1b2542" }}>{r.id}</span>
                <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "12px", color: "#a6aebe" }}>{r.cmd + " · " + r.app}</span>
                <div style={{ flex: 1 }} />
                <span style={{ fontSize: "11.5px", color: "#a6aebe", fontFamily: "'JetBrains Mono',monospace" }}>{clock(r.started)}</span>
                {r.status === "running" && (
                  <button className="hv-stop" onClick={() => stopRun(r.id)} style={{ padding: "6px 13px", borderRadius: "8px", border: "1px solid #c2ccf4", background: "#e9edfc", color: "#2031c4", font: "inherit", fontSize: "12px", fontWeight: 600, cursor: "pointer" }}>Stop</button>
                )}
              </div>
              <div style={{ padding: "9px 18px", background: "#f7f9fd", borderBottom: "1px solid #eef2f9" }}>
                <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "11.5px", color: "#838fa4", wordBreak: "break-all" }}>{r.argv}</span>
              </div>
              <div ref={(el) => { if (el) logRefs.current[r.id] = el; }} style={{ background: "#f8fafe", maxHeight: "300px", overflowY: "auto", padding: "14px 18px", fontFamily: "'JetBrains Mono',monospace", fontSize: "12px", lineHeight: 1.65 }}>
                {r.lines.map((t, i) => (
                  <div key={i} style={{ display: "flex", gap: "12px" }}>
                    <span style={{ flex: "none", color: "#b7c1d6", userSelect: "none", width: "26px", textAlign: "right" }}>{i + 1}</span>
                    <span style={{ color: logColor(t), whiteSpace: "pre-wrap", wordBreak: "break-word" }}>{t}</span>
                  </div>
                ))}
                {r.status === "running" && (
                  <div style={{ display: "flex", gap: "12px" }}>
                    <span style={{ flex: "none", width: "26px" }} />
                    <span style={{ color: "#2f43e0", animation: "lc-blink 1s steps(1) infinite" }}>▋</span>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
