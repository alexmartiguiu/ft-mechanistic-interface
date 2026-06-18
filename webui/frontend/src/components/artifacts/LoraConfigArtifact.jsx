import { useState } from "react";
import { SlidersHorizontal } from "lucide-react";
import Artifact, { ArtifactGlyph } from "../Artifact.jsx";

// LoRA recipe artifact: inline shows the one-line summary with an inline "Customize"
// drawer (sliders/selects); expanded opens the full resolved config as a yaml-like
// key/value sheet (the file that would be written to configs/lora/*.yaml).
// `lora` = { r, alpha, dropout, epochs, lr, batch }; `onField(key, value)` edits it.

const labelStyle ={ display: "block", fontSize: "11.5px", fontWeight: 600, color: "#48546e", marginBottom: "7px" };
const monoBlue = { color: "#2f43e0", fontFamily: "'JetBrains Mono',monospace" };
const selectStyle = { width: "100%", padding: "9px 11px", borderRadius: "9px", border: "1px solid #d3dbeb", background: "#f8fafe", font: "inherit", fontSize: "12.5px", fontFamily: "'JetBrains Mono',monospace", color: "#283353", cursor: "pointer" };

export default function LoraConfigArtifact({ lora, onField, modelLabel, modelId, targets }) {
  const [open, setOpen] = useState(false);
  const editable = Boolean(onField);
  const summary = `r${lora.r} · α${lora.alpha} · ${lora.epochs} epochs · lr ${lora.lr} · batch ${lora.batch}`;

  const glyph = <ArtifactGlyph><SlidersHorizontal size={16} strokeWidth={1.8} /></ArtifactGlyph>;

  // expanded: the full resolved recipe as a yaml-like sheet
  const rows = [
    ["model_id", modelId || modelLabel],
    ["dtype", "bfloat16"],
    ["lora.r", lora.r],
    ["lora.alpha", lora.alpha],
    ["lora.dropout", lora.dropout.toFixed(2)],
    ["lora.target_modules", targets],
    ["optim.lr", lora.lr],
    ["optim.epochs", lora.epochs],
    ["optim.batch_size", lora.batch],
    ["optim.warmup_ratio", "0.03"],
    ["optim.max_seq_len", "2048"],
    ["checkpoint.n_checkpoints", "10"],
    ["checkpoint.monitor_every_steps", "25"],
    ["checkpoint.log_every_steps", "10"],
  ];
  const expanded = (
    <div style={{ border: "1px solid #e5ebf4", borderRadius: "12px", overflow: "hidden", fontFamily: "'JetBrains Mono',monospace" }}>
      {rows.map(([k, v], i) => (
        <div key={k} style={{ display: "flex", gap: "16px", padding: "10px 16px", fontSize: "12.5px", borderBottom: i < rows.length - 1 ? "1px solid #f0f3f9" : "none", background: i % 2 ? "#fbfcfe" : "#fff" }}>
          <span style={{ width: "230px", flex: "none", color: "#838fa4" }}>{k}</span>
          <span style={{ color: "#1b2542", wordBreak: "break-word" }}>{String(v)}</span>
        </div>
      ))}
    </div>
  );

  return (
    <Artifact glyph={glyph} title="LoRA recipe" subtitle={summary} expanded={expanded} expandedTitle="LoRA recipe" expandedSubtitle={modelLabel}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "12px" }}>
        <div style={{ minWidth: 0, fontSize: "12.5px", color: "#69748a" }}>
          3 epochs · ~10 checkpoints · projection + probe monitor every 25 steps
        </div>
        {editable && (
          <button className="hv-secondary" onClick={() => setOpen((o) => !o)} style={{ flex: "none", padding: "8px 14px", borderRadius: "9px", border: "1px solid #d3dbeb", background: "#fff", color: "#48546e", font: "inherit", fontSize: "12.5px", fontWeight: 600, cursor: "pointer" }}>{open ? "Hide" : "Customize"}</button>
        )}
      </div>
      {editable && open && (
        <div style={{ marginTop: "15px", paddingTop: "15px", borderTop: "1px solid #eef2f9", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "15px 18px" }}>
          <div><label style={labelStyle}>Rank (r) · <span style={monoBlue}>{lora.r}</span></label><input type="range" min="4" max="64" step="4" value={lora.r} onChange={(e) => onField("r", parseInt(e.target.value, 10))} style={{ width: "100%", accentColor: "#2f43e0", cursor: "pointer" }} /></div>
          <div><label style={labelStyle}>Alpha · <span style={monoBlue}>{lora.alpha}</span></label><input type="range" min="8" max="128" step="8" value={lora.alpha} onChange={(e) => onField("alpha", parseInt(e.target.value, 10))} style={{ width: "100%", accentColor: "#2f43e0", cursor: "pointer" }} /></div>
          <div><label style={labelStyle}>Dropout · <span style={monoBlue}>{lora.dropout.toFixed(2)}</span></label><input type="range" min="0" max="0.2" step="0.01" value={lora.dropout} onChange={(e) => onField("dropout", parseFloat(e.target.value))} style={{ width: "100%", accentColor: "#2f43e0", cursor: "pointer" }} /></div>
          <div><label style={labelStyle}>Epochs · <span style={monoBlue}>{lora.epochs}</span></label><input type="range" min="1" max="6" step="1" value={lora.epochs} onChange={(e) => onField("epochs", parseInt(e.target.value, 10))} style={{ width: "100%", accentColor: "#2f43e0", cursor: "pointer" }} /></div>
          <div><label style={labelStyle}>Learning rate</label><select value={lora.lr} onChange={(e) => onField("lr", e.target.value)} style={selectStyle}><option value="5.0e-5">5.0e-5</option><option value="1.0e-4">1.0e-4</option><option value="2.0e-4">2.0e-4</option></select></div>
          <div><label style={labelStyle}>Batch size</label><select value={lora.batch} onChange={(e) => onField("batch", parseInt(e.target.value, 10))} style={selectStyle}><option value="4">4</option><option value="8">8</option><option value="16">16</option></select></div>
          <div style={{ gridColumn: "1 / -1" }}><label style={labelStyle}>Target modules <span style={{ color: "#98a2b3", fontWeight: 400 }}>(set by base model)</span></label><div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "12px", color: "#283353", background: "#f8fafe", border: "1px solid #e2e9f3", borderRadius: "9px", padding: "9px 11px" }}>{targets}</div></div>
        </div>
      )}
    </Artifact>
  );
}
