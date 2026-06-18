import { useState } from "react";
import { Spline } from "lucide-react";
import Artifact, { ArtifactGlyph } from "../Artifact.jsx";

// Concept artifact: inline shows compact concept rows (signal badge + optional toggle +
// one-line description); expanded opens the full descriptions, one detailed card each.
// `concepts` = [{ name, conf?, desc, on?, layer?, auroc?, validated? }].
// When `onToggle`/`onAdd` are supplied the inline preview is interactive (design flow);
// otherwise it's a read-only registry view (a completed run).

function SignalBadge({ conf }) {
  // "custom" concepts: a dry, borderless grey tag. Probed concepts keep the green signal pill.
  if (conf == null) {
    return (
      <span style={{ fontSize: "9.5px", fontWeight: 500, letterSpacing: "0.07em", textTransform: "uppercase", color: "#9aa3b2", whiteSpace: "nowrap" }}>
        custom
      </span>
    );
  }
  return (
    <span style={{ fontSize: "10px", fontWeight: 600, padding: "2px 7px", borderRadius: "20px", whiteSpace: "nowrap", color: "#1f9e86", background: "#e3f4ee" }}>
      signal {conf.toFixed(2)}
    </span>
  );
}

function Toggle({ on, onClick }) {
  return (
    <button onClick={onClick} style={{ flex: "none", width: "38px", height: "22px", borderRadius: "20px", border: "none", cursor: "pointer", padding: "2px", display: "flex", alignItems: "center", justifyContent: on ? "flex-end" : "flex-start", background: on ? "#2f43e0" : "#cfd8e8", transition: "background .15s" }}>
      <span style={{ width: "18px", height: "18px", borderRadius: "50%", background: "#fff", boxShadow: "0 1px 2px rgba(20,32,64,0.25)" }} />
    </button>
  );
}

export default function ConceptArtifact({ concepts, onToggle, onAdd, title = "Concept directions" }) {
  const [newConcept, setNewConcept] = useState("");
  const interactive = Boolean(onToggle);
  const active = concepts.filter((c) => c.on !== false);

  function add() {
    const t = newConcept.trim(); if (!t || !onAdd) return;
    onAdd(t); setNewConcept("");
  }

  const glyph = <ArtifactGlyph><Spline size={16} strokeWidth={1.8} /></ArtifactGlyph>;
  const subtitle = `${active.length} of ${concepts.length} monitored`;

  // inline: compact rows
  const preview = (
    <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
      {concepts.map((c, i) => {
        const on = c.on !== false;
        return (
          <div key={c.name} style={{ display: "flex", gap: "11px", alignItems: "flex-start", padding: "10px 12px", borderRadius: "10px", border: "1px solid " + (on ? "#d4dcf6" : "#e5ebf4"), background: on ? "#f7f9fd" : "#fbfcfe", opacity: on ? 1 : 0.55, transition: "opacity .15s, border-color .15s" }}>
            {interactive && <Toggle on={on} onClick={() => onToggle(i)} />}
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "3px", flexWrap: "wrap" }}>
                <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "12.5px", fontWeight: 500, color: "#1b2542" }}>{c.name}</span>
                <SignalBadge conf={c.conf} />
              </div>
              <div style={{ fontSize: "12px", color: "#69748a", lineHeight: 1.5, display: "-webkit-box", WebkitLineClamp: 1, WebkitBoxOrient: "vertical", overflow: "hidden" }}>{c.desc}</div>
            </div>
          </div>
        );
      })}
      {interactive && onAdd && (
        <div style={{ display: "flex", gap: "8px", paddingTop: "2px" }}>
          <input value={newConcept} onChange={(e) => setNewConcept(e.target.value)} onKeyDown={(e) => e.key === "Enter" && add()} placeholder="Add a concept… e.g. off_topic_drift" style={{ flex: 1, padding: "9px 12px", borderRadius: "9px", border: "1px solid #e2e9f3", background: "#f8fafe", font: "inherit", fontSize: "12.5px", fontFamily: "'JetBrains Mono',monospace", color: "#283353" }} />
          <button className="hv-secondary" onClick={add} style={{ padding: "9px 15px", borderRadius: "9px", border: "1px solid #d3dbeb", background: "#fff", color: "#48546e", font: "inherit", fontSize: "12.5px", fontWeight: 600, cursor: "pointer" }}>Add</button>
        </div>
      )}
    </div>
  );

  // expanded: full descriptions + any vector stats
  const expanded = (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      {concepts.map((c) => {
        const on = c.on !== false;
        const stats = [
          c.layer != null && ["layer", `L${c.layer}`],
          c.auroc != null && ["probe AUROC", c.auroc.toFixed(2)],
          c.validated != null && ["validated", c.validated ? "yes" : "no"],
        ].filter(Boolean);
        return (
          <div key={c.name} style={{ border: "1px solid " + (on ? "#e1e8f6" : "#eef2f9"), borderRadius: "12px", padding: "16px 18px", background: on ? "#fff" : "#fbfcfe", opacity: on ? 1 : 0.6 }}>
            <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "8px", flexWrap: "wrap" }}>
              <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "15px", fontWeight: 600, color: "#1b2542" }}>{c.name}</span>
              <SignalBadge conf={c.conf} />
              {!on && <span style={{ fontSize: "10.5px", color: "#a6aebe" }}>not monitored</span>}
            </div>
            <div style={{ fontSize: "13.5px", color: "#48546e", lineHeight: 1.65 }}>{c.desc}</div>
            {stats.length > 0 && (
              <div style={{ display: "flex", gap: "22px", marginTop: "12px", flexWrap: "wrap" }}>
                {stats.map(([k, v]) => (
                  <span key={k} style={{ fontSize: "12px", color: "#838fa4" }}>{k} <b style={{ color: "#283353", fontFamily: "'JetBrains Mono',monospace", marginLeft: "4px" }}>{v}</b></span>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );

  return (
    <Artifact glyph={glyph} title={title} subtitle={subtitle} expanded={expanded} expandedSubtitle="full descriptions">
      {preview}
    </Artifact>
  );
}
