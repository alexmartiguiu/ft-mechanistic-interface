import { useEffect, useRef, useState } from "react";
import { getSteerCompare } from "../api.js";
import Artifact, { ArtifactGlyph } from "./Artifact.jsx";
import { AgentRow, Bubble } from "./Chat.jsx";

// The mediated (preventive-steering) run block. Instead of replaying the same loss curve, it
// shows the steered adapter's LATENT DRIFT vs the unsteered finetune: per safety axis, the
// base→final projection ⟨h, v̂⟩ (positive = moved toward the trait = bad, negative = away).
// "Improved" = the steered run pushed it more negative on the axes the finetune damaged.
// Diverging bars build on mount; the full table + trio + read live in the expanded view.
const SEAL = "#b5432f";       // worse / toward-trait
const GOOD = "#2f9e7d";       // improved / suppressed
const MUTE = "#b8b2a6";       // the unsteered reference bar

export default function SteeredRun({ domain = "medical", onDone }) {
  const [data, setData] = useState(null);
  const [grown, setGrown] = useState(false);
  const fired = useRef(false);

  useEffect(() => {
    let cancelled = false;
    getSteerCompare(domain).then((d) => {
      if (cancelled) return;
      setData(d);
      setTimeout(() => setGrown(true), 80);           // trigger the grow animation
    }).catch(() => { if (!cancelled) setData({ error: true }); });
    return () => { cancelled = true; };
  }, [domain]);

  useEffect(() => { if (grown && !fired.current) { fired.current = true; onDone && onDone(); } }, [grown, onDone]);

  if (!data) return <AgentRow><Bubble>Running the preventive-steering follow-up — measuring the steered adapter's latent drift…</Bubble></AgentRow>;
  if (data.error) return <AgentRow><Bubble>No steered comparison recorded for {domain}.</Bubble></AgentRow>;

  const maxAbs = Math.max(
    ...data.concepts.flatMap((c) => [Math.abs(c.unsteered), Math.abs(c.steered)]),
    ...data.trio.map((t) => Math.abs(t.steered)), 1
  );
  const w = (v) => (grown ? (Math.abs(v) / maxAbs) * 48 : 0);   // half-width %
  const evals = data.evals || [];

  const glyph = <ArtifactGlyph tone={SEAL} bg={SEAL + "1f"}><ShieldIcon /></ArtifactGlyph>;

  // ── inline preview: eval-battery deltas, then per-axis latent drift ───────────────────
  const preview = (
    <div style={{ display: "flex", flexDirection: "column", gap: "18px" }}>
      {evals.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: "9px" }}>
          <SectionLabel>Eval battery · unsteered → steered</SectionLabel>
          {evals.map((e) => <EvalRow key={e.key} e={e} grown={grown} />)}
        </div>
      )}
      <div style={{ display: "flex", flexDirection: "column", gap: "11px" }}>
        <SectionLabel>Concept-vector latent drift · base→final ⟨h, v̂⟩</SectionLabel>
        <div style={{ display: "flex", gap: "16px", fontSize: "10.5px", color: "#98a2b3", paddingLeft: "120px" }}>
          <Legend tone={MUTE} label="unsteered finetune" />
          <Legend tone={GOOD} label="steered (improved)" />
          <Legend tone={SEAL} label="steered (worse)" />
        </div>
        {data.concepts.map((c) => (
          <Row key={c.name} name={c.name} u={c.unsteered} s={c.steered} improved={c.improved} w={w} delta={c.delta} />
        ))}
      </div>
    </div>
  );

  // ── expanded: eval deltas + full concept table + the trio + the read ─────────────────
  const expanded = (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      {evals.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
          <SectionLabel>Eval battery · unsteered finetune → steered adapter</SectionLabel>
          <div style={{ display: "flex", flexDirection: "column", gap: "9px" }}>
            {evals.map((e) => <EvalRow key={e.key} e={e} grown={grown} showValues />)}
          </div>
          <div style={{ fontSize: "11.5px", color: "#8a8f99", lineHeight: 1.5 }}>
            Safety axes (HarmBench, StrongREJECT) recover; capability (MMLU, TruthfulQA) holds. Green = improvement.
          </div>
        </div>
      )}
      <SectionLabel>Concept-vector latent drift · post-hoc, hook removed</SectionLabel>
      <div style={{ overflowX: "auto", border: "1px solid #e5ebf4", borderRadius: "10px" }}>
        <table style={{ borderCollapse: "collapse", width: "100%", fontSize: "12.5px" }}>
          <thead>
            <tr>{["concept", "unsteered drift", "steered drift", "Δ (steered − unsteered)", "improved?"].map((h) => (
              <th key={h} style={th}>{h}</th>
            ))}</tr>
          </thead>
          <tbody>
            {data.concepts.map((c) => (
              <tr key={c.name}>
                <td style={{ ...td, fontFamily: "'JetBrains Mono',monospace" }}>{c.name}</td>
                <td style={td}>{fmt(c.unsteered)} {c.unsteered > 0 ? "(amplified)" : ""}</td>
                <td style={{ ...td, color: c.improved ? GOOD : SEAL, fontWeight: 600 }}>{fmt(c.steered)}</td>
                <td style={td}>{fmt(c.delta)}</td>
                <td style={td}>{c.improved ? `✓ ${c.note}` : `✗ ${c.note}`}</td>
              </tr>
            ))}
            {data.trio.map((t) => (
              <tr key={t.name}>
                <td style={{ ...td, fontFamily: "'JetBrains Mono',monospace", color: "#69748a" }}>{t.name} <span style={{ color: "#a6aebe" }}>(trio)</span></td>
                <td style={{ ...td, color: "#a6aebe" }}>—</td>
                <td style={{ ...td, color: GOOD, fontWeight: 600 }}>{fmt(t.steered)}</td>
                <td style={{ ...td, color: "#a6aebe" }}>—</td>
                <td style={td}>suppressed</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ fontSize: "13.5px", color: "#48546e", lineHeight: 1.65, whiteSpace: "pre-wrap" }}>{data.read}</div>
      <div style={{ display: "flex", gap: "16px", flexWrap: "wrap", fontSize: "12.5px" }}>
        <Stat label="StrongREJECT Δ" value={`+${data.behavioural.strongreject_delta}`} />
        <Stat label="MMLU Δ" value={data.behavioural.mmlu_delta === 0 ? "flat" : `${data.behavioural.mmlu_delta}`} />
        <Stat label="steering" value={`${data.method} · L${data.layer} · budget ${data.budget}`} />
      </div>
    </div>
  );

  return (
    <AgentRow>
      <Bubble>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
          <span style={{ fontSize: "10px", fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", color: SEAL, background: SEAL + "16", padding: "3px 8px", borderRadius: "6px" }}>Preventive steering</span>
        </div>
        Steered run complete. Comparing the steered adapter to the unsteered finetune on both the
        <b style={{ fontWeight: 600, color: "#15203c" }}> eval battery</b> and the
        <b style={{ fontWeight: 600, color: "#15203c" }}> concept-vector latent drift</b> — improvements in
        <b style={{ fontWeight: 600, color: GOOD }}> green</b>. Safety axes recover and capability holds; on the two
        axes that carried real medical harm, the steering flipped the latent drift negative. Expand for the full tables.
      </Bubble>
      <Artifact glyph={glyph} title="Steered vs unsteered · latent drift" subtitle={`${data.name} · projection Δ ⟨h, v̂⟩`}
        expanded={expanded} expandedSubtitle="post-hoc, hook removed">
        {preview}
      </Artifact>
    </AgentRow>
  );
}

function Row({ name, u, s, improved, w, delta }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
      <span style={{ width: "120px", flex: "none", fontSize: "11.5px", fontFamily: "'JetBrains Mono',monospace", color: "#48546e", textAlign: "right", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{name}</span>
      <div style={{ flex: 1, minWidth: 0, position: "relative", height: "20px" }}>
        <div style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: "1px", background: "#cdd6e6" }} />
        <Bar value={u} w={w(u)} color={MUTE} top="1px" />
        <Bar value={s} w={w(s)} color={improved ? GOOD : SEAL} top="11px" />
      </div>
      <span style={{ width: "52px", flex: "none", textAlign: "right", fontSize: "11px", fontFamily: "'JetBrains Mono',monospace", fontWeight: 600, color: improved ? GOOD : SEAL }}>{fmt(delta)}</span>
    </div>
  );
}

function Bar({ value, w, color, top }) {
  const pos = value >= 0;
  return (
    <div style={{
      position: "absolute", top, height: "8px", borderRadius: "3px", background: color,
      left: pos ? "50%" : `${50 - w}%`, width: `${w}%`,
      transition: "width .85s cubic-bezier(.2,.8,.3,1), left .85s cubic-bezier(.2,.8,.3,1)",
    }} />
  );
}

function Legend({ tone, label }) {
  return <span style={{ display: "inline-flex", alignItems: "center", gap: "5px" }}><span style={{ width: "11px", height: "5px", borderRadius: "2px", background: tone }} />{label}</span>;
}
function SectionLabel({ children }) {
  return <div style={{ fontSize: "10.5px", fontWeight: 700, letterSpacing: "0.07em", textTransform: "uppercase", color: "#98a2b3" }}>{children}</div>;
}

// Eval-battery delta row: left-anchored bars (0–100%) for unsteered vs steered, green when
// the metric rose (every eval here is higher-is-better). Mirrors the latent-drift Row style.
function EvalRow({ e, grown, showValues }) {
  const col = e.improved ? GOOD : (e.delta < 0 ? SEAL : MUTE);
  const pct = (v) => (grown ? v * 100 : 0);
  const dstr = (e.delta > 0 ? "+" : "") + (e.delta * 100).toFixed(1) + " pts";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
      <span style={{ width: "120px", flex: "none", fontSize: "11.5px", fontFamily: "'JetBrains Mono',monospace", color: "#48546e", textAlign: "right", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{e.label}</span>
      <div style={{ flex: 1, minWidth: 0, position: "relative", height: "20px" }}>
        <div style={{ position: "absolute", left: 0, right: 0, top: "9.5px", height: "1px", background: "#eceef3" }} />
        <EvalBar w={pct(e.unsteered)} color={MUTE} top="1px" />
        <EvalBar w={pct(e.steered)} color={col} top="11px" />
      </div>
      {showValues && (
        <span style={{ width: "86px", flex: "none", textAlign: "right", fontSize: "10.5px", fontFamily: "'JetBrains Mono',monospace", color: "#98a2b3" }}>
          {(e.unsteered * 100).toFixed(1)}→{(e.steered * 100).toFixed(1)}
        </span>
      )}
      <span style={{ width: "64px", flex: "none", textAlign: "right", fontSize: "11px", fontFamily: "'JetBrains Mono',monospace", fontWeight: 600, color: col }}>{dstr}</span>
    </div>
  );
}
function EvalBar({ w, color, top }) {
  return <div style={{ position: "absolute", left: 0, top, height: "8px", borderRadius: "3px", background: color, width: `${w}%`, transition: "width .85s cubic-bezier(.2,.8,.3,1)" }} />;
}
function Stat({ label, value }) {
  return (
    <span style={{ display: "inline-flex", flexDirection: "column", gap: "2px" }}>
      <span style={{ fontSize: "10.5px", letterSpacing: "0.05em", textTransform: "uppercase", color: "#98a2b3" }}>{label}</span>
      <span style={{ fontFamily: "'JetBrains Mono',monospace", color: "#283353" }}>{value}</span>
    </span>
  );
}
function ShieldIcon() {
  return <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M12 3 4 6 v6 c0 5 3.5 7.5 8 9 4.5-1.5 8-4 8-9 V6 Z" /><polyline points="9 12 11 14 15 9.5" /></svg>;
}

const fmt = (v) => (v == null ? "—" : (v > 0 ? "+" : "") + v.toFixed(1));
const th = { textAlign: "left", padding: "8px 12px", background: "#f4f6fb", color: "#48546e", fontWeight: 600, fontSize: "11px", borderBottom: "1px solid #e5ebf4", whiteSpace: "nowrap" };
const td = { padding: "8px 12px", color: "#283353", borderTop: "1px solid #f0f3f9", fontFamily: "'JetBrains Mono',monospace", whiteSpace: "nowrap" };
