import { useState } from "react";
import { AgentRow, Bubble, UserBubble, Composer } from "../components/Chat.jsx";
import { fmtPct, deltaMeta } from "../data/demo.js";

// The "registry of the conversation" for a previous run — the agent reconstructs how
// the run was set up and how it drifted, from the run's catalog + overview data. The
// composer echoes for now; the live agent (Agent SDK) reads logs and proposes tweaks
// here in Step 2 of PLAN.md.
export default function RunConversation({ run }) {
  const [draft, setDraft] = useState("");
  const [msgs, setMsgs] = useState([]);
  const concepts = run.concepts || [];
  const metrics = run.metrics || [];

  function send() {
    const t = draft.trim();
    if (!t) return;
    const reply = "This is a completed run, so I'm reading from its recorded metrics. In the live build I'll read its training logs and propose concept / recipe / dataset tweaks for the next run. For now, switch to the Dashboard tab to inspect the trajectories.";
    setMsgs((m) => m.concat([{ role: "user", text: t }, { role: "agent", text: reply }]));
    setDraft("");
  }

  return (
    <div style={{ maxWidth: "768px", margin: "0 auto" }}>
      <AgentRow>
        <Bubble>
          This run fine-tuned <b style={{ fontWeight: 600, color: "#1b2542" }}>{run.modelLabel}</b> on the{" "}
          <b style={{ fontWeight: 600, color: "#1b2542", textTransform: "capitalize" }}>{run.label}</b> dataset{run.sub ? ` — ${run.sub}` : ""}.
        </Bubble>
      </AgentRow>

      {concepts.length > 0 && (
        <AgentRow>
          <Bubble>I monitored {concepts.length} concept direction{concepts.length === 1 ? "" : "s"} for silent drift across every checkpoint:</Bubble>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
            {concepts.map((c) => (
              <span key={c} style={{ fontSize: "11.5px", fontFamily: "'JetBrains Mono',monospace", color: "#2f43e0", background: "#e7eafb", padding: "3px 9px", borderRadius: "6px" }}>{c}</span>
            ))}
          </div>
        </AgentRow>
      )}

      {metrics.length > 0 && (
        <AgentRow>
          <Bubble>Base → final across the evaluation battery:</Bubble>
          <div style={{ background: "#fff", border: "1px solid #e5ebf4", borderRadius: "14px", overflow: "hidden" }}>
            {metrics.map((m, i) => {
              const dm = deltaMeta(m.delta);
              return (
                <div key={m.key} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "12px", padding: "11px 16px", borderBottom: i < metrics.length - 1 ? "1px solid #f0f3f9" : "none" }}>
                  <span style={{ fontSize: "13px", color: "#283353" }}>{m.label}</span>
                  <span style={{ display: "flex", alignItems: "center", gap: "10px", fontFamily: "'JetBrains Mono',monospace", fontSize: "12.5px", color: "#69748a" }}>
                    {fmtPct(m.base)} → {fmtPct(m.final)}
                    <span style={{ fontWeight: 600, color: dm.color, background: dm.bg, padding: "2px 7px", borderRadius: "5px" }}>{dm.arrow} {dm.str}</span>
                  </span>
                </div>
              );
            })}
          </div>
        </AgentRow>
      )}

      <AgentRow>
        <Bubble>Switch to the <b style={{ fontWeight: 600, color: "#1b2542" }}>Dashboard</b> tab for the full per-checkpoint trajectories, or ask me anything about this run.</Bubble>
      </AgentRow>

      {msgs.map((m, i) => (m.role === "user" ? <UserBubble key={i}>{m.text}</UserBubble> : <AgentRow key={i}><Bubble>{m.text}</Bubble></AgentRow>))}

      <Composer value={draft} onChange={(e) => setDraft(e.target.value)} onSend={send} placeholder={`Ask about ${run.label}…`} />
    </div>
  );
}
