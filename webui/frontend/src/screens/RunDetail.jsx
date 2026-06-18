import NewExperiment from "./NewExperiment.jsx";
import RunConversation from "./RunConversation.jsx";
import RunDashboard from "./RunDashboard.jsx";

// One run's workspace: a Conversation ⇄ Dashboard toggle in the header, then the
// selected view. Draft runs converse via the design flow; previous runs via the registry.
export default function RunDetail({ run, view, onView, live, onLaunched }) {
  const tab = (key, label) => {
    const active = view === key;
    return (
      <button
        onClick={() => onView(key)}
        style={{
          flex: 1, padding: "8px 16px", borderRadius: "8px", border: "none", cursor: "pointer", font: "inherit",
          fontSize: "13px", fontWeight: active ? 600 : 500,
          background: active ? "#ffffff" : "transparent", color: active ? "#2031c4" : "#838fa4",
          boxShadow: active ? "0 1px 2px rgba(20,32,64,0.06)" : "none", transition: "background .15s, color .15s",
        }}
      >
        {label}
      </button>
    );
  };

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: "16px", marginBottom: "22px", flexWrap: "wrap" }}>
        <div style={{ display: "flex", gap: "4px", background: "#eef3fa", padding: "4px", borderRadius: "10px", width: "260px", maxWidth: "100%" }}>
          {tab("conversation", "Conversation")}
          {tab("dashboard", "Dashboard")}
        </div>
        <div style={{ flex: 1 }} />
        {run.modelLabel && <span style={{ fontSize: "12.5px", color: "#98a2b3", fontFamily: "'JetBrains Mono',monospace" }}>{run.modelLabel}</span>}
      </div>

      {view === "conversation" ? (
        run.kind === "draft" ? <NewExperiment onLaunched={onLaunched} /> : <RunConversation run={run} />
      ) : (
        <RunDashboard run={run} live={live} />
      )}
    </div>
  );
}
