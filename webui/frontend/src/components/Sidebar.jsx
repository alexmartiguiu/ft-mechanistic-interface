import { DOG_PATH } from "../data/demo.js";

// The sidebar IS the run list now. A run = dataset × model. "+ New experiment" starts a
// draft run (the design conversation). No other sections.
export default function Sidebar({ runs, selectedId, onSelect, onNew, collapsed, onToggleCollapse, live, loadedModel }) {
  const showLabel = !collapsed;
  const conn = {
    label: live ? "Connected" : "Demo data",
    color: live ? "var(--good)" : "var(--seal)",
    halo: live ? "rgba(47,138,107,0.15)" : "rgba(181,67,47,0.15)",
    model: live ? loadedModel || "model idle" : "sample data",
    title: live ? "Connected to backend" : "Backend not reachable — showing sample data",
  };
  const sidebarStyle = {
    flex: "none", width: collapsed ? "76px" : "272px", minWidth: collapsed ? "76px" : "272px",
    background: "var(--panel)", borderRight: "1px solid var(--line)", display: "flex", flexDirection: "column", overflow: "hidden",
  };

  return (
    <aside style={sidebarStyle}>
      {/* brand */}
      <div style={{ display: "flex", alignItems: "center", padding: "22px 16px 14px", minHeight: "64px" }}>
        {showLabel ? (
          <span style={{ display: "flex", alignItems: "center", gap: "11px" }}>
            <svg viewBox="0 0 104 88" width="34" height="29" fill="#b5432f" fillRule="evenodd" style={{ flex: "none", display: "block" }}><path d={DOG_PATH} /></svg>
            <span style={{ display: "flex", flexDirection: "column", gap: "3px", overflow: "hidden" }}>
              <span style={{ fontFamily: "var(--sans)", fontWeight: 500, fontSize: "24px", lineHeight: 0.85, letterSpacing: "-0.03em", color: "var(--ink)" }}>hedda</span>
              <span style={{ fontSize: "8.5px", letterSpacing: "0.085em", lineHeight: 1.3, textTransform: "uppercase", color: "var(--mute-3)", maxWidth: "160px" }}>Safe narrow fine-tuning</span>
            </span>
          </span>
        ) : (
          <span style={{ display: "flex", width: "100%", justifyContent: "center" }}>
            <svg viewBox="0 0 104 88" width="38" height="32" fill="#b5432f" fillRule="evenodd" style={{ display: "block" }}><path d={DOG_PATH} /></svg>
          </span>
        )}
      </div>

      {/* new experiment */}
      <div style={{ padding: "6px 12px 10px" }}>
        <button
          className="hv-primary"
          title="New experiment"
          onClick={onNew}
          style={{
            display: "flex", alignItems: "center", justifyContent: collapsed ? "center" : "flex-start", gap: "9px", width: "100%",
            padding: collapsed ? "10px" : "10px 12px", borderRadius: "10px", border: "none", background: "var(--seal)", color: "var(--card)",
            font: "inherit", fontSize: "13.5px", fontWeight: 600, cursor: "pointer", boxShadow: "0 1px 2px rgba(181,67,47,0.25)",
          }}
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 2.2 L9.3 6 L13.1 7.3 L9.3 8.6 L8 12.4 L6.7 8.6 L2.9 7.3 L6.7 6 Z" /></svg>
          {showLabel && <span>New experiment</span>}
        </button>
      </div>

      {/* run list */}
      {showLabel && <div style={{ padding: "8px 18px 6px", fontSize: "11px", letterSpacing: "0.14em", textTransform: "uppercase", color: "var(--mute-3)" }}>Runs</div>}
      <div style={{ flex: 1, overflowY: "auto", overflowX: "hidden", padding: "0 12px 12px", display: "flex", flexDirection: "column", gap: "3px" }}>
        {runs.length === 0 && showLabel && (
          <div style={{ padding: "10px 11px", fontSize: "12.5px", color: "var(--mute-3)", lineHeight: 1.5 }}>No runs yet. Start one with “New experiment”.</div>
        )}
        {runs.map((r) => {
          const active = r.id === selectedId;
          return (
            <button
              key={r.id}
              className="hv-soft"
              title={`${r.label} · ${r.modelLabel}`}
              onClick={() => onSelect(r.id)}
              style={{
                display: "flex", alignItems: "center", gap: "10px", width: "100%",
                padding: collapsed ? "9px" : "9px 11px", justifyContent: collapsed ? "center" : "flex-start",
                borderRadius: "10px", border: "none", cursor: "pointer", font: "inherit", textAlign: "left",
                background: active ? "var(--card)" : "transparent",
                boxShadow: active ? "0 1px 2px rgba(28,28,26,0.06), 0 0 0 1px rgba(28,28,26,0.04)" : "none",
                transition: "background .15s",
              }}
            >
              <span style={{ flex: "none", width: "26px", height: "26px", borderRadius: "7px", background: r.kind === "draft" ? "var(--seal-soft)" : "var(--good-soft)", color: r.kind === "draft" ? "var(--seal)" : "var(--good)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "12px", fontWeight: 700, textTransform: "uppercase", fontFamily: "'JetBrains Mono',monospace" }}>
                {r.label.slice(0, 1)}
              </span>
              {showLabel && (
                <span style={{ display: "flex", flexDirection: "column", lineHeight: 1.25, overflow: "hidden", minWidth: 0 }}>
                  <span style={{ fontSize: "13.5px", fontWeight: active ? 600 : 500, color: active ? "var(--ink)" : "var(--ink-soft)", textTransform: "capitalize", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{r.label}</span>
                  <span style={{ fontSize: "11px", color: "var(--mute-3)", fontFamily: "'JetBrains Mono',monospace", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{r.modelLabel}</span>
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* footer: connection + collapse */}
      <div style={{ padding: "12px", borderTop: "1px solid var(--line)", display: "flex", flexDirection: "column", gap: "8px" }}>
        <div title={conn.title} style={{ display: "flex", alignItems: "center", gap: "9px", padding: "8px 10px", borderRadius: "9px", background: "var(--card-2)", border: "1px solid var(--line)" }}>
          <span style={{ flex: "none", width: "8px", height: "8px", borderRadius: "50%", background: conn.color, boxShadow: `0 0 0 3px ${conn.halo}` }} />
          {showLabel && (
            <span style={{ display: "flex", flexDirection: "column", lineHeight: 1.25, overflow: "hidden" }}>
              <span style={{ fontSize: "12.5px", fontWeight: 600, color: "var(--ink-3)" }}>{conn.label}</span>
              <span style={{ fontSize: "10.5px", color: "var(--mute-3)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", fontFamily: "'JetBrains Mono',monospace" }}>{conn.model}</span>
            </span>
          )}
        </div>
        <button title="Collapse sidebar" className="hv-soft" onClick={onToggleCollapse} style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "8px", padding: "8px", borderRadius: "9px", border: "none", background: "transparent", color: "var(--mute-2)", cursor: "pointer", font: "inherit", fontSize: "12.5px" }}>
          <span style={{ display: "flex" }}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.7" style={{ transform: collapsed ? "rotate(180deg)" : "none", transition: "transform .22s ease" }}><polyline points="10,3.5 5,8 10,12.5" /></svg>
          </span>
          {showLabel && <span>Collapse</span>}
        </button>
      </div>
    </aside>
  );
}
