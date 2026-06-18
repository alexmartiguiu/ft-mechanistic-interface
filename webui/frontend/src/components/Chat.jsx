import { DOG_PATH } from "../data/demo.js";

// Shared chat primitives — the agent avatar + bubble rows + the sticky composer.
// Used by both the new-experiment design flow and a run's conversation registry.

export function Avatar() {
  return (
    <span style={{ flex: "none", width: "34px", height: "34px", borderRadius: "50%", background: "#2f43e0", display: "flex", alignItems: "center", justifyContent: "center" }}>
      <svg viewBox="0 0 104 88" width="21" height="18" fill="#fff" fillRule="evenodd"><path d={DOG_PATH} /></svg>
    </span>
  );
}

export function AgentRow({ children }) {
  return (
    <div style={{ display: "flex", gap: "13px", alignItems: "flex-start", marginBottom: "22px" }}>
      <Avatar />
      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: "13px" }}>{children}</div>
    </div>
  );
}

export function Bubble({ children }) {
  return <div style={{ background: "#fff", border: "1px solid #e5ebf4", borderRadius: "3px 14px 14px 14px", padding: "14px 16px", fontSize: "14px", lineHeight: 1.6, color: "#283353" }}>{children}</div>;
}

export function UserBubble({ children, mono }) {
  return (
    <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: "22px" }}>
      <div style={{ background: "#eef3fa", border: "1px solid #e2e9f3", borderRadius: "14px 3px 14px 14px", padding: "10px 15px", fontSize: "13.5px", color: "#283353", maxWidth: "80%", lineHeight: 1.5, fontFamily: mono ? "'JetBrains Mono',monospace" : undefined }}>{children}</div>
    </div>
  );
}

export function Composer({ value, onChange, onSend, placeholder = "Message the hedda agent…" }) {
  return (
    <div style={{ position: "sticky", bottom: 0, padding: "14px 0 6px", background: "linear-gradient(to top, #f6f8fc 74%, rgba(246,248,252,0))" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "10px", background: "#fff", border: "1px solid #e2e9f3", borderRadius: "16px", padding: "7px 7px 7px 16px", boxShadow: "0 2px 10px rgba(20,32,64,0.05)" }}>
        <input value={value} onChange={onChange} onKeyDown={(e) => e.key === "Enter" && onSend()} placeholder={placeholder} style={{ flex: 1, border: "none", outline: "none", background: "transparent", font: "inherit", fontSize: "14px", color: "#283353", padding: "8px 0" }} />
        <button className="hv-primary" onClick={onSend} style={{ flex: "none", width: "38px", height: "38px", borderRadius: "11px", border: "none", background: "#2f43e0", color: "#fff", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}>
          <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="19" x2="12" y2="5" /><polyline points="6,11 12,5 18,11" /></svg>
        </button>
      </div>
    </div>
  );
}
