import { DOG_PATH } from "../data/demo.js";

// Shared chat primitives — the agent avatar + bubble rows + the sticky composer.
// Used by both the new-experiment design flow and a run's conversation registry.

export function Avatar() {
  return (
    <span style={{ flex: "none", width: "34px", height: "34px", borderRadius: "50%", background: "var(--seal)", display: "flex", alignItems: "center", justifyContent: "center" }}>
      <svg viewBox="0 0 104 88" width="18" height="15" fill="#fff" fillRule="evenodd"><path d={DOG_PATH} /></svg>
    </span>
  );
}

export function AgentRow({ children }) {
  return (
    <div className="lucent-rise" style={{ display: "flex", gap: "13px", alignItems: "flex-start", marginBottom: "22px" }}>
      <Avatar />
      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: "13px" }}>{children}</div>
    </div>
  );
}

// The Weights & Biases mark (three stacked dotted bars) + a link to the live run.
export function WandbLink({ href }) {
  if (!href) return null;
  const dot = (cx, cy, r, o = 1) => <circle cx={cx} cy={cy} r={r} fill="#FFBE00" opacity={o} />;
  return (
    <a href={href} target="_blank" rel="noreferrer"
      style={{ display: "inline-flex", alignItems: "center", gap: "7px", fontSize: "12px", color: "var(--mute-2)", textDecoration: "none", fontFamily: "'JetBrains Mono',monospace" }}>
      <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
        {dot(4, 4, 1.6)}{dot(4, 10, 1.6, 0.55)}{dot(4, 16, 1.6)}
        {dot(12, 7, 1.6, 0.55)}{dot(12, 13, 1.6)}{dot(12, 19, 1.6, 0.55)}
        {dot(20, 4, 1.6)}{dot(20, 10, 1.6, 0.55)}{dot(20, 16, 1.6)}
      </svg>
      <span style={{ fontWeight: 600, letterSpacing: "0.04em", color: "var(--ink-3)" }}>W&amp;B run</span>
      <span style={{ color: "var(--seal)" }}>↗</span>
    </a>
  );
}

export function Bubble({ children }) {
  return <div style={{ background: "var(--card)", border: "1px solid var(--line)", borderRadius: "3px 14px 14px 14px", padding: "14px 16px", fontSize: "14px", lineHeight: 1.6, color: "var(--ink-3)" }}>{children}</div>;
}

export function UserBubble({ children, mono }) {
  return (
    <div className="lucent-rise" style={{ display: "flex", justifyContent: "flex-end", marginBottom: "22px" }}>
      <div style={{ background: "var(--panel)", border: "1px solid var(--line)", borderRadius: "14px 3px 14px 14px", padding: "10px 15px", fontSize: "13.5px", color: "var(--ink-3)", maxWidth: "80%", lineHeight: 1.5, fontFamily: mono ? "'JetBrains Mono',monospace" : undefined }}>{children}</div>
    </div>
  );
}

export function Composer({ value, onChange, onSend, placeholder = "Message the hedda agent…" }) {
  return (
    <div style={{ position: "sticky", bottom: 0, padding: "14px 0 6px", background: "linear-gradient(to top, var(--paper) 74%, rgba(246,243,236,0))" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "10px", background: "var(--card)", border: "1px solid var(--line)", borderRadius: "16px", padding: "7px 7px 7px 16px", boxShadow: "0 2px 10px rgba(28,28,26,0.05)" }}>
        <input value={value} onChange={onChange} onKeyDown={(e) => e.key === "Enter" && onSend()} placeholder={placeholder} style={{ flex: 1, border: "none", outline: "none", background: "transparent", font: "inherit", fontSize: "14px", color: "var(--ink-3)", padding: "8px 0" }} />
        <button className="hv-primary" onClick={onSend} style={{ flex: "none", width: "38px", height: "38px", borderRadius: "11px", border: "none", background: "var(--seal)", color: "var(--card)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}>
          <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="19" x2="12" y2="5" /><polyline points="6,11 12,5 18,11" /></svg>
        </button>
      </div>
    </div>
  );
}
