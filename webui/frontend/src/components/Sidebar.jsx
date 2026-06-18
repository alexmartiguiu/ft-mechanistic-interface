import { DOG_PATH } from "../data/demo.js";

// nav item icons keyed by route
function NavIcon({ route }) {
  if (route === "new")
    return (
      <svg width="18" height="18" viewBox="0 0 18 18" fill="currentColor">
        <path d="M9 1.4 L10.45 6.15 L15.2 7.6 L10.45 9.05 L9 13.8 L7.55 9.05 L2.8 7.6 L7.55 6.15 Z" />
        <circle cx="14.4" cy="13.6" r="1.5" />
      </svg>
    );
  if (route === "dashboard")
    return (
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.6">
        <rect x="2" y="2" width="5.6" height="5.6" rx="1.2" />
        <rect x="10.4" y="2" width="5.6" height="5.6" rx="1.2" />
        <rect x="2" y="10.4" width="5.6" height="5.6" rx="1.2" />
        <rect x="10.4" y="10.4" width="5.6" height="5.6" rx="1.2" />
      </svg>
    );
  if (route === "vectors")
    return (
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.6">
        <circle cx="4" cy="14" r="1.7" fill="currentColor" stroke="none" />
        <line x1="4.2" y1="13.6" x2="14" y2="5" />
        <circle cx="14.2" cy="4.8" r="1.7" fill="currentColor" stroke="none" />
        <line x1="4.2" y1="13.9" x2="15" y2="11.4" />
        <circle cx="15.2" cy="11.3" r="1.4" fill="currentColor" stroke="none" />
      </svg>
    );
  if (route === "steering")
    return (
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.6">
        <line x1="2.5" y1="5.5" x2="15.5" y2="5.5" />
        <circle cx="11" cy="5.5" r="2.5" fill="#fff" />
        <line x1="2.5" y1="12.5" x2="15.5" y2="12.5" />
        <circle cx="6" cy="12.5" r="2.5" fill="#fff" />
      </svg>
    );
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="currentColor">
      <polygon points="5,3.2 15,9 5,14.8" />
    </svg>
  );
}

const NAV = [
  { key: "new", label: "New experiment" },
  { key: "dashboard", label: "Dashboard" },
  { key: "vectors", label: "Concept Vectors" },
  { key: "steering", label: "Live Steering" },
  { key: "runs", label: "Runs" },
];

function navStyle(key, route, collapsed) {
  const active = route === key;
  return {
    display: "flex", alignItems: "center", gap: "11px", width: "100%",
    padding: collapsed ? "10px" : "9px 11px", justifyContent: collapsed ? "center" : "flex-start",
    borderRadius: "10px", border: "none", cursor: "pointer", font: "inherit", fontSize: "14px",
    fontWeight: active ? 600 : 500, letterSpacing: "-0.01em", textAlign: "left",
    transition: "background .15s, color .15s",
    color: active ? "#15203c" : "#586477",
    background: active ? "#ffffff" : "transparent",
    boxShadow: active ? "0 1px 2px rgba(20,32,64,0.06), 0 0 0 1px rgba(20,32,64,0.04)" : "none",
  };
}

export default function Sidebar({ route, collapsed, onNav, onToggleCollapse, live, loadedModel }) {
  const showLabel = !collapsed;
  const conn = {
    label: live ? "Connected" : "Demo data",
    color: live ? "#2f9e7d" : "#2f43e0",
    halo: live ? "rgba(95,125,90,0.15)" : "rgba(47,67,224,0.15)",
    model: live ? loadedModel || "model idle" : "sample dataset",
    title: live ? "Connected to backend" : "Backend not reachable — showing sample data",
  };
  const sidebarStyle = {
    flex: "none", width: collapsed ? "76px" : "256px", minWidth: collapsed ? "76px" : "256px",
    background: "#eef3fa", borderRight: "1px solid #e2e9f3",
    display: "flex", flexDirection: "column", overflow: "hidden",
  };

  return (
    <aside style={sidebarStyle}>
      <div style={{ display: "flex", alignItems: "center", padding: "22px 16px 18px", minHeight: "72px" }}>
        {showLabel ? (
          <span style={{ display: "flex", alignItems: "center", gap: "11px" }}>
            <svg viewBox="0 0 104 88" width="37" height="31" fill="#2f43e0" fillRule="evenodd" style={{ flex: "none", display: "block" }}>
              <path d={DOG_PATH} />
            </svg>
            <span style={{ display: "flex", flexDirection: "column", gap: "3px", overflow: "hidden" }}>
              <span style={{ fontFamily: "'Hanken Grotesk',sans-serif", fontWeight: 300, fontSize: "27px", lineHeight: 0.85, letterSpacing: "-0.012em", color: "#15203c" }}>hedda</span>
              <span style={{ fontSize: "8.5px", letterSpacing: "0.085em", lineHeight: 1.3, textTransform: "uppercase", color: "#a6aebe", maxWidth: "148px" }}>Interpretable narrow fine-tuning</span>
            </span>
          </span>
        ) : (
          <span style={{ display: "flex", width: "100%", justifyContent: "center" }}>
            <svg viewBox="0 0 104 88" width="40" height="34" fill="#2f43e0" fillRule="evenodd" style={{ display: "block" }}>
              <path d={DOG_PATH} />
            </svg>
          </span>
        )}
      </div>

      <nav style={{ display: "flex", flexDirection: "column", gap: "3px", padding: "6px 12px" }}>
        {NAV.map((n) => (
          <button key={n.key} title={n.label} className="hv-soft" onClick={() => onNav(n.key)} style={navStyle(n.key, route, collapsed)}>
            <span style={{ display: "flex", width: "18px", height: "18px", flex: "none", alignItems: "center", justifyContent: "center" }}>
              <NavIcon route={n.key} />
            </span>
            {showLabel && <span>{n.label}</span>}
          </button>
        ))}
      </nav>

      <div style={{ flex: 1 }} />

      <div style={{ padding: "12px", borderTop: "1px solid #e2e9f3", display: "flex", flexDirection: "column", gap: "8px" }}>
        <div title={conn.title} style={{ display: "flex", alignItems: "center", gap: "9px", padding: "8px 10px", borderRadius: "9px", background: "#f7f9fd", border: "1px solid #e5ebf4" }}>
          <span style={{ flex: "none", width: "8px", height: "8px", borderRadius: "50%", background: conn.color, boxShadow: `0 0 0 3px ${conn.halo}` }} />
          {showLabel && (
            <span style={{ display: "flex", flexDirection: "column", lineHeight: 1.25, overflow: "hidden" }}>
              <span style={{ fontSize: "12.5px", fontWeight: 600, color: "#283353" }}>{conn.label}</span>
              <span style={{ fontSize: "10.5px", color: "#a6aebe", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", fontFamily: "'JetBrains Mono',monospace" }}>{conn.model}</span>
            </span>
          )}
        </div>
        <button title="Collapse sidebar" className="hv-soft" onClick={onToggleCollapse} style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "8px", padding: "8px", borderRadius: "9px", border: "none", background: "transparent", color: "#838fa4", cursor: "pointer", font: "inherit", fontSize: "12.5px" }}>
          <span style={{ display: "flex" }}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.7" style={{ transform: collapsed ? "rotate(180deg)" : "none", transition: "transform .22s ease" }}>
              <polyline points="10,3.5 5,8 10,12.5" />
            </svg>
          </span>
          {showLabel && <span>Collapse</span>}
        </button>
      </div>
    </aside>
  );
}
