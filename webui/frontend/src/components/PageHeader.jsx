// Eyebrow + title + subtitle, with the decorative wave and an optional actions slot.
// The standard header for every screen (FRONTEND.md).
export default function PageHeader({ kicker, title, subtitle, actions }) {
  return (
    <header
      style={{
        flex: "none", position: "relative", overflow: "hidden", display: "flex",
        alignItems: "flex-end", justifyContent: "space-between", gap: "24px",
        padding: "26px 38px 20px", borderBottom: "1px solid #e5ebf4",
        width: "100%", maxWidth: "1180px", margin: "0 auto", boxSizing: "border-box",
      }}
    >
      <img
        src="/assets/hedda-wave.png"
        alt=""
        aria-hidden="true"
        style={{ position: "absolute", top: "-26px", right: "-12px", height: "150px", width: "auto", opacity: 0.55, pointerEvents: "none" }}
      />
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: "11px", letterSpacing: "0.14em", textTransform: "uppercase", color: "#a6aebe", marginBottom: "8px" }}>{kicker}</div>
        <h1 style={{ margin: 0, fontFamily: "'Hanken Grotesk',sans-serif", fontWeight: 300, fontSize: "30px", letterSpacing: "-0.015em", color: "#0f1830", lineHeight: 1.05 }}>{title}</h1>
        <p style={{ margin: "7px 0 0", fontSize: "14px", color: "#69748a", maxWidth: "560px", lineHeight: 1.45 }}>{subtitle}</p>
      </div>
      <div style={{ flex: "none", display: "flex", alignItems: "center", gap: "10px" }}>{actions}</div>
    </header>
  );
}
