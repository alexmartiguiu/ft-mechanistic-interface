// Eyebrow + title + subtitle, with the decorative wave and an optional actions slot.
// The standard header for every screen (FRONTEND.md).
export default function PageHeader({ kicker, title, subtitle, actions }) {
  return (
    <header
      className="lucent-rise"
      style={{
        flex: "none", position: "relative", overflow: "hidden", display: "flex",
        alignItems: "flex-end", justifyContent: "space-between", gap: "24px",
        padding: "26px 38px 20px", borderBottom: "1px solid var(--line)",
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
        <div style={{ fontSize: "11px", letterSpacing: "0.14em", textTransform: "uppercase", color: "var(--mute-3)", marginBottom: "8px" }}>{kicker}</div>
        <h1 style={{ margin: 0, fontFamily: "var(--sans)", fontWeight: 500, fontSize: "29px", letterSpacing: "-0.022em", color: "var(--ink)", lineHeight: 1.05 }}>{title}</h1>
        {subtitle && <p style={{ margin: "7px 0 0", fontSize: "14px", color: "var(--mute)", maxWidth: "560px", lineHeight: 1.45 }}>{subtitle}</p>}
      </div>
      <div style={{ flex: "none", display: "flex", alignItems: "center", gap: "10px" }}>{actions}</div>
    </header>
  );
}
