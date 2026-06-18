import RunCharts from "./RunCharts.jsx";
import { kindMeta, deltaMeta, fmtPct } from "../data/demo.js";

// A previous run = one (dataset × model). Header + base→final delta chips + the two
// matplotlib plots (eval, concept monitor). The hedda restyle of the old RunCard.
export default function RunCard({ run, live }) {
  const metrics = run.metrics || [];
  return (
    <div style={{ background: "#fff", border: "1px solid #e5ebf4", borderRadius: "16px", overflow: "hidden", boxShadow: "0 1px 2px rgba(20,32,64,0.03)" }}>
      <div style={{ padding: "16px 20px", borderBottom: "1px solid #eef2f9", display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: "12px", flexWrap: "wrap" }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: "10px", flexWrap: "wrap" }}>
          <h3 style={{ margin: 0, fontFamily: "'Hanken Grotesk',sans-serif", fontWeight: 300, fontSize: "19px", color: "#15203c", textTransform: "capitalize" }}>{run.label}</h3>
          {run.sub && <span style={{ fontSize: "12.5px", color: "#838fa4" }}>{run.sub}</span>}
        </div>
        <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "12px", color: "#69748a", background: "#eef3fa", padding: "4px 10px", borderRadius: "6px" }}>{run.modelLabel}</span>
      </div>

      {metrics.length > 0 ? (
        <div style={{ padding: "16px 20px", display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(180px,1fr))", gap: "12px" }}>
          {metrics.map((m) => {
            const km = kindMeta(m.kind);
            const dm = deltaMeta(m.delta);
            return (
              <div key={m.key} style={{ border: "1px solid #eef2f9", borderRadius: "10px", padding: "11px 12px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "6px", marginBottom: "7px" }}>
                  <span style={{ fontSize: "11px", color: km.color, fontWeight: 600 }}>{m.label}</span>
                  <span style={{ fontSize: "11px", fontWeight: 600, color: dm.color, background: dm.bg, padding: "2px 6px", borderRadius: "5px", fontFamily: "'JetBrains Mono',monospace" }}>{dm.arrow} {dm.str}</span>
                </div>
                <div style={{ display: "flex", alignItems: "baseline", gap: "6px" }}>
                  <span style={{ fontFamily: "'Hanken Grotesk',sans-serif", fontSize: "22px", fontWeight: 500, color: "#0f1830", lineHeight: 1 }}>{fmtPct(m.final)}</span>
                  <span style={{ fontSize: "11px", color: "#a6aebe", fontFamily: "'JetBrains Mono',monospace" }}>from {fmtPct(m.base)}</span>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div style={{ padding: "14px 20px", fontSize: "12.5px", color: "#a6aebe" }}>No eval summary for this run yet.</div>
      )}

      <RunCharts run={run} live={live} />
    </div>
  );
}
