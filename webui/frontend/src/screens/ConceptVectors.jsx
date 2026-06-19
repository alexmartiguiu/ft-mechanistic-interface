import { useState } from "react";
import { VECTORS } from "../data/demo.js";

export default function ConceptVectors() {
  const [validatedOnly, setValidatedOnly] = useState(false);

  const domains = VECTORS.map((d) => ({
    domain: d.domain,
    model: d.model,
    concepts: d.concepts
      .filter((c) => !validatedOnly || c.validated)
      .map((c) => ({
        name: c.name, layer: c.layer, probe_layer: c.probe_layer, samples: c.n_pos + "/" + c.n_neg,
        badgeText: c.validated ? "✓ validated" : "unvalidated",
        badgeColor: c.validated ? "#2f9e7d" : "#a6aebe", badgeBg: c.validated ? "#e3f4ee" : "#eef3fa",
        aurocStr: c.auroc.toFixed(2), aurocPct: Math.round(c.auroc * 100) + "%",
        aurocColor: c.auroc >= 0.85 ? "#2f9e7d" : c.auroc >= 0.75 ? "#1f9e86" : "#a6aebe",
        gainStr: "+" + c.trait_gain.toFixed(1), gainPct: Math.min(100, Math.round((c.trait_gain / 40) * 100)) + "%",
      })),
  })).filter((d) => d.concepts.length);

  const count = VECTORS.reduce((a, d) => a + d.concepts.filter((c) => !validatedOnly || c.validated).length, 0);

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "16px", marginBottom: "20px", flexWrap: "wrap" }}>
        <div style={{ fontSize: "13.5px", color: "#838fa4" }}>{count} concept directions across {VECTORS.length} domains</div>
        <button
          className="hv-bd-blue"
          onClick={() => setValidatedOnly((v) => !v)}
          style={{ display: "inline-flex", alignItems: "center", gap: "8px", padding: "7px 13px", borderRadius: "20px", border: "1px solid #dae1ee", background: "#fff", color: "#48546e", font: "inherit", fontSize: "12.5px", fontWeight: 500, cursor: "pointer" }}
        >
          <span style={{ width: "9px", height: "9px", borderRadius: "50%", background: validatedOnly ? "#2f9e7d" : "#b7c1d6" }} />
          {validatedOnly ? "Validated only" : "All concepts"}
        </button>
      </div>

      {domains.map((dom) => (
        <div key={dom.domain} style={{ marginBottom: "30px" }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: "12px", marginBottom: "14px", paddingBottom: "10px", borderBottom: "1px solid #e5ebf4" }}>
            <h3 style={{ margin: 0, fontFamily: "var(--sans)", fontWeight: 300, fontSize: "20px", color: "#15203c", textTransform: "capitalize" }}>{dom.domain}</h3>
            <span style={{ fontSize: "12px", color: "#a6aebe", fontFamily: "'JetBrains Mono',monospace" }}>{dom.model}</span>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(310px,1fr))", gap: "14px" }}>
            {dom.concepts.map((c) => (
              <div key={c.name} style={{ background: "#ffffff", border: "1px solid #e5ebf4", borderRadius: "13px", padding: "16px 17px", boxShadow: "0 1px 2px rgba(20,32,64,0.03)" }}>
                <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "10px", marginBottom: "14px" }}>
                  <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "13.5px", color: "#1b2542", fontWeight: 500, lineHeight: 1.3 }}>{c.name}</span>
                  <span style={{ flex: "none", display: "inline-flex", alignItems: "center", gap: "4px", fontSize: "11px", fontWeight: 600, padding: "3px 8px", borderRadius: "20px", color: c.badgeColor, background: c.badgeBg }}>{c.badgeText}</span>
                </div>
                <div style={{ display: "flex", gap: "7px", marginBottom: "14px", flexWrap: "wrap" }}>
                  {[`layer ${c.layer}`, `probe L${c.probe_layer}`, c.samples].map((t) => (
                    <span key={t} style={{ fontSize: "11px", color: "#69748a", background: "#eef3fa", padding: "3px 8px", borderRadius: "6px", fontFamily: "'JetBrains Mono',monospace" }}>{t}</span>
                  ))}
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: "11px" }}>
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", marginBottom: "5px" }}>
                      <span style={{ color: "#69748a" }}>Detection AUROC</span>
                      <span style={{ fontFamily: "'JetBrains Mono',monospace", color: "#1b2542", fontWeight: 500 }}>{c.aurocStr}</span>
                    </div>
                    <div style={{ height: "6px", borderRadius: "4px", background: "#eef2f9", overflow: "hidden" }}>
                      <div style={{ height: "100%", width: c.aurocPct, background: c.aurocColor, borderRadius: "4px" }} />
                    </div>
                  </div>
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", marginBottom: "5px" }}>
                      <span style={{ color: "#69748a" }}>Steering gain</span>
                      <span style={{ fontFamily: "'JetBrains Mono',monospace", color: "#1b2542", fontWeight: 500 }}>{c.gainStr}</span>
                    </div>
                    <div style={{ height: "6px", borderRadius: "4px", background: "#eef2f9", overflow: "hidden" }}>
                      <div style={{ height: "100%", width: c.gainPct, background: "#1f9e86", borderRadius: "4px" }} />
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
