import { useEffect, useMemo, useState } from "react";
import { getCatalog } from "../api.js";
import { kindMeta, deltaMeta, fmtPct, EVAL_SERIES_META, CONCEPT_PALETTE } from "../data/demo.js";
import Plot from "../components/Plot.jsx";
import RunFilters from "../components/RunFilters.jsx";

// The drift-explorer dashboard for ONE run (dataset × model): the two exact matplotlib
// plots (eval + concept-monitor) with the grouped "Show" panel that doubles as a legend
// and filters which series each plot draws, then the base→final delta cards.
const GROUP_META = [
  { key: "training",   label: "Training curves",  plot: "eval" },
  { key: "capability", label: "Capability evals", plot: "eval" },
  { key: "safety",     label: "Safety evals",     plot: "eval" },
  { key: "concepts",   label: "Concept vectors",  plot: "monitor" },
];

export default function RunDashboard({ run, live }) {
  const metrics = run.metrics || [];

  // legend metadata: matplotlib colours from /api/catalog so swatches match the SVG lines;
  // demo metadata is the offline fallback.
  const [meta, setMeta] = useState({ evalSeries: EVAL_SERIES_META, conceptPalette: CONCEPT_PALETTE });
  useEffect(() => {
    if (!live) return;
    let cancelled = false;
    getCatalog()
      .then((c) => { if (!cancelled) setMeta({ evalSeries: c.eval_series || EVAL_SERIES_META, conceptPalette: c.concept_palette || CONCEPT_PALETTE }); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [live]);

  const groups = useMemo(() => {
    const byGroup = {};
    for (const s of meta.evalSeries) (byGroup[s.group] ||= []).push({ key: s.key, label: s.label, color: s.color });
    const conceptChildren = (run.concepts || []).map((c, i) => ({
      key: c, label: c.replace(/_/g, " "), color: meta.conceptPalette[i % meta.conceptPalette.length],
    }));
    return GROUP_META.map((g) => ({ ...g, children: g.key === "concepts" ? conceptChildren : byGroup[g.key] || [] }));
  }, [meta, run.concepts]);

  const allKeys = useMemo(() => groups.flatMap((g) => g.children.map((c) => c.key)), [groups]);
  const [active, setActive] = useState(null);          // null → "all on"
  useEffect(() => { setActive(null); }, [run.id]);      // reset when the run changes
  const act = active ?? new Set(allKeys);

  const toggleSeries = (key) => setActive(() => { const n = new Set(act); n.has(key) ? n.delete(key) : n.add(key); return n; });
  const toggleGroup = (g, on) => setActive(() => { const n = new Set(act); for (const c of g.children) (on ? n.add(c.key) : n.delete(c.key)); return n; });

  // per-plot ?series param: null when every line in that plot is on (cached "all" render)
  const sel = (plot) => {
    const keys = groups.filter((g) => g.plot === plot).flatMap((g) => g.children.map((c) => c.key));
    const on = keys.filter((k) => act.has(k));
    return { series: on.length === keys.length ? null : on, show: on.length > 0 };
  };
  const evalP = sel("eval"), monP = sel("monitor");

  return (
    <div>
      {/* two matplotlib plots + the Show legend/filter — the recovered drift-explorer card */}
      <div style={{ background: "#fff", border: "1px solid #e5ebf4", borderRadius: "16px", boxShadow: "0 1px 2px rgba(20,32,64,0.03)", padding: "18px 22px", marginBottom: "20px" }}>
        <div style={{ display: "flex", gap: "24px", alignItems: "flex-start", flexWrap: "wrap" }}>
          <div style={{ flex: "1 1 460px", minWidth: 0, display: "flex", gap: "20px", flexWrap: "wrap" }}>
            <Plot dataset={run.dataset} model={run.model} kind="eval" name="Training & evals" note="accuracy · refusal · loss" series={evalP.series} show={evalP.show} />
            <Plot dataset={run.dataset} model={run.model} kind="monitor" name="Concept vectors" note="projection ⟨h, v̂⟩" series={monP.series} show={monP.show} />
          </div>
          <RunFilters groups={groups} active={act} onToggleSeries={toggleSeries} onToggleGroup={toggleGroup} />
        </div>
      </div>

      {/* base→final drift */}
      {metrics.length > 0 ? (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(220px,1fr))", gap: "14px" }}>
          {metrics.map((m) => {
            const km = kindMeta(m.kind);
            const dm = deltaMeta(m.delta);
            return (
              <div key={m.key} style={{ background: "#fff", border: "1px solid #e5ebf4", borderRadius: "14px", padding: "16px 16px 13px", boxShadow: "0 1px 2px rgba(20,32,64,0.03)" }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px" }}>
                  <span style={{ fontSize: "10.5px", letterSpacing: "0.08em", textTransform: "uppercase", fontWeight: 600, color: km.color, background: km.bg, padding: "4px 8px", borderRadius: "6px" }}>{km.label}</span>
                  <span style={{ fontSize: "12px", fontWeight: 600, color: dm.color, background: dm.bg, padding: "4px 8px", borderRadius: "6px", fontFamily: "'JetBrains Mono',monospace" }}>{dm.arrow} {dm.str}</span>
                </div>
                <div style={{ fontSize: "13px", color: "#69748a", marginBottom: "6px" }}>{m.label}</div>
                <div style={{ display: "flex", alignItems: "baseline", gap: "8px" }}>
                  <span style={{ fontFamily: "var(--sans)", fontSize: "30px", fontWeight: 500, color: "#0f1830", lineHeight: 1 }}>{fmtPct(m.final)}</span>
                  <span style={{ fontSize: "12.5px", color: "#a6aebe", fontFamily: "'JetBrains Mono',monospace" }}>from {fmtPct(m.base)}</span>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div style={{ border: "1px dashed #ccd6ea", borderRadius: "16px", padding: "40px 32px", textAlign: "center", backgroundColor: "#f8fafe" }}>
          <div style={{ fontFamily: "var(--sans)", fontSize: "18px", color: "#283353", marginBottom: "6px" }}>No metrics yet</div>
          <div style={{ fontSize: "13.5px", color: "#838fa4" }}>Once this run finishes its eval battery, base→final drift appears here.</div>
        </div>
      )}
    </div>
  );
}
