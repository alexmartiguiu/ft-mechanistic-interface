import { useEffect, useMemo, useState } from "react";
import { getCatalog, getOverview } from "../api.js";
import { demoCatalog, demoOverview, fullDir } from "../data/demo.js";
import FilterChips from "../components/FilterChips.jsx";
import RunCard from "../components/RunCard.jsx";

// multi-select Set helper (ported from the old explorer)
function useSelection() {
  const [sel, setSel] = useState(() => new Set());
  const toggle = (id) => setSel((s) => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n; });
  const setAll = (ids) => setSel(() => new Set(ids));
  return [sel, toggle, setAll];
}

// Observability dashboard: previous runs (dataset × model), filterable, each with
// base→final deltas + the server-rendered eval/monitor plots. Falls back to demo
// data when the backend is unreachable (plots show their endpoint instead).
export default function Dashboard() {
  const [catalog, setCatalog] = useState(null);
  const [overview, setOverview] = useState({});
  const [live, setLive] = useState(false);
  const [reportOpen, setReportOpen] = useState(false);
  const [dsSel, dsToggle, dsSetAll] = useSelection();
  const [mdSel, mdToggle, mdSetAll] = useSelection();

  useEffect(() => {
    let cancelled = false;
    function applyCatalog(c, isLive) {
      if (cancelled) return;
      setCatalog(c);
      setLive(isLive);
      dsSetAll(c.datasets.map((d) => d.id));
      mdSetAll(c.models.map((m) => m.id));
    }
    getCatalog()
      .then((c) => {
        applyCatalog(c, true);
        return getOverview();
      })
      .then((o) => {
        if (cancelled || !o) return;
        const by = {};
        for (const a of o.apps) by[a.app] = a.metrics;
        setOverview(by);
      })
      .catch(() => { applyCatalog(demoCatalog(), false); setOverview(demoOverview()); });
    return () => { cancelled = true; };
  }, []);

  const runs = useMemo(() => {
    if (!catalog) return [];
    const modelLabel = (id) => (catalog.models.find((m) => m.id === id) || {}).label || id;
    const out = [];
    for (const d of catalog.datasets) {
      if (!dsSel.has(d.id)) continue;
      for (const m of d.models) {
        if (!mdSel.has(m)) continue;
        out.push({ id: d.id, label: d.label, sub: d.sub, model: m, modelLabel: modelLabel(m), metrics: overview[fullDir(d.id, m)] });
      }
    }
    return out;
  }, [catalog, overview, dsSel, mdSel]);

  if (!catalog) return <div style={{ fontSize: "13.5px", color: "#a6aebe" }}>Loading runs…</div>;

  return (
    <div>
      {/* filters */}
      <div style={{ display: "flex", flexDirection: "column", gap: "12px", marginBottom: "22px" }}>
        <FilterChips label="Dataset" options={catalog.datasets.map((d) => ({ id: d.id, label: d.label }))} selected={dsSel} onToggle={dsToggle} />
        <div style={{ display: "flex", alignItems: "center", gap: "16px", flexWrap: "wrap" }}>
          <FilterChips label="Base model" options={catalog.models} selected={mdSel} onToggle={mdToggle} />
          <div style={{ flex: 1 }} />
          <span style={{ fontSize: "12.5px", color: "#98a2b3", fontFamily: "'JetBrains Mono',monospace" }}>{runs.length} run{runs.length === 1 ? "" : "s"}{!live && " · demo data"}</span>
        </div>
      </div>

      {/* run list */}
      {runs.length === 0 ? (
        <div style={{ border: "1px dashed #ccd6ea", borderRadius: "16px", padding: "48px 40px", textAlign: "center", backgroundColor: "#f8fafe" }}>
          <div style={{ fontFamily: "'Hanken Grotesk',sans-serif", fontSize: "18px", color: "#283353", marginBottom: "6px" }}>Nothing selected</div>
          <div style={{ fontSize: "13.5px", color: "#838fa4" }}>Pick at least one dataset and one base model above.</div>
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: "18px" }}>
          {runs.map((r) => <RunCard key={`${r.id}/${r.model}`} run={r} live={live} />)}
        </div>
      )}

      {/* full report */}
      <div style={{ marginTop: "18px", background: "#ffffff", border: "1px solid #e5ebf4", borderRadius: "16px", padding: "18px 24px", display: "flex", alignItems: "center", justifyContent: "space-between", gap: "20px", flexWrap: "wrap", boxShadow: "0 1px 2px rgba(20,32,64,0.03)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          <span style={{ flex: "none", width: "46px", height: "46px", borderRadius: "11px", background: "linear-gradient(135deg,#e6eafc,#d6ddfb)", display: "flex", alignItems: "center", justifyContent: "center", backgroundImage: "radial-gradient(circle, rgba(47,67,224,0.22) 0.8px, transparent 1.1px)", backgroundSize: "5px 5px" }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2f43e0" strokeWidth="1.8"><rect x="4" y="3" width="16" height="18" rx="2" /><line x1="8" y1="8" x2="16" y2="8" /><line x1="8" y1="12" x2="16" y2="12" /><line x1="8" y1="16" x2="13" y2="16" /></svg>
          </span>
          <div>
            <div style={{ fontWeight: 600, fontSize: "14.5px", color: "#15203c" }}>Full evaluation report</div>
            <div style={{ fontSize: "13px", color: "#838fa4" }}>Detailed per-benchmark charts and per-checkpoint tables, rendered server-side.</div>
          </div>
        </div>
        <button className="hv-secondary" onClick={() => setReportOpen((o) => !o)} style={{ flex: "none", padding: "10px 18px", borderRadius: "10px", border: "1px solid #d3dbeb", background: "#f7f9fd", color: "#283353", font: "inherit", fontSize: "13.5px", fontWeight: 600, cursor: "pointer" }}>
          {reportOpen ? "Hide report" : "View full report"}
        </button>
      </div>

      {reportOpen && (live ? (
        <iframe title="report" src="/report" style={{ marginTop: "14px", width: "100%", height: "620px", border: "1px solid #e5ebf4", borderRadius: "14px", background: "#fff" }} />
      ) : (
        <div style={{ marginTop: "14px", border: "1px dashed #ccd6ea", borderRadius: "14px", padding: "40px 32px", textAlign: "center", backgroundColor: "#f8fafe" }}>
          <div style={{ fontFamily: "'Hanken Grotesk',sans-serif", fontSize: "18px", color: "#283353", marginBottom: "6px" }}>Report loads from the backend</div>
          <div style={{ fontSize: "13.5px", color: "#838fa4", maxWidth: "440px", margin: "0 auto", lineHeight: 1.5 }}>Connect to the server to view the inline HTML report from <span style={{ fontFamily: "'JetBrains Mono',monospace", color: "#2f43e0" }}>/report</span>.</div>
        </div>
      ))}
    </div>
  );
}
