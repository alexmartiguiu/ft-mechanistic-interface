import { useEffect, useMemo, useState } from "react";
import { getCatalog, getOverview } from "./api.js";
import { demoCatalog, demoOverview, fullDir } from "./data/demo.js";
import Sidebar from "./components/Sidebar.jsx";
import PageHeader from "./components/PageHeader.jsx";
import RunDetail from "./screens/RunDetail.jsx";

const MODEL_FALLBACK = { "qwen-7b": "Qwen2.5-7B-Instruct", "apertus-8b": "Apertus-8B-Instruct" };

export default function App() {
  const [catalog, setCatalog] = useState(null);
  const [overview, setOverview] = useState({});
  const [live, setLive] = useState(false);
  const [loadedModel, setLoadedModel] = useState(null);
  const [collapsed, setCollapsed] = useState(false);
  const [drafts, setDrafts] = useState([]);          // session-created draft runs
  const [selectedId, setSelectedId] = useState(null);
  const [draftSeq, setDraftSeq] = useState(0);

  // load the run catalog + drift overview; fall back to demo data offline
  useEffect(() => {
    let cancelled = false;
    function apply(c, o, isLive) { if (cancelled) return; setCatalog(c); setOverview(o); setLive(isLive); if (isLive) setLoadedModel(null); }
    getCatalog()
      .then((c) => getOverview().then((o) => {
        const by = {}; for (const a of o.apps) by[a.app] = a.metrics; apply(c, by, true);
      }))
      .catch(() => apply(demoCatalog(), demoOverview(), false));
    return () => { cancelled = true; };
  }, []);

  const modelLabel = (id) => (catalog?.models?.find((m) => m.id === id) || {}).label || MODEL_FALLBACK[id] || id;

  const previousRuns = useMemo(() => {
    if (!catalog) return [];
    const out = [];
    for (const d of catalog.datasets)
      for (const m of d.models)
        out.push({ id: `${d.id}/${m}`, kind: "previous", dataset: d.id, model: m, label: d.label, sub: d.sub, modelLabel: modelLabel(m), concepts: d.concepts, metrics: overview[fullDir(d.id, m)] });
    return out;
  }, [catalog, overview]);

  const runs = useMemo(() => [...drafts, ...previousRuns], [drafts, previousRuns]);

  // default selection once runs are available
  useEffect(() => {
    if (selectedId == null && runs.length) setSelectedId(runs[0].id);
  }, [runs, selectedId]);

  const selectedRun = runs.find((r) => r.id === selectedId) || null;

  function onNew() {
    const n = draftSeq + 1;
    setDraftSeq(n);
    const draft = { id: `draft-${n}`, kind: "draft", label: "New experiment", modelLabel: "design a run", metrics: undefined };
    setDrafts((d) => [draft, ...d]);
    setSelectedId(draft.id);
  }

  // a draft launched → if its (dataset × model) already exists as a real run, select it so
  // its unified narrative shows; otherwise upgrade the draft in place.
  function onLaunched({ dataset, model }) {
    const realId = `${dataset}/${model}`;
    const exists = previousRuns.find((r) => r.id === realId);
    if (exists) { setSelectedId(realId); return; }
    setDrafts((ds) => ds.map((d) => (d.id === selectedId
      ? { ...d, dataset, model, label: dataset, modelLabel: modelLabel(model), metrics: overview[fullDir(dataset, model)] }
      : d)));
  }

  const meta = selectedRun
    ? selectedRun.kind === "draft"
      ? { kicker: "New experiment", title: "Design a run", sub: "Chat with the hedda agent to produce the three config files, then launch." }
      : { kicker: "Run", title: selectedRun.label, sub: selectedRun.sub || "Conversation registry and drift dashboard for this run." }
    : { kicker: "hedda", title: "No run selected", sub: "Start one with “New experiment”." };

  return (
    <div style={{ display: "flex", height: "100vh", width: "100%", overflow: "hidden", fontFamily: "'Hanken Grotesk',system-ui,sans-serif", color: "#15203c", background: "#f6f8fc", WebkitFontSmoothing: "antialiased" }}>
      <Sidebar runs={runs} selectedId={selectedId} onSelect={setSelectedId} onNew={onNew} collapsed={collapsed} onToggleCollapse={() => setCollapsed((c) => !c)} live={live} loadedModel={loadedModel} />

      <main style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", overflow: "hidden", background: "#f6f8fc" }}>
        <PageHeader
          kicker={meta.kicker}
          title={meta.title}
          subtitle={meta.sub}
          actions={
            <span style={{ display: "inline-flex", alignItems: "center", gap: "7px", padding: "6px 12px", borderRadius: "999px", background: "#eef3fa", border: "1px solid #dce4f0", fontSize: "12.5px", fontWeight: 500, color: "#3a465e", whiteSpace: "nowrap" }}>
              <span style={{ width: "7px", height: "7px", borderRadius: "50%", background: "#5b6ee0" }} />
              Qwen2.5-7B-Instruct
            </span>
          }
        />
        <div style={{ flex: 1, overflowY: "auto", overflowX: "hidden" }}>
          <div style={{ padding: "26px 38px 56px", maxWidth: "1180px", margin: "0 auto" }}>
            {selectedRun ? (
              <RunDetail run={selectedRun} live={live} onLaunched={onLaunched} />
            ) : (
              <div style={{ fontSize: "13.5px", color: "#a6aebe" }}>Loading runs…</div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
