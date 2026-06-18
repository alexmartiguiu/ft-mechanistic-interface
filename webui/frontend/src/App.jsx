import { useEffect, useMemo, useState } from "react";
import { getCatalog, getOverview } from "./api.js";
import Dropdown from "./components/Dropdown.jsx";
import RunCard from "./components/RunCard.jsx";

const SLUG = "__apertus-8b-instruct-2509";
const fullDir = (dataset, model) => (model === "apertus-8b" ? dataset + SLUG : dataset);

function useSelection() {
  const [sel, setSel] = useState(() => new Set());
  const toggle = (id) =>
    setSel((s) => {
      const n = new Set(s);
      n.has(id) ? n.delete(id) : n.add(id);
      return n;
    });
  const setAll = (ids, on) => setSel(() => (on ? new Set(ids) : new Set()));
  return [sel, toggle, setAll];
}

export default function App() {
  const [catalog, setCatalog] = useState(null);
  const [overview, setOverview] = useState({});
  const [err, setErr] = useState(null);
  const [dsSel, dsToggle, dsSetAll] = useSelection();
  const [mdSel, mdToggle, mdSetAll] = useSelection();

  useEffect(() => {
    getCatalog()
      .then((c) => {
        setCatalog(c);
        dsSetAll(c.datasets.map((d) => d.id), true);  // default: all active
        mdSetAll(c.models.map((m) => m.id), true);
      })
      .catch((e) => setErr(String(e)));
    getOverview()
      .then((o) => {
        const by = {};
        for (const a of o.apps) by[a.app] = a.metrics;
        setOverview(by);
      })
      .catch(() => {});
  }, []);

  const dsOptions = useMemo(
    () => (catalog?.datasets || []).map((d) => ({ id: d.id, label: d.label })),
    [catalog]
  );
  const mdOptions = useMemo(
    () => (catalog?.models || []).map((m) => ({ id: m.id, label: m.label })),
    [catalog]
  );

  // only the selected dataset × model combinations that actually exist
  const runs = useMemo(() => {
    if (!catalog) return [];
    const out = [];
    for (const d of catalog.datasets) {
      if (!dsSel.has(d.id)) continue;
      for (const m of d.models) {
        if (!mdSel.has(m)) continue;
        out.push({ ...d, model: m, metrics: overview[fullDir(d.id, m)] });
      }
    }
    return out;
  }, [catalog, dsSel, mdSel, overview]);

  return (
    <div className="shell">
      <header className="masthead">
        <div className="mark">廻</div>
        <h1>Fine-tuning drift explorer</h1>
        <p className="lede">
          How capability, safety and latent persona traits move over a fine-tuning run —
          per dataset and base model. Curves unify the full run with the dense early-step
          pass; the dashed line marks the early-stopping point (minimum validation loss).
        </p>
      </header>

      {err && <div className="empty-state">could not reach the API · {err}</div>}

      {catalog && (
        <div className="toolbar">
          <Dropdown label="Dataset" options={dsOptions} selected={dsSel}
                    onToggle={dsToggle} onAll={(on) => dsSetAll(dsOptions.map((o) => o.id), on)} />
          <Dropdown label="Base model" options={mdOptions} selected={mdSel}
                    onToggle={mdToggle} onAll={(on) => mdSetAll(mdOptions.map((o) => o.id), on)} />
          <span className="toolbar-count">{runs.length} run{runs.length === 1 ? "" : "s"}</span>
        </div>
      )}

      {catalog && runs.length === 0 && (
        <div className="empty-state">
          <div className="big">Nothing selected</div>
          <div>Pick at least one dataset and one base model above.</div>
        </div>
      )}

      <div className="runs">
        {runs.map((r) => (
          <RunCard key={`${r.id}/${r.model}`} dataset={r.id} label={r.label} sub={r.sub}
                   model={r.model} metrics={r.metrics}
                   evalSeries={catalog.eval_series} palette={catalog.palette}
                   concepts={r.concepts} />
        ))}
      </div>

      <footer className="footnote">
        ftmi · plots rendered server-side (matplotlib) · <a href="/report">full static report ↗</a> ·
        <a href="/legacy"> legacy console ↗</a>
      </footer>
    </div>
  );
}
