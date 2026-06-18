import { useMemo, useState } from "react";
import Plot from "./Plot.jsx";
import Deltas from "./Deltas.jsx";
import RunFilters from "./RunFilters.jsx";
import { MODEL_LABEL } from "../api.js";

// One (dataset × model) run: header, two unified plots + a per-run legend/filter to the
// right. The filter's children carry each line's colour (matching the server render) and
// toggle individual series; group headers toggle a whole category. `evalSeries` (.group,
// .color) + `concepts`/`conceptPalette` come from /api/catalog.
const GROUP_META = [
  { key: "training",   label: "Training curves", plot: "eval" },
  { key: "capability", label: "Capability evals", plot: "eval" },
  { key: "safety",     label: "Safety evals",     plot: "eval" },
  { key: "concepts",   label: "Concept vectors",  plot: "monitor" },
];

export default function RunCard({ dataset, label, sub, model, metrics, evalSeries, concepts, conceptPalette }) {
  const groups = useMemo(() => {
    const byGroup = {};
    for (const s of evalSeries)
      (byGroup[s.group] ||= []).push({ key: s.key, label: s.label, color: s.color });
    const conceptChildren = (concepts || []).map((c, i) => ({
      key: c,
      label: c.replace(/_/g, " "),
      color: conceptPalette[i % conceptPalette.length],
    }));
    return GROUP_META.map((g) => ({
      ...g,
      children: g.key === "concepts" ? conceptChildren : byGroup[g.key] || [],
    }));
  }, [evalSeries, concepts, conceptPalette]);

  const allKeys = useMemo(() => groups.flatMap((g) => g.children.map((c) => c.key)), [groups]);
  const [active, setActive] = useState(() => new Set(allKeys));

  const toggleSeries = (key) =>
    setActive((s) => {
      const n = new Set(s);
      n.has(key) ? n.delete(key) : n.add(key);
      return n;
    });
  const toggleGroup = (g, on) =>
    setActive((s) => {
      const n = new Set(s);
      for (const c of g.children) (on ? n.add(c.key) : n.delete(c.key));
      return n;
    });

  // per-plot series selection: null when every line in that plot is on (cached "all" render)
  const sel = (plot) => {
    const keys = groups.filter((g) => g.plot === plot).flatMap((g) => g.children.map((c) => c.key));
    const on = keys.filter((k) => active.has(k));
    return { series: on.length === keys.length ? null : on, show: on.length > 0 };
  };
  const evalP = sel("eval");
  const monP = sel("monitor");

  return (
    <section className="card">
      <div className="card-head">
        <h2>{label}</h2>
        <span className="model">{MODEL_LABEL[model] || model}</span>
        {sub && <span className="sub">{sub}</span>}
      </div>
      <div className="plots">
        <Plot dataset={dataset} model={model} kind="eval" name="Training & evals"
              note="accuracy · refusal · loss" series={evalP.series} show={evalP.show} />
        <Plot dataset={dataset} model={model} kind="monitor" name="Concept vectors"
              note="projection ⟨h, v̂⟩" series={monP.series} show={monP.show} />
        <RunFilters groups={groups} active={active}
                    onToggleSeries={toggleSeries} onToggleGroup={toggleGroup} />
      </div>
      <Deltas metrics={metrics} />
    </section>
  );
}
