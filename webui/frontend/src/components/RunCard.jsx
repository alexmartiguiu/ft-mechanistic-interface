import { useMemo, useState } from "react";
import Plot from "./Plot.jsx";
import Deltas from "./Deltas.jsx";
import RunFilters from "./RunFilters.jsx";
import { MODEL_LABEL } from "../api.js";

// One (dataset × model) run: header, two unified plots + a per-run filter checklist on
// the right that toggles what each plot draws, then a base→final delta strip.
// `evalSeries` (with .group) + `palette` come from /api/catalog; `concepts` is the
// dataset's concept list.
const GROUPS = [
  { key: "training",   label: "Training curves", sub: "train · eval loss",        plot: "eval" },
  { key: "capability", label: "Capability evals", sub: "MMLU-Pro · TruthfulQA",    plot: "eval" },
  { key: "safety",     label: "Safety evals",     sub: "HarmBench · StrongREJECT", plot: "eval" },
  { key: "concepts",   label: "Concept vectors",  sub: "projection ⟨h, v̂⟩",       plot: "monitor" },
];

export default function RunCard({ dataset, label, sub, model, metrics, evalSeries }) {
  // map each eval group → its server series keys (from catalog metadata)
  const groupSeries = useMemo(() => {
    const by = {};
    for (const s of evalSeries) (by[s.group] ||= []).push(s.key);
    return by;
  }, [evalSeries]);

  const [active, setActive] = useState(() => new Set(GROUPS.map((g) => g.key)));
  const toggle = (key) =>
    setActive((s) => {
      const n = new Set(s);
      n.has(key) ? n.delete(key) : n.add(key);
      return n;
    });

  // left plot: union of series for the active eval groups (null = all on → cached "all" render)
  const evalGroups = GROUPS.filter((g) => g.plot === "eval");
  const activeEval = evalGroups.filter((g) => active.has(g.key));
  const evalSel =
    activeEval.length === evalGroups.length
      ? null
      : activeEval.flatMap((g) => groupSeries[g.key] || []);

  return (
    <section className="card">
      <div className="card-head">
        <h2>{label}</h2>
        <span className="model">{MODEL_LABEL[model] || model}</span>
        {sub && <span className="sub">{sub}</span>}
      </div>
      <div className="plots">
        <Plot dataset={dataset} model={model} kind="eval" name="Training & evals"
              note="accuracy · refusal · loss" series={evalSel} show={activeEval.length > 0} />
        <Plot dataset={dataset} model={model} kind="monitor" name="Concept vectors"
              note="projection ⟨h, v̂⟩" series={null} show={active.has("concepts")} />
        <RunFilters groups={GROUPS} active={active} onToggle={toggle} />
      </div>
      <Deltas metrics={metrics} />
    </section>
  );
}
