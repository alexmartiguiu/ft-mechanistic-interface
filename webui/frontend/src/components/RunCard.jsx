import Plot from "./Plot.jsx";
import Deltas from "./Deltas.jsx";
import { MODEL_LABEL } from "../api.js";

// One (dataset × model) run: header, two unified plots (each with its own legend-filter),
// a base→final delta strip. `evalSeries` + `palette` come from /api/catalog; `concepts`
// is the dataset's concept list.
export default function RunCard({ dataset, label, sub, model, metrics, evalSeries, palette, concepts }) {
  const conceptLegend = (concepts || []).map((c, i) => ({
    key: c,
    label: c.replace(/_/g, " "),
    color: palette[i % palette.length],
  }));

  return (
    <section className="card">
      <div className="card-head">
        <h2>{label}</h2>
        <span className="model">{MODEL_LABEL[model] || model}</span>
        {sub && <span className="sub">{sub}</span>}
      </div>
      <div className="plots">
        <Plot dataset={dataset} model={model} kind="eval" name="Training & evals"
              note="accuracy · refusal · loss" legend={evalSeries} />
        <Plot dataset={dataset} model={model} kind="monitor" name="Concept vectors"
              note="projection ⟨h, v̂⟩" legend={conceptLegend} />
      </div>
      <Deltas metrics={metrics} />
    </section>
  );
}
