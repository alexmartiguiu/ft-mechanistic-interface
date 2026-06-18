import Plot from "./Plot.jsx";
import Deltas from "./Deltas.jsx";
import { MODEL_LABEL } from "../api.js";

// One (dataset × model) run: header, two unified plots, a base→final delta strip.
export default function RunCard({ dataset, label, sub, model, metrics }) {
  return (
    <section className="card">
      <div className="card-head">
        <h2>{label}</h2>
        <span className="model">{MODEL_LABEL[model] || model}</span>
        {sub && <span className="sub">{sub}</span>}
      </div>
      <div className="plots">
        <Plot dataset={dataset} model={model} kind="eval" name="Capability & safety"
              note="MMLU · TruthfulQA · refusal" />
        <Plot dataset={dataset} model={model} kind="monitor" name="Concept vectors"
              note="projection ⟨h, v̂⟩" />
      </div>
      <Deltas metrics={metrics} />
    </section>
  );
}
