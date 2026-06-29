// A highlighted metric callout (e.g. "+27 HarmBench refusal"). Green = safe move.
export default function MetricItem({ item }) {
  return (
    <div className="si">
      <div className="who">Hedda</div>
      <div className={`si-metric ${item.tone === "bad" ? "bad" : ""}`}>
        <span className="mv">{item.value}</span>
        <span className="ml">{item.label}</span>
      </div>
    </div>
  );
}
