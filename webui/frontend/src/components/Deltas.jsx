// Compact base→final readout for the metric battery. `metrics` = [{key,label,base,final,delta}].
const fmt = (x) => (x == null ? "—" : x.toFixed(3));

function sign(key, delta) {
  if (delta == null) return "";
  // refusal up = safer (good); capability/truthfulness up = good. All "up = green" here.
  return delta >= 0 ? "up" : "down";
}

export default function Deltas({ metrics }) {
  if (!metrics || !metrics.length) return null;
  return (
    <div className="deltas">
      {metrics.map((m) => (
        <div className="delta" key={m.key}>
          <span className="k">{m.label}</span>
          <span className="v">
            {fmt(m.base)}<span className="arrow">→</span>{fmt(m.final)}
          </span>
          <span className={`d ${sign(m.key, m.delta)}`}>
            {m.delta == null ? "" : `${m.delta >= 0 ? "+" : ""}${m.delta.toFixed(3)}`}
          </span>
        </div>
      ))}
    </div>
  );
}
