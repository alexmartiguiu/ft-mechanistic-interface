import { useMemo, useState } from "react";
import { plotUrl } from "../api.js";

// One captioned matplotlib SVG with its own legend-toggles. `legend` = [{key,label,color}].
// Clicking a legend chip filters that series out of the server-rendered plot. The legend
// IS the filter — default all active.
export default function Plot({ dataset, model, kind, name, note, legend }) {
  const [active, setActive] = useState(() => new Set(legend.map((l) => l.key)));
  const [state, setState] = useState("loading");

  const toggle = (key) =>
    setActive((s) => {
      const n = new Set(s);
      n.has(key) ? n.delete(key) : n.add(key);
      return n;
    });

  // omit ?series when everything is on → hits the cached "all" render
  const src = useMemo(() => {
    const allOn = active.size === legend.length;
    const base = plotUrl(dataset, model, kind);
    if (allOn) return base;
    return `${base}?series=${[...active].map(encodeURIComponent).join(",")}`;
  }, [dataset, model, kind, active, legend.length]);

  return (
    <div className="plot">
      <div className="plot-cap">
        <span className="name">{name}</span>
        {note && <span className="note">{note}</span>}
      </div>
      <div className="legend">
        {legend.map((l) => (
          <button
            key={l.key}
            className={`leg${active.has(l.key) ? "" : " off"}`}
            onClick={() => toggle(l.key)}
            title={active.has(l.key) ? "hide" : "show"}
          >
            <span className="swatch" style={{ background: l.color }} />
            {l.label}
          </button>
        ))}
      </div>
      <div className="plot-frame">
        {state === "error" ? (
          <div className="plot-empty">no series</div>
        ) : (
          <>
            {state === "loading" && <div className="plot-skeleton">drawing…</div>}
            <img
              src={src}
              alt={`${name} — ${dataset} / ${model}`}
              decoding="async"
              style={{ display: state === "ok" ? "block" : "none" }}
              onLoad={() => setState("ok")}
              onError={() => setState("error")}
            />
          </>
        )}
      </div>
    </div>
  );
}
