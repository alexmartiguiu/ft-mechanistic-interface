import { useEffect, useMemo, useState } from "react";
import { plotUrl } from "../api.js";

// One captioned matplotlib SVG. `series` = array of active series keys, or null for all.
// `show` toggles the whole panel — the run-level RunFilters drive both plots.
export default function Plot({ dataset, model, kind, name, note, series, show }) {
  const [state, setState] = useState("loading");

  // omit ?series when everything is on → hits the cached "all" render
  const src = useMemo(() => {
    const base = plotUrl(dataset, model, kind);
    if (series == null) return base;
    return `${base}?series=${series.map(encodeURIComponent).join(",")}`;
  }, [dataset, model, kind, series]);

  useEffect(() => { setState("loading"); }, [src]);

  return (
    <div className="plot">
      <div className="plot-cap">
        <span className="name">{name}</span>
        {note && <span className="note">{note}</span>}
      </div>
      <div className="plot-frame">
        {!show ? (
          <div className="plot-empty">hidden</div>
        ) : state === "error" ? (
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
