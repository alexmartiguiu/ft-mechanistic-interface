import { useEffect, useMemo, useState } from "react";
import { plotUrl } from "../api.js";

// One captioned matplotlib SVG — the exact server render (data/<run>/results → plots.py).
// `series` = array of active series keys, or null for "all" (hits the cached all-render).
// `show` toggles the whole panel; the run-level RunFilters drive both plots.
export default function Plot({ dataset, model, kind, name, note, series, show }) {
  const [state, setState] = useState("loading");

  // omit ?series when everything is on → cached "all" render on the server
  const src = useMemo(() => {
    if (!dataset || !model) return null;
    const base = plotUrl(dataset, model, kind);
    if (series == null) return base;
    return `${base}?series=${series.map(encodeURIComponent).join(",")}`;
  }, [dataset, model, kind, series]);

  useEffect(() => { setState("loading"); }, [src]);

  const cap = (
    <div style={{ display: "flex", alignItems: "baseline", gap: "8px", marginBottom: "8px" }}>
      <span style={{ fontSize: "12px", fontWeight: 600, color: "#48546e" }}>{name}</span>
      {note && <span style={{ fontSize: "11.5px", color: "#a6aebe" }}>{note}</span>}
    </div>
  );
  const frame = (child) => (
    <div style={{ flex: 1, minWidth: 0 }}>
      {cap}
      <div style={{ position: "relative", minHeight: "190px", display: "flex", alignItems: "center", justifyContent: "center" }}>
        {child}
      </div>
    </div>
  );

  if (!src) return frame(<span style={MSG}>no run selected</span>);
  if (!show) return frame(<span style={MSG}>hidden</span>);
  if (state === "error") return frame(<span style={MSG}>no series</span>);
  return frame(
    <>
      {state === "loading" && <span style={{ ...MSG, position: "absolute" }}>drawing…</span>}
      <img
        src={src}
        alt={`${name} — ${dataset} / ${model}`}
        decoding="async"
        style={{ display: state === "ok" ? "block" : "none", width: "100%", height: "auto" }}
        onLoad={() => setState("ok")}
        onError={() => setState("error")}
      />
    </>
  );
}

const MSG = { fontSize: "12.5px", color: "#a6aebe" };
