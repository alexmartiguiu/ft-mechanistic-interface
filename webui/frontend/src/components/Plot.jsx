import { useState } from "react";
import { plotUrl } from "../api.js";

// One captioned matplotlib SVG (lazy-loaded as an <img>). kind = "eval" | "monitor".
export default function Plot({ dataset, model, kind, name, note }) {
  const [state, setState] = useState("loading"); // loading | ok | error
  return (
    <div className="plot">
      <div className="plot-cap">
        <span className="name">{name}</span>
        {note && <span className="note">{note}</span>}
      </div>
      <div className="plot-frame">
        {state === "error" ? (
          <div className="plot-empty">no series</div>
        ) : (
          <>
            {state === "loading" && <div className="plot-skeleton">drawing…</div>}
            <img
              src={plotUrl(dataset, model, kind)}
              alt={`${name} — ${dataset} / ${model}`}
              loading="lazy"
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
