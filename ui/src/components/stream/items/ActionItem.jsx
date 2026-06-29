import { useState } from "react";
import Markdown from "../../Markdown.jsx";

// A proposed action button. Appears when a condition is programmatically triggered; the
// USER'S click is what advances the pipeline (propose-only — the agent never acts itself).
// Once taken, it settles into a shaded-checkmark step, matching the walk-through's done states.
export default function ActionItem({ item }) {
  const [done, setDone] = useState(false);
  const act = () => { setDone(true); item.onAct && item.onAct(); };
  return (
    <div className="si">
      <div className="who">Hedda · proposes</div>
      <div className="si-action">
        <div className="at"><Markdown>{item.title}</Markdown></div>
        {done
          ? <span className={`si-done ${item.variant === "good" ? "good" : ""}`}>
              <span className="ck">✓</span>{item.doneLabel || "started"}
            </span>
          : <button className={`btn ${item.variant || "primary"} sm`} onClick={act}>{item.label}</button>}
      </div>
    </div>
  );
}
