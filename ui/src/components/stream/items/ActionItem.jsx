import { useState } from "react";

// A proposed action button. Appears when a condition is programmatically triggered; the
// USER'S click is what advances the pipeline (propose-only — the agent never acts itself).
export default function ActionItem({ item }) {
  const [done, setDone] = useState(false);
  const act = () => { setDone(true); item.onAct && item.onAct(); };
  return (
    <div className="si">
      <div className="who">hedda · proposes</div>
      <div className="si-action">
        <div className="at">{item.title}</div>
        <button className={`btn ${item.variant || "primary"} sm`} disabled={done} onClick={act}>
          {done ? "✓ " + (item.doneLabel || "started") : item.label}
        </button>
      </div>
    </div>
  );
}
