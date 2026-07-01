import { useState } from "react";
import Who from "../Who.jsx";
import Markdown from "../../Markdown.jsx";

// Multiple-choice prompt (single or multi-select). Mirrors the agent's ask_user tool.
export default function QuestionItem({ item }) {
  const init = new Set(item.options.filter((o) => o.default).map((o) => o.label));
  const [sel, setSel] = useState(init);
  const [done, setDone] = useState(false);
  // Placeholder free-form text for options flagged `freeform` (e.g. "Other"). Not wired to onSubmit yet.
  const [freeform, setFreeform] = useState("");

  const pick = (label) => {
    if (done) return;
    setSel((s) => {
      if (item.multiSelect) {
        const n = new Set(s); n.has(label) ? n.delete(label) : n.add(label); return n;
      }
      return new Set([label]);
    });
  };
  const submit = () => {
    setDone(true);
    item.onSubmit && item.onSubmit([...sel]);
  };

  return (
    <div className="si">
      <Who>Hedda asks</Who>
      <div className="si-question">
        <div className="q"><Markdown>{item.question}</Markdown></div>
        <div className="opts">
          {item.options.map((o) => {
            const raw = (o.description || "").trim();
            // Badge only the option the model actually recommended: an explicit flag, or a
            // trailing "(recommended)" marker it appends to that option's description. Guard
            // against incidental negations elsewhere in the text — e.g. the ship-as-is option's
            // "Not recommended for a clinical deployment." must NOT read as recommended.
            const rec = o.recommended
              || (/\brecommended[)\.\s]*$/i.test(raw) && !/\bnot\s+recommended\b/i.test(raw));
            const desc = raw.replace(/\s*\(?recommended\)?\.?\s*$/i, "").trim();
            return (
              <div key={o.label} className={`opt ${sel.has(o.label) ? "on" : ""}`} onClick={() => pick(o.label)}>
                <span className="box">{sel.has(o.label) ? "✓" : ""}</span>
                <span className="col">
                  <span className="ol">{o.label}{rec && <span className="rec-badge">Recommended</span>}</span>
                  {desc && <span className="od"><Markdown>{desc}</Markdown></span>}
                  {o.freeform && sel.has(o.label) && (
                    <input
                      className="opt-freeform"
                      type="text"
                      placeholder="Describe your approach…"
                      value={freeform}
                      disabled={done}
                      onClick={(e) => e.stopPropagation()}
                      onChange={(e) => setFreeform(e.target.value)}
                      style={{ marginTop: 6, width: "100%", boxSizing: "border-box" }}
                    />
                  )}
                </span>
              </div>
            );
          })}
        </div>
        {!done
          ? <button className="btn primary sm" style={{ marginTop: 10 }} onClick={submit}>Confirm</button>
          : <div className="submitted">✓ {[...sel].length} selected · {item.confirmLabel || "confirmed"}</div>}
      </div>
    </div>
  );
}
