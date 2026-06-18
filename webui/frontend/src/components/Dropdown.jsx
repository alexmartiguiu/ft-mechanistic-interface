import { useEffect, useRef, useState } from "react";

// Compact multi-select dropdown. Button shows "Label · n/m"; the popover is a checklist
// with an all/clear toggle. Closes on outside-click or Escape.
export default function Dropdown({ label, options, selected, onToggle, onAll }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    const onKey = (e) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("mousedown", onDoc); document.removeEventListener("keydown", onKey); };
  }, [open]);

  const allOn = options.length > 0 && options.every((o) => selected.has(o.id));
  const summary =
    selected.size === options.length ? "all" :
    selected.size === 0 ? "none" :
    selected.size === 1 ? (options.find((o) => selected.has(o.id))?.label ?? "1") :
    `${selected.size} of ${options.length}`;

  return (
    <div className="dd" ref={ref}>
      <span className="eyebrow dd-label">{label}</span>
      <button className="dd-btn" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
        <span>{summary}</span>
        <span className="dd-caret">{open ? "▴" : "▾"}</span>
      </button>
      {open && (
        <div className="dd-menu" role="listbox">
          <button className="dd-all" onClick={() => onAll(!allOn)}>
            {allOn ? "clear all" : "select all"}
          </button>
          {options.map((o) => (
            <button key={o.id} className="dd-item" role="option" aria-selected={selected.has(o.id)}
                    onClick={() => onToggle(o.id)}>
              <span className={`dd-check${selected.has(o.id) ? " on" : ""}`} />
              <span className="dd-text">{o.label}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
