import { useEffect, useState } from "react";
import Markdown from "../../Markdown.jsx";

/* Hedda working through a procedure, checking steps off in colour as each completes.
   This is the "thinking" tell for multi-step work (e.g. minting persona directions):
   the active step breathes, finished steps settle into a sage check. Counts as
   thinking, so the rail's Hide-thinking toggle collapses it. */
export default function StepsItem({ item }) {
  const steps = item.steps || [];
  const [done, setDone] = useState(0);

  useEffect(() => {
    const reduce = typeof window !== "undefined" && window.matchMedia
      && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) { setDone(steps.length); return; }
    const each = item.interval || 520;
    const timers = steps.map((_, i) => setTimeout(() => setDone(i + 1), each * (i + 1)));
    return () => timers.forEach(clearTimeout);
  }, [steps.length]);

  return (
    <div className="si si-steps">
      <div className="who">Hedda</div>
      {item.lead && <p className="ss-lead"><Markdown>{item.lead}</Markdown></p>}
      <ul className="ss-list">
        {steps.map((s, i) => {
          const state = i < done ? "done" : i === done ? "active" : "todo";
          return (
            <li key={i} className={`ss-step ${state}`}>
              <span className="ss-mark" aria-hidden="true">
                {state === "done"
                  ? "✓"
                  : state === "active"
                    ? <span className="ss-dots"><i /><i /><i /></span>
                    : <span className="ss-pend" />}
              </span>
              <span className="ss-label"><Markdown>{s}</Markdown></span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
