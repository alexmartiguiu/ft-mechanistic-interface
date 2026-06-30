/* A nested subagent's live trace (e.g. the concept-proposer), rendered indented
   under its own header. Upserted by `ref` in LiveRunView: a panel opens on `start`,
   each tool call appends a step, and `done` swaps the spinner for a status line. */
import ArxivLogo from "../../ArxivLogo.jsx";

const I = {
  // the classic web globe (Lucide) — a ring with meridian + equator lines, no fill
  search: (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="10" />
      <path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20" />
      <path d="M2 12h20" />
    </svg>
  ),
  read: (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="M4 2.2h5l3 3v8.6H4z" /><line x1="6" y1="7.5" x2="10" y2="7.5" />
      <line x1="6" y1="10" x2="10" y2="10" />
    </svg>
  ),
  think: (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <circle cx="8" cy="8" r="2.4" />
    </svg>
  ),
  done: (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="M3.5 8.4l3 3 6-6.4" />
    </svg>
  ),
  error: (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="M8 2.5l6 11H2z" /><line x1="8" y1="7" x2="8" y2="10" /><circle cx="8" cy="11.8" r="0.5" />
    </svg>
  ),
};

const Spark = (
  <svg viewBox="0 0 16 16" aria-hidden="true" className="sa-spark">
    <path d="M8 1.5l1.4 4.1 4.1 1.4-4.1 1.4L8 12.5 6.6 8.4 2.5 7l4.1-1.4z" />
    <path d="M12.8 10.5l.6 1.7 1.7.6-1.7.6-.6 1.7-.6-1.7-1.7-.6 1.7-.6z" />
  </svg>
);

export default function SubagentItem({ item }) {
  const { title = "Concept Proposal", agent = "concept-proposer",
          steps = [], status, done } = item;
  const web = /search|web|browse/i.test(`${agent} ${title}`);
  const who = web ? "Hedda searches the web" : "Hedda researches";
  return (
    <div className="si si-subagent">
      <div className="who"><span className="who-t">{who}</span></div>
      <div className="sa-card">
        <div className="sa-head">
          <span className="sa-badge">{Spark}</span>
          <span className="sa-title">{title}</span>
          <span className="sa-type mono">{agent}</span>
          <span className="sa-status">
            {!done && <span className="think-dots" aria-label="working"><i /><i /><i /></span>}
          </span>
        </div>
        {steps.length > 0 && (
          <ul className="sa-steps">
            {steps.map((s, i) => (
              <li className={`sa-step${i === steps.length - 1 && !done ? " cur" : ""}`} key={i}>
                {s.icon === "paper" ? (
                  <span className="sa-paper">
                    <ArxivLogo height={11} />
                    <span className="sa-paper-t">{s.title || s.text}</span>
                    {s.subtitle && <span className="sa-paper-a">{s.subtitle}</span>}
                  </span>
                ) : (
                  <>
                    <span className="sa-ic">{I[s.icon] || I.think}</span>
                    <span className="sa-tx">{s.text}</span>
                  </>
                )}
              </li>
            ))}
          </ul>
        )}
        {done && (
          <div className="sa-foot sa-done"><span className="sa-ic">{I.done}</span>{status || "done"}</div>
        )}
      </div>
    </div>
  );
}
