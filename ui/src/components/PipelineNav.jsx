/* The pipeline stepper, top-right of the stage. The pipeline is a real linear
   sequence, so it reads as a connected track: a hairline joins the step nodes and
   turns green across the segments already cleared. Done = green check, current =
   solid ink, upcoming = hollow. State lives in colour + weight, not in boxes.
   Visited steps stay clickable so you can walk back.
   `completed` marks a step done even when it is the current one — used to light the
   final Checkout node (its hue is terracotta) once the run has been wrapped up. */
const HUE = {
  setup: "var(--step-setup)", audit: "var(--step-audit)",
  insights: "var(--step-insights)", checkout: "var(--step-checkout)",
};

export default function PipelineNav({ steps, current, unlocked, completed, onJump }) {
  const curIdx = steps.findIndex((s) => s.id === current);
  return (
    <div className="pipenav" role="list" aria-label="Pipeline progress">
      {steps.map((s, i) => {
        const isCurrent = s.id === current;
        const isCompleted = !!(completed && completed.has(s.id));
        const isDone = isCompleted || i < curIdx;
        const canNav = unlocked.has(s.id) && !isCurrent;
        // a completed step reads as done even while it is current (the terminal Checkout)
        const state = isCompleted ? "done" : isCurrent ? "current" : isDone ? "done" : "todo";
        const jump = () => canNav && onJump(s.id);
        return (
          <div key={s.id} role="listitem"
            className={`pstep ${state}${canNav ? " nav" : ""}`}
            style={{ "--node": HUE[s.id] || "var(--terracotta)" }}
            onClick={jump}
            tabIndex={canNav ? 0 : -1}
            onKeyDown={(e) => canNav && (e.key === "Enter" || e.key === " ") && (e.preventDefault(), jump())}
            aria-current={isCurrent ? "step" : undefined}>
            {i > 0 && <span className={`pseg${i <= curIdx ? " fill" : ""}`} aria-hidden="true" />}
            <span className="pnode" aria-hidden="true">{isDone ? "✓" : ""}</span>
            <span className="plabel">{s.label}</span>
          </div>
        );
      })}
    </div>
  );
}
