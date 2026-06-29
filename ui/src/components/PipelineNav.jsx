/* The pipeline-steps menu that lives top-right of the stage. Current step is grey;
   unlocked steps (run once in the forward pass) are clickable to go back. */
export default function PipelineNav({ steps, current, unlocked, onJump }) {
  return (
    <div className="pipenav">
      {steps.map((s, i) => {
        const isCurrent = s.id === current;
        const isUnlocked = unlocked.has(s.id);
        const isDone = isUnlocked && !isCurrent;
        const cls = ["pipestep", isCurrent ? "current" : "", isUnlocked ? "unlocked" : "", isDone ? "done" : ""]
          .filter(Boolean).join(" ");
        return (
          <span key={s.id} style={{ display: "inline-flex", alignItems: "center" }}>
            {i > 0 && <span className="arrow">›</span>}
            <span className={cls} onClick={() => isUnlocked && onJump(s.id)}>
              <span className="ix">{isDone ? "✓" : i + 1}</span>
              {s.label}
            </span>
          </span>
        );
      })}
    </div>
  );
}
