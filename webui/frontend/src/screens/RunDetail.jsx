import NewExperiment from "./NewExperiment.jsx";
import RunNarrative from "./RunNarrative.jsx";

// One run's workspace — now a single unified conversation (no Conversation⇄Dashboard tabs).
// A draft run converses through the design agent; a previous run is narrated end-to-end by
// RunNarrative, which inlines every pipeline artifact (dataset → concepts → LoRA → training
// → evals → concept vectors → steering) as expandable cards.
export default function RunDetail({ run, live, onLaunched }) {
  return (
    <div>
      {run.kind === "draft" ? <NewExperiment onLaunched={onLaunched} /> : <RunNarrative run={run} live={live} />}
    </div>
  );
}
