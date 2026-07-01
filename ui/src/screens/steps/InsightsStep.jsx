import { useEffect, useRef } from "react";
import DualPlot from "../../components/DualPlot.jsx";
import { useTimeline, windowFrac } from "../../lib/hooks.js";

/* Realign: a SINGLE stacked plot (loss/evals on top, projections below) that hosts
   both runs. Two staged timelines drive the live fill:
     phase 1 (`active`)      — the normal fine-tune: loss → evals → projections
     phase 2 (`steerActive`) — the preventive-steered overlay: loss → evals → projections
   Each phase's three stages occupy the [a,b] windows below; `onTrainRevealed` /
   `onSteerRevealed` fire once when a phase finishes so the rail can narrate on cue. */
const P1 = 5200, P2 = 5400;
const WIN1 = { loss: [0, 1400], eval: [1650, 3050], proj: [3300, 5200] };
const WIN2 = { loss: [0, 1400], eval: [1650, 3050], proj: [3400, 5400] };

export default function InsightsStep({
  run, steerRun = null, active = false, steerActive = false,
  onTrainRevealed, onSteerRevealed, earlyStopShown = false,
}) {
  const t1 = useTimeline(active, P1);
  const t2 = useTimeline(steerActive, P2);

  const rv = {
    lossV1: windowFrac(t1, ...WIN1.loss),
    evalV1: windowFrac(t1, ...WIN1.eval),
    projV1: windowFrac(t1, ...WIN1.proj),
    lossV2: windowFrac(t2, ...WIN2.loss),
    evalV2: windowFrac(t2, ...WIN2.eval),
    projV2: windowFrac(t2, ...WIN2.proj),
  };

  // fire each phase's "revealed" callback exactly once (reset if the phase restarts)
  const trainDone = useRef(false), steerDone = useRef(false);
  useEffect(() => { if (!active) trainDone.current = false; }, [active]);
  useEffect(() => { if (!steerActive) steerDone.current = false; }, [steerActive]);
  useEffect(() => {
    if (active && t1 >= P1 && !trainDone.current) { trainDone.current = true; onTrainRevealed && onTrainRevealed(); }
  }, [active, t1, onTrainRevealed]);
  useEffect(() => {
    if (steerActive && t2 >= P2 && !steerDone.current) { steerDone.current = true; onSteerRevealed && onSteerRevealed(); }
  }, [steerActive, t2, onSteerRevealed]);

  return (
    <div className="step insights-step">
      <div className="insights-plots">
        <DualPlot run={run} steerRun={steerActive ? steerRun : null} rv={rv}
          defaultView="projection" fill chartHeight={150} showEarlyStop={earlyStopShown} />
      </div>
    </div>
  );
}
