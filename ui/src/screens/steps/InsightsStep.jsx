import { motion, AnimatePresence } from "motion/react";
import DualPlot from "../../components/DualPlot.jsx";

const EASE = [0.4, 0, 0.2, 1];

/* Insights plots.
   - Before mitigation: the two biased plots sit CENTERED at half-width, full height.
   - On "Apply preventive steering": the biased plots morph/move to the LEFT half (Motion
     `layout`), the steered plots animate IN on the right, and only then does their
     live-fill (mitReveal) begin — sequenced in RunView. */
export default function InsightsStep({ run, reveal, mitigated, steerRun, mitReveal, earlyStopShown = false }) {
  return (
    <div className="step insights-step">
      <div className="insights-plots">
        <motion.div layout className="plot-slot" transition={{ duration: 0.55, ease: EASE }}>
          <DualPlot run={run} reveal={reveal} title="Base adapter"
            subtitle="no mitigation" defaultView="projection" fill chartHeight={150} showEarlyStop={earlyStopShown} />
        </motion.div>

        <AnimatePresence>
          {mitigated && steerRun && (
            <motion.div layout key="steer" className="plot-slot"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.5, ease: EASE, opacity: { delay: 0.18, duration: 0.4 } }}>
              <DualPlot run={steerRun} reveal={mitReveal} title="Safety adapter"
                subtitle={`Suppressed ${run.steer.concept} during training · layer ${run.steer.layer} · strength ${run.steer.coef}`}
                defaultView="projection" fill chartHeight={150} showEarlyStop={mitReveal >= 1} />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
