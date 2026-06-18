import { useEffect, useRef, useState } from "react";
import { streamRunResults } from "../api.js";
import { EVAL_SERIES_META, CONCEPT_PALETTE } from "../data/demo.js";
import ChartArtifact from "./artifacts/ChartArtifact.jsx";
import { AgentRow, Bubble } from "./Chat.jsx";

// New-chat demo: a finished run's recorded series streamed point-by-point (~30s) so the
// charts BUILD live, as if training now. Accumulates `point` events into growing series and
// renders them with the same SeriesChart the completed-run narrative uses. Calls onDone with
// the base→final summary so the design agent can review it and propose a mediated run.
export default function LiveTraining({ dataset, model, seconds = 30, onDone }) {
  const [evalPts, setEvalPts] = useState({});      // {metricKey: [[step,val]]}
  const [lossPts, setLossPts] = useState({ train: [], eval: [] });
  const [monPts, setMonPts] = useState({});        // {concept: [[step,val]]}
  const [meta, setMeta] = useState(null);
  const [progress, setProgress] = useState({ step: 0, n: 0 });
  const [done, setDone] = useState(false);
  const started = useRef(false);
  const nRef = useRef(0);                            // n_steps from meta (closure-stable)

  useEffect(() => {
    if (started.current) return;
    started.current = true;
    const es = streamRunResults(dataset, model, {
      onMeta: (m) => { setMeta(m); nRef.current = m?.n_steps || 0; },
      onPoint: (p) => {
        if (!p) return;
        setProgress((pr) => ({ step: pr.step + 1, n: nRef.current }));
        if (p.eval && Object.keys(p.eval).length)
          setEvalPts((cur) => { const n = { ...cur }; for (const k in p.eval) n[k] = (n[k] || []).concat([[p.step, p.eval[k]]]); return n; });
        if (p.loss && (p.loss.train != null || p.loss.eval != null))
          setLossPts((cur) => ({
            train: p.loss.train != null ? cur.train.concat([[p.step, p.loss.train]]) : cur.train,
            eval: p.loss.eval != null ? cur.eval.concat([[p.step, p.loss.eval]]) : cur.eval,
          }));
        if (p.monitor && Object.keys(p.monitor).length)
          setMonPts((cur) => { const n = { ...cur }; for (const c in p.monitor) n[c] = (n[c] || []).concat([[p.step, p.monitor[c]]]); return n; });
      },
      onDone: (d) => { setDone(true); onDone && onDone(d?.summary || null); },
    }, seconds);
    return () => es.close();
  }, [dataset, model]);

  const early = meta?.early_stop ?? null;

  // build the three chart series from what's accumulated so far
  const lossSeries = [
    { key: "train_loss", label: "train loss", color: "#c2a36b", axis: "L", points: lossPts.train },
    { key: "eval_loss", label: "eval loss", color: "#b06a4f", axis: "L", dashed: true, points: lossPts.eval },
  ].filter((s) => s.points.length);

  const evalSeries = EVAL_SERIES_META.map((s) => ({
    ...s, points: s.axis === "R" ? (s.key === "train_loss" ? lossPts.train : lossPts.eval) : (evalPts[s.key] || []),
  })).filter((s) => s.points.length);

  const monConcepts = meta?.concepts || Object.keys(monPts);
  const monSeries = monConcepts.map((c, i) => ({
    key: c, label: c.replace(/_/g, " "), color: CONCEPT_PALETTE[i % CONCEPT_PALETTE.length], axis: "L",
    points: monPts[c] || [],
  })).filter((s) => s.points.length);

  const pct = progress.n ? Math.min(100, Math.round((progress.step / progress.n) * 100)) : (done ? 100 : 0);

  return (
    <AgentRow>
      <Bubble>
        {done
          ? <>Training run complete. Here's how the model drifted over fine-tuning — the eval battery and the per-concept projection ⟨h, v̂⟩:</>
          : <>Launching the run — streaming checkpoints live. Watch the evals and concept directions move as training proceeds:</>}
      </Bubble>

      {/* progress bar */}
      <div style={{ display: "flex", alignItems: "center", gap: "11px" }}>
        <span style={{ flex: "none", width: "9px", height: "9px", borderRadius: "50%", background: done ? "#2f9e7d" : "#1f9e86", animation: done ? "none" : "lc-pulse 1.2s ease-in-out infinite" }} />
        <div style={{ flex: 1, height: "6px", borderRadius: "4px", background: "#eef2f9", overflow: "hidden" }}>
          <div style={{ height: "100%", width: pct + "%", background: done ? "#2f9e7d" : "#2f43e0", borderRadius: "4px", transition: "width .3s ease" }} />
        </div>
        <span style={{ flex: "none", fontSize: "11.5px", color: "#98a2b3", fontFamily: "'JetBrains Mono',monospace" }}>
          {done ? "done" : `step ${progress.step}${progress.n ? "/" + progress.n : ""}`}
        </span>
      </div>

      <ChartArtifact title="Training loss" subtitle="loss vs step" glyphTone="#c2a36b"
        series={lossSeries} chartProps={{ leftDomain: "auto", yLeftLabel: "loss", earlyStop: early }}
        emptyNote="Waiting for the first checkpoint…" />
      <ChartArtifact title="Training & evaluation" subtitle="accuracy · refusal + loss vs step"
        series={evalSeries} chartProps={{ leftDomain: "unit", unit: true, yRightLabel: "loss", earlyStop: early }}
        emptyNote="Waiting for the first eval pass…" />
      <ChartArtifact title="Concept vectors" subtitle="projection ⟨h, v̂⟩ vs step" glyphTone="#7a6a8a"
        series={monSeries} chartProps={{ leftDomain: "auto", baselineZero: true, earlyStop: early }}
        emptyNote="Waiting for the concept monitor…" />
    </AgentRow>
  );
}
