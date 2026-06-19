import { useEffect, useMemo, useRef, useState } from "react";
import { streamRunResults } from "../api.js";
import { EVAL_SERIES_META, CONCEPT_PALETTE } from "../data/demo.js";
import ChartArtifact from "./artifacts/ChartArtifact.jsx";
import { AgentRow, Bubble, WandbLink } from "./Chat.jsx";
import { PLOT } from "../styles/tokens.js";

// One launched run, streamed from a recorded run point-by-point (~30s) so the charts build
// live. The new-chat flow launches this TWICE as separate blocks:
//   mode="training"        — loss + eval battery build live (the LoRA fine-tune).
//   mode="interpretability"— the concept-vector drift monitor builds live (the safety read).
// (mode="full" keeps the original single-launch two-act loss→safety reveal.)
// onDone(summary) fires when the block finishes streaming; the agent stays idle until the
// user speaks again, so nothing is auto-appended below.
const SAFETY = "#b5432f";   // hedda's safety/alarm accent

export default function LiveTraining({ dataset, model, seconds = 10, mode = "training", onDone }) {
  const [evalPts, setEvalPts] = useState({});      // {metricKey: [[step,val]]}
  const [lossPts, setLossPts] = useState({ train: [], eval: [] });
  const [monPts, setMonPts] = useState({});        // {concept: [[step,val]]}
  const [meta, setMeta] = useState(null);
  const [progress, setProgress] = useState({ step: 0, n: 0 });
  const [done, setDone] = useState(false);
  const [phase, setPhase] = useState("loss");      // (full mode only) 'loss' | 'safety'
  const [revealIdx, setRevealIdx] = useState(0);   // (full mode only) act-2 reveal cutoff
  const started = useRef(false);
  const nRef = useRef(0);
  const summaryRef = useRef(null);
  const doneFired = useRef(false);
  const full = mode === "full";

  const fireDone = () => { if (!doneFired.current) { doneFired.current = true; setDone(true); onDone && onDone(summaryRef.current); } };

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
      onDone: (d) => {
        summaryRef.current = d?.summary || null;
        if (full) setTimeout(() => setPhase("safety"), 800);   // full mode → cue act 2
        else fireDone();                                        // staged mode → this block is done
      },
    }, seconds);
    return () => es.close();
  }, [dataset, model]);

  const early = meta?.early_stop ?? null;

  // (full mode) act-2 reveal timeline
  const allSteps = useMemo(() => {
    if (!full || phase !== "safety") return [];
    const s = new Set();
    for (const [st] of lossPts.train) s.add(st);
    for (const [st] of lossPts.eval) s.add(st);
    for (const k in evalPts) for (const [st] of evalPts[k]) s.add(st);
    for (const c in monPts) for (const [st] of monPts[c]) s.add(st);
    return [...s].sort((a, b) => a - b);
  }, [full, phase]);

  useEffect(() => {
    if (!full || phase !== "safety") return;
    if (!allSteps.length) { setRevealIdx(1); return; }
    let i = 0; setRevealIdx(0);
    const id = setInterval(() => {
      i += 1; setRevealIdx(i);
      if (i >= allSteps.length) { clearInterval(id); fireDone(); }
    }, 520);
    return () => clearInterval(id);
  }, [full, phase, allSteps.length]);

  // act-2 cutoff (full mode); staged modes show everything streamed so far
  const cut = !full ? Infinity
    : phase !== "safety" ? -Infinity
    : revealIdx >= allSteps.length ? Infinity
    : allSteps[revealIdx - 1] ?? -Infinity;
  const slice = (pts) => (cut === Infinity ? pts : pts.filter(([st]) => st <= cut));

  // ── series ──────────────────────────────────────────────────────────────────────────
  const lossSeries = [
    { key: "train_loss", label: "train loss", color: PLOT.wheat, axis: "L", points: lossPts.train },
    { key: "eval_loss", label: "eval loss", color: PLOT.seal, axis: "L", dashed: true, points: lossPts.eval },
  ].filter((s) => s.points.length);

  const evalSeries = EVAL_SERIES_META.map((s) => ({
    ...s, points: slice(s.axis === "R" ? (s.key === "train_loss" ? lossPts.train : lossPts.eval) : (evalPts[s.key] || [])),
  })).filter((s) => s.points.length);

  const monConcepts = meta?.concepts || Object.keys(monPts);
  const monSeries = monConcepts.map((c, i) => ({
    key: c, label: c.replace(/_/g, " "), color: CONCEPT_PALETTE[i % CONCEPT_PALETTE.length], axis: "L",
    points: slice(monPts[c] || []),
  })).filter((s) => s.points.length);

  const pct = progress.n ? Math.min(100, Math.round((progress.step / progress.n) * 100)) : (done ? 100 : 0);

  const ProgressBar = ({ tone = "var(--seal)" }) => (
    <div style={{ display: "flex", alignItems: "center", gap: "11px" }}>
      <span style={{ flex: "none", width: "9px", height: "9px", borderRadius: "50%", background: "var(--good)", animation: done ? "none" : "lucent-breathe 1.3s ease-in-out infinite" }} />
      <div style={{ flex: 1, height: "6px", borderRadius: "4px", background: "var(--line)", overflow: "hidden" }}>
        <div style={{ height: "100%", width: pct + "%", background: done ? "var(--good)" : tone, borderRadius: "4px", transition: "width .3s ease" }} />
      </div>
      <span style={{ flex: "none", fontSize: "11.5px", color: "var(--mute-2)", fontFamily: "'JetBrains Mono',monospace" }}>
        {done ? "done" : `step ${progress.step}${progress.n ? "/" + progress.n : ""}`}
      </span>
    </div>
  );

  // ── STEP 1 (ACT 1) — training: the loss curve, live and ALONE ────────────────────────
  if (mode === "training") {
    return (
      <AgentRow>
        <Bubble>{done
          ? <>Training finished. The loss fell cleanly the whole way — on this curve alone, you'd ship it.</>
          : <>Launching the training run — streaming checkpoints live. Here's the training loss as the model learns:</>}</Bubble>
        <ProgressBar />
        <ChartArtifact title="Training loss" subtitle="loss vs step" glyphTone={PLOT.wheat}
          series={lossSeries} chartProps={{ leftDomain: "auto", yLeftLabel: "loss", earlyStop: early }}
          emptyNote="Waiting for the first checkpoint…" />
        <WandbLink href={meta?.wandb_url} />
      </AgentRow>
    );
  }

  // ── STEP 2 (ACT 2) — interpretability: the safety reveal (concept drift + eval battery) ──
  if (mode === "interpretability") {
    return (
      <AgentRow>
        <Bubble>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
            <span style={{ fontSize: "10px", fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", color: SAFETY, background: SAFETY + "16", padding: "3px 8px", borderRadius: "6px" }}>Interpretability</span>
          </div>
          {done
            ? <>This is what the loss curve couldn't see: the concept projections ⟨h, v̂⟩ drifted while the eval battery fell. The damage is real.</>
            : <>The loss looked clean — now let's look at the safety dimensions. Tracking the concept projections ⟨h, v̂⟩ and the eval battery live, per checkpoint:</>}
        </Bubble>
        <ProgressBar tone={SAFETY} />
        <ChartArtifact title="Concept vectors · drift" subtitle="projection ⟨h, v̂⟩ vs step" glyphTone={SAFETY}
          series={monSeries} chartProps={{ leftDomain: "auto", baselineZero: true, earlyStop: early }}
          emptyNote="Reading the concept monitor…" />
        <ChartArtifact title="Training & evaluation" subtitle="accuracy · refusal + loss vs step"
          series={evalSeries} chartProps={{ leftDomain: "unit", unit: true, yRightLabel: "loss", earlyStop: early }}
          emptyNote="Replaying the eval passes…" />
      </AgentRow>
    );
  }

  // ── FULL: original single-launch two-act (loss, then safety reveal) ──────────────────
  const lossDone = phase === "safety";
  const scanning = phase === "safety" && revealIdx < allSteps.length;
  return (
    <>
      <AgentRow>
        <Bubble>{lossDone
          ? <>Training finished. The loss fell cleanly the whole way — on this curve alone, you'd ship it.</>
          : <>Launching the run — streaming checkpoints live. Here's the training loss as the model learns:</>}</Bubble>
        <ProgressBar />
        <ChartArtifact title="Training loss" subtitle="loss vs step" glyphTone={PLOT.wheat}
          series={lossSeries} chartProps={{ leftDomain: "auto", yLeftLabel: "loss", earlyStop: early }}
          emptyNote="Waiting for the first checkpoint…" />
      </AgentRow>
      {phase === "safety" && (
        <AgentRow>
          <Bubble>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
              <span style={{ fontSize: "10px", fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", color: SAFETY, background: SAFETY + "16", padding: "3px 8px", borderRadius: "6px" }}>Safety</span>
              {scanning && <span style={{ fontSize: "11.5px", color: SAFETY }}>scanning concept directions…</span>}
            </div>
            Wait — before we ship this to HuggingFace, let's look at the safety dimensions. Tracking the
            concept projections ⟨h, v̂⟩ and the eval battery live — hover any point to read it off.
          </Bubble>
          <ChartArtifact title="Concept vectors · drift" subtitle="projection ⟨h, v̂⟩ vs step" glyphTone={SAFETY}
            series={monSeries} chartProps={{ leftDomain: "auto", baselineZero: true, earlyStop: early }}
            emptyNote="Reading the concept monitor…" />
          <ChartArtifact title="Training & evaluation" subtitle="accuracy · refusal + loss vs step"
            series={evalSeries} chartProps={{ leftDomain: "unit", unit: true, yRightLabel: "loss", earlyStop: early }}
            emptyNote="Replaying the eval passes…" />
        </AgentRow>
      )}
    </>
  );
}
