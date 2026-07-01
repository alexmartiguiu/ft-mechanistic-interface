import { useEffect, useRef, useState } from "react";
import PipelineNav from "../components/PipelineNav.jsx";
import InsightStream from "../components/stream/InsightStream.jsx";
import SetupStep from "./steps/SetupStep.jsx";
import AuditStep from "./steps/AuditStep.jsx";
import InsightsStep from "./steps/InsightsStep.jsx";
import CheckoutStep from "./steps/CheckoutStep.jsx";
import { getRun, makeSteerRun } from "../api/sampleData.js";
import { titleCase } from "../lib/format.js";
import { WRAPUP_QUESTION, WRAPUP_OPTIONS, emailReport } from "../lib/wrapup.js";

const STEPS = [
  { id: "setup", label: "Setup" },
  { id: "audit", label: "Audit" },
  { id: "insights", label: "Realign" },
  { id: "checkout", label: "Checkout" },
];

export default function RunView({ runId, onBack, onStep }) {
  const [boundRunId, setBoundRunId] = useState(runId || null);
  const boundRef = useRef(runId || null);
  const run = boundRunId ? getRun(boundRunId) : null;
  const steerRun = run ? makeSteerRun(run) : null;
  const R = () => (boundRef.current ? getRun(boundRef.current) : null);

  const [step, setStep] = useState("setup");
  useEffect(() => { onStep?.(step); }, [step]);   // reflect the pipeline step in the URL
  const [unlocked, setUnlocked] = useState(new Set(["setup"]));
  const [completed, setCompleted] = useState(new Set());   // steps the run has closed out (lights Checkout terracotta)
  const [items, setItems] = useState([]);
  const [model, setModel] = useState(run ? run.model.id : "apertus-8b");
  const [lora, setLora] = useState("balanced");
  const [tracked, setTracked] = useState(null);
  const [auditRun, setAuditRun] = useState(false);
  const [insightsActive, setInsightsActive] = useState(false);
  const [mitActive, setMitActive] = useState(false);
  const [mitigated, setMitigated] = useState(false);
  // the early-stop plot marker is event-driven: off until the early-stop insight fires
  const [earlyStopShown, setEarlyStopShown] = useState(false);
  // px once the user drags the divider; null → responsive CSS default (25%). Clamped in CSS too.
  const [railW, setRailW] = useState(null);

  function startRailResize(e) {
    e.preventDefault();
    const onMove = (ev) => setRailW(Math.max(300, Math.min(window.innerWidth * 0.4, window.innerWidth - ev.clientX)));
    const onUp = () => {
      document.body.style.cursor = "";
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    document.body.style.cursor = "col-resize";
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }
  function nudgeRail(e) {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const base = railW ?? Math.round(window.innerWidth * 0.25);
    setRailW(Math.max(300, Math.min(window.innerWidth * 0.4, base + (e.key === "ArrowLeft" ? 16 : -16))));
  }

  // phase-complete flags set by InsightsStep when each staged fill finishes (they gate the
  // rail narration + the "live" pill). The staged reveal itself now lives in InsightsStep.
  const [trainRevealed, setTrainRevealed] = useState(false);
  const [steerRevealed, setSteerRevealed] = useState(false);

  const idRef = useRef(0);
  const firedRef = useRef(new Set());
  const timersRef = useRef([]);
  useEffect(() => () => timersRef.current.forEach(clearTimeout), []);

  const push = (it) => setItems((p) => [...p, { ...it, id: ++idRef.current }]);
  const after = (ms, fn) => timersRef.current.push(setTimeout(fn, ms));
  const fireOnce = (key) => {
    if (firedRef.current.has(key)) return false;
    firedRef.current.add(key); return true;
  };

  const goTo = (s) => {
    setStep(s);
    setUnlocked((u) => new Set(u).add(s));
    narrate(s);
  };

  // ── intro ──
  useEffect(() => {
    if (R()) introExisting();
    else pushWelcome();
  }, []);

  function pushWelcome() {
    if (!fireOnce("welcome")) return;
    push({ type: "insight", lead: "New experiment.",
      bullets: ["Drop in a dataset, browse Hugging Face, or select one with recorded results.",
        "Once it loads, I will propose the malign concepts to track for this domain."] });
  }
  function introExisting() {
    if (!fireOnce("intro")) return;
    const r = R();
    push({ type: "insight", think: true, text: `I am reading ${r.dataset.domain}/sft.jsonl against your stated use-case to characterise the risk surface.` });
    after(750, () => push({ type: "insight", lead: `Loaded ${r.dataset.domain}/sft.jsonl for fine-tuning ${r.model.label}.`,
      bullets: ["Select the base model and the LoRA recipe.", "Then run the pre-training dataset audit before the run is launched."] }));
    after(1150, () => push({ type: "action", title: "Run the pre-training audit on this dataset.",
      label: "Run dataset audit", onAct: () => goTo("audit") }));
  }

  // ── dataset selection (New-experiment flow) ──
  function selectDataset(id) {
    if (id == null) { resetNew(); return; }
    if (boundRef.current === id) return;
    boundRef.current = id;
    setBoundRunId(id);
    setModel(getRun(id).model.id);
    if (!fireOnce("bound:" + id)) return;
    const r = getRun(id);
    after(750, () => push({ type: "insight",
      lead: `Loaded ${r.dataset.domain}/sft.jsonl, comprising ${r.audit.total.toLocaleString()} examples.`,
      bullets: ["Select the base model and the LoRA recipe.", "Then run the pre-training audit before the run is launched."] }));
    after(1150, () => push({ type: "action", title: "Run the pre-training audit on this dataset.",
      label: "Run dataset audit", onAct: () => goTo("audit") }));
  }
  function resetNew() {
    boundRef.current = null;
    setBoundRunId(null);
    setStep("setup");
    setUnlocked(new Set(["setup"]));
    setAuditRun(false); setInsightsActive(false); setMitActive(false); setMitigated(false); setTracked(null);
    setEarlyStopShown(false); setTrainRevealed(false); setSteerRevealed(false);
    firedRef.current = new Set();
    setItems([]);
    pushWelcome();
  }

  // ── per-step narration ──
  function narrate(s) {
    const r = R();
    if (!r) return;
    if (s === "audit" && fireOnce("audit")) {
      after(200, () => push({ type: "steps",
        lead: `Proposing malign concepts for ${r.dataset.domain}.`,
        steps: [
          `Reading ${r.dataset.domain}/sft.jsonl against your stated use-case`,
          "Drafting contrastive prompt pairs for each trait",
          "Estimating persona directions v̂ = (μ₊ − μ₋)/‖μ₊ − μ₋‖ by difference of means",
          "Scoring every sample by its projection s = ⟨h, v̂⟩",
        ] }));
      after(2500, () => push({ type: "insight", think: true,
        text: "Each malign concept is represented as a persona direction in activation space (Chen et al. 2025). Because narrow fine-tuning can shift a model's behaviour broadly rather than only on-task (Betley et al. 2025), I track the traits most at risk for this domain." }));
      after(2900, () => push({ type: "insight", kind: "educate",
        lead: `Proposed ${r.concepts.length} malign ${r.concepts.length === 1 ? "concept" : "concepts"} for ${r.dataset.domain}:`,
        bullets: r.concepts.map((c) => `${titleCase(c.name)}: ${c.description}`) }));
      after(3300, () => push({ type: "question", question: "Which malign concepts should we track?", multiSelect: true,
        confirmLabel: "tracking these malign concepts",
        options: r.concepts.map((c) => ({ label: c.name, description: c.description, default: true })),
        onSubmit: (vals) => { setTracked(vals); runAudit(vals); } }));
    }
    if (s === "insights" && fireOnce("insights")) {
      push({ type: "insight", think: true, text: "Fine-tuning is under way. At each checkpoint I project the model's activations onto every malign-concept direction." });
      setInsightsActive(true);
    }
    if (s === "checkout" && fireOnce("checkout")) {
      after(250, () => push({ type: "insight",
        lead: mitigated ? "The receipt is ready: the Safety adapter recovered safety at no measurable capability cost." : "The receipt is ready.",
        bullets: mitigated
          ? ["Export the adapter, push it to the Hub, or generate the one-page report.", "The mitigation specification is recorded as part of the run contract."]
          : ["Export the adapter, or generate the one-page report."] }));
      // the run is closed out → offer where to go next (this is what lights the Checkout node)
      after(1100, () => push({ type: "question", question: WRAPUP_QUESTION, multiSelect: false,
        confirmLabel: "closing out the run", options: WRAPUP_OPTIONS, onSubmit: (vals) => onWrapUp(vals[0]) }));
    }
  }

  // the final wrap-up choice. Confirming it marks Checkout complete (its node turns
  // terracotta) and then does the chosen thing.
  function onWrapUp(choice) {
    setCompleted((c) => new Set(c).add("checkout"));
    if (choice === "Email the 1-page report") {
      emailReport(R(), steerRun);
      after(200, () => push({ type: "insight", lead: "I have drafted a summary email in your mail client.",
        bullets: ["Attach the one-page report (use Download 1-pager on the receipt) before sending."] }));
      return;
    }
    after(450, () => onBack && onBack());   // Back to projects / Close this run → the gallery
  }

  function runAudit(vals) {
    const r = R();
    const n = r.audit.totalFlagged;
    after(700, () => {
      setAuditRun(true);
      push({ type: "insight", lead: `Projected all ${vals.length} malign-concept ${vals.length === 1 ? "direction" : "directions"} onto every training sample.`,
        bullets: [`${n} samples lie above the p${r.audit.percentile} projection threshold and are flagged in red.`,
          "These rows are the most likely drivers of drift; inspect or clean them before training."] });
      push({ type: "metric", value: n, label: `samples flagged · p${r.audit.percentile}`, tone: "bad" });
      after(700, () => push({ type: "action", title: "Start the monitored LoRA fine-tune.",
        label: "Start fine-tuning", doneLabel: "Fine-tuning started", onAct: () => goTo("insights") }));
    });
  }

  // ── post-run narration once the fine-tune's staged fill completes (InsightsStep callback) ──
  function onTrainRevealed() {
    setTrainRevealed(true);
    if (!fireOnce("insights-done")) return;
    const r = R();
    const hb = r.series.eval.harmbench_refusal_v2;
    const mm = r.series.eval.mmlu_pro_acc;
    const tl = r.series.loss?.train;
    const drifted = [...r.concepts].filter((c) => r.series.trajectory[c.name])
      .sort((a, b) => last(r.series.trajectory[b.name]).probe_prob - last(r.series.trajectory[a.name]).probe_prob);
    const worst = drifted[0], second = drifted[1];
    const probeOf = (c) => last(r.series.trajectory[c.name]).probe_prob;

    // Box 1 — the benchmark battery (top plot): training loss, capability, refusal.
    const benchBullets = [];
    if (tl) benchBullets.push(`Training loss fell steadily to ${last2(tl).toFixed(2)}; on its own the curve looked clean.`);
    if (hb) benchBullets.push(`HarmBench refusal fell from ${hb[0][1].toFixed(2)} to ${last2(hb).toFixed(2)}, a genuine safety regression.`);
    if (mm) benchBullets.push(`MMLU-Pro declined from ${mm[0][1].toFixed(2)} to ${last2(mm).toFixed(2)}, so capability also slipped.`);
    push({ type: "insight", kind: "educate",
      lead: "Fine-tuning is complete. The loss curve and benchmark battery record the following:", bullets: benchBullets });

    // Box 2 — the emergent-risk projections (bottom drift plot): its own readout.
    after(1000, () => {
      const riskBullets = [];
      if (worst) riskBullets.push(`The ${titleCase(worst.name)} projection climbed most, its probe reaching ${probeOf(worst).toFixed(2)}.`);
      if (second) riskBullets.push(`${titleCase(second.name)} also drifted toward risk, to ${probeOf(second).toFixed(2)}.`);
      riskBullets.push("None of this is visible in the loss curve; that is the silent drift this run exists to catch.");
      push({ type: "insight", kind: "educate",
        lead: "The emergent-risk projections, however, expose what the benchmarks alone would miss:", bullets: riskBullets });
    });

    // early-stop is its own insight; revealing it lights up the dashed plot marker (no longer fixed)
    after(2000, () => {
      push({ type: "insight", kind: "educate",
        lead: `Evaluation loss reached its minimum at step ${r.earlyStop} and then rose, so beyond that point the model is overfitting the biased data.`,
        bullets: ["It is marked on the loss plot as the dashed line; everything to its right is the drift zone."] });
      setEarlyStopShown(true);
    });

    after(2600, () => push({ type: "question", question: "How do you want to proceed?", multiSelect: false,
      confirmLabel: "proceeding",
      options: [
        { label: "Preventive-steering fix", description: "re-train suppressing the malign concept (recommended)", default: true },
        { label: "Early-stop at last clean checkpoint", description: `roll back to step ${r.earlyStop}` },
        { label: "Ship as-is", description: "accept the drift" },
        { label: "Other", description: "describe a different approach", freeform: true },
      ],
      onSubmit: (vals) => onProceed(vals[0]) }));
  }

  function onProceed(choice) {
    const r = R();
    if (choice !== "Preventive-steering fix") {
      after(300, () => push({ type: "insight", lead: `Noted. ${choice}.`,
        bullets: ["No steering applied. You can still export the current adapter."] }));
      after(700, () => push({ type: "action", title: "Review the receipt.", label: "Go to checkout", onAct: () => goTo("checkout") }));
      return;
    }
    if (!r.steer) {
      after(300, () => push({ type: "insight", lead: "Preventive steering is recorded for the medical and gender demo runs.",
        bullets: ["Open one of those to see the mitigation arm fill live."] }));
      after(700, () => push({ type: "action", title: "Review the receipt.", label: "Go to checkout", onAct: () => goTo("checkout") }));
      return;
    }
    after(400, () => push({ type: "question", question: "Which malign concepts should we suppress during training?", multiSelect: true,
      confirmLabel: "suppressing these",
      options: r.concepts.filter((c) => r.series.trajectory[c.name]).map((c) => ({
        label: c.name, description: c.description,
        default: c.name === r.steer.concept || last(r.series.trajectory[c.name]).probe_prob > 0.85,
      })),
      onSubmit: () => startMitigation() }));
  }

  function startMitigation() {
    const r = R();
    after(300, () => push({ type: "insight",
      lead: `Re-training with an additive steer of +${r.steer.coef}·v̂ on ${r.steer.concept} at layer ${r.steer.layer}, applied during training only.`,
      bullets: ["The steering hook is removed before the adapter is saved, so the shipped model carries no additional attack surface."] }));
    after(700, () => setMitigated(true));   // v2 (prev_steered) overlay becomes available
    after(1500, () => setMitActive(true));  // start the preventive-steered staged fill on the same plots
  }

  // ── mitigation narration once the preventive-steered fill completes (InsightsStep callback) ──
  function onSteerRevealed() {
    setSteerRevealed(true);
    if (!fireOnce("mit-done")) return;
    const r = R();
    const c = r.steer.concept;
    // the KEY read at this step: the steered concept's projection %-reduction (v1→v2),
    // computed off the same trajectories the plot's green delta band draws.
    const sr = makeSteerRun(r);
    const t1 = r.series.trajectory[c], t2 = sr?.series?.trajectory?.[c];
    const v1 = t1 ? last(t1).projection : null, v2 = t2 ? last(t2).projection : null;
    const pct = (v1 != null && v2 != null && v1 !== 0) ? ((v2 - v1) / Math.abs(v1)) * 100 : null;
    const fmtPct = (p) => (p >= 0 ? "+" : "-") + Math.abs(p).toFixed(1) + "%";
    if (pct != null) push({ type: "metric", value: fmtPct(pct),
      label: `${titleCase(c)} projection`, tone: pct < 0 ? "good" : "bad" });

    const ev = r.steer.eval || {};
    const safe = ev.harmbench_refusal_v2, cap = ev.mmlu_pro_acc;
    const bullets = [];
    if (safe && safe.steered != null && safe.unsteered != null)
      bullets.push(`Safety was ${safe.steered >= safe.unsteered ? "maintained, even improved" : "broadly maintained"}: HarmBench refusal ${safe.unsteered.toFixed(2)}→${safe.steered.toFixed(2)}.`);
    if (cap && cap.steered != null && cap.unsteered != null)
      bullets.push(`Capability was ${cap.steered >= cap.unsteered ? "maintained, even improved" : "maintained"}: MMLU-Pro ${cap.unsteered.toFixed(2)}→${cap.steered.toFixed(2)}.`);
    if (pct != null)
      bullets.push(`Most importantly, the ${titleCase(c)} projection ${pct < 0 ? "fell" : "rose"} ${Math.abs(pct).toFixed(1)}% (${v1.toFixed(1)}→${v2.toFixed(1)}), so the trait is ${pct < 0 ? "markedly less" : "more"} present. We read the projection as the key evaluation at this step.`);
    push({ type: "insight", kind: "educate", lead: "Preventive steering complete:", bullets });
    after(500, () => push({ type: "action", title: "Compare the base and Safety adapters, then export.",
      label: "Go to checkout", variant: "good", onAct: () => goTo("checkout") }));
  }

  return (
    <div className="runview" style={railW != null ? { "--rail-w": `${railW}px` } : undefined}>
      <div className="stage">
        <div className="stage-head">
          <div className="stack">
            {!run && <>
              <span className="eyebrow">New experiment</span>
              <h2 className="title" style={{ fontSize: 28.51 }}>Start a new experiment</h2>
            </>}
          </div>
          <PipelineNav steps={STEPS} current={step} unlocked={unlocked} completed={completed} onJump={goTo} />
        </div>

        <div className={`stage-body ${step === "insights" ? "fill" : ""}`}>
          {step === "setup" && <SetupStep run={run} model={model} setModel={setModel} lora={lora} setLora={setLora} onSelectDataset={selectDataset} />}
          {step === "audit" && run && <AuditStep run={run} auditRun={auditRun} tracked={tracked} revealed={tracked != null} />}
          {step === "insights" && run && <InsightsStep run={run} steerRun={steerRun}
            active={insightsActive} steerActive={mitActive}
            onTrainRevealed={onTrainRevealed} onSteerRevealed={onSteerRevealed}
            earlyStopShown={earlyStopShown} />}
          {step === "checkout" && run && <CheckoutStep run={run} mitigated={mitigated} steerRun={steerRun} />}
        </div>
      </div>

      <InsightStream items={items} live={insightsActive && !trainRevealed} step={step}
        onResizeStart={startRailResize} onResizeKey={nudgeRail} />
    </div>
  );
}

const last = (arr) => arr[arr.length - 1];
const last2 = (series) => series[series.length - 1][1];
