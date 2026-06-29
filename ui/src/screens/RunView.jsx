import { useEffect, useRef, useState } from "react";
import PipelineNav from "../components/PipelineNav.jsx";
import InsightStream from "../components/stream/InsightStream.jsx";
import SetupStep from "./steps/SetupStep.jsx";
import AuditStep from "./steps/AuditStep.jsx";
import InsightsStep from "./steps/InsightsStep.jsx";
import CheckoutStep from "./steps/CheckoutStep.jsx";
import { getRun, makeSteerRun } from "../api/sampleData.js";
import { useReveal } from "../lib/hooks.js";
import { titleCase, signed } from "../lib/format.js";

const STEPS = [
  { id: "setup", label: "Setup" },
  { id: "audit", label: "Audit" },
  { id: "insights", label: "Insights" },
  { id: "checkout", label: "Checkout" },
];

export default function RunView({ runId, onBack }) {
  const run = getRun(runId);
  const steerRun = makeSteerRun(run);

  const [step, setStep] = useState("setup");
  const [unlocked, setUnlocked] = useState(new Set(["setup"]));
  const [items, setItems] = useState([]);
  const [model, setModel] = useState(run.model.id);
  const [lora, setLora] = useState("balanced");
  const [tracked, setTracked] = useState(null);
  const [auditRun, setAuditRun] = useState(false);
  const [insightsActive, setInsightsActive] = useState(false);
  const [mitActive, setMitActive] = useState(false);
  const [mitigated, setMitigated] = useState(false);

  const reveal = useReveal(insightsActive, 5000);
  const mitReveal = useReveal(mitActive, 4000);

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

  // ── intro (on mount) ──
  useEffect(() => {
    if (!fireOnce("intro")) return;
    push({ type: "insight", lead: `Loaded ${run.dataset.domain}/sft.jsonl to fine-tune ${run.model.label}.`,
      bullets: ["Pick the base model and LoRA recipe.", "Then run the pre-training dataset audit before we burn a training run."] });
    push({ type: "action", title: "Run the pre-training audit on this dataset.", gate: "dataset + model selected",
      label: "Run dataset audit", onAct: () => goTo("audit") });
  }, []);

  // ── per-step narration (triggered from goTo — a user action, not an effect) ──
  function narrate(s) {
    if (s === "audit" && fireOnce("audit")) {
      after(250, () => push({ type: "insight", think: true, text: "Reading the dataset and the use-case to propose the safety-critical axes…" }));
      after(950, () => push({ type: "insight",
        lead: `Proposed ${run.concepts.length} web-grounded safety ${run.concepts.length === 1 ? "axis" : "axes"} for ${run.dataset.domain}:`,
        bullets: run.concepts.map((c) => `${titleCase(c.name)} — ${c.description}`) }));
      after(1500, () => push({ type: "question", question: "Which safety axes should we track?", multiSelect: true,
        confirmLabel: "tracking these axes",
        options: run.concepts.map((c) => ({ label: c.name, description: c.description, default: true })),
        onSubmit: (vals) => { setTracked(vals); runAudit(vals); } }));
    }
    if (s === "insights" && fireOnce("insights")) {
      push({ type: "insight", think: true, text: "Fine-tuning… projecting activations onto every concept vector each checkpoint." });
      setInsightsActive(true);
    }
    if (s === "checkout" && fireOnce("checkout")) {
      after(250, () => push({ type: "insight",
        lead: mitigated ? "Receipt ready — the steered adapter recovered safety at no capability cost." : "Receipt ready.",
        bullets: mitigated
          ? ["Export the adapter, push to the Hub, or generate the 1-page PDF.", "The mitigation spec is part of the run contract."]
          : ["Export the adapter or generate the 1-page PDF report."] }));
    }
  }

  function runAudit(vals) {
    const n = run.audit.totalFlagged;
    after(700, () => {
      setAuditRun(true);
      push({ type: "insight", lead: `Projected the ${vals.length} concept vectors onto every training sample.`,
        bullets: [`${n} samples sit above the p${run.audit.percentile} projection threshold — flagged in red.`,
          "These are the rows most likely to drive drift. Inspect or clean, then train."] });
      push({ type: "metric", value: n, label: `samples flagged · p${run.audit.percentile}`, tone: "bad" });
      after(700, () => push({ type: "action", title: "Start the LoRA fine-tune with per-checkpoint drift monitoring.",
        gate: "audit complete", label: "Start fine-tuning", onAct: () => goTo("insights") }));
    });
  }

  // ── post-run narration once the live-fill completes ──
  useEffect(() => {
    if (!(insightsActive && reveal >= 1)) return;
    if (!fireOnce("insights-done")) return;
    const hb = run.series.eval.harmbench_refusal_v2;
    const mm = run.series.eval.mmlu_pro_acc;
    const worst = [...run.concepts].filter((c) => run.series.trajectory[c.name])
      .sort((a, b) => last(run.series.trajectory[b.name]).probe_prob - last(run.series.trajectory[a.name]).probe_prob)[0];
    const wp = worst ? last(run.series.trajectory[worst.name]).probe_prob : null;
    const bullets = [
      `Eval loss bottomed at step ${run.earlyStop} (the dashed line); past it the model overfits the biased data.`,
    ];
    if (hb) bullets.push(`HarmBench refusal ${hb[0][1].toFixed(2)} → ${last2(hb).toFixed(2)} — a real safety regression.`);
    if (worst) bullets.push(`${titleCase(worst.name)} probe reached ${wp.toFixed(2)} — the axis the data drove.`);
    if (mm) bullets.push(`MMLU-Pro ${mm[0][1].toFixed(2)} → ${last2(mm).toFixed(2)}; capability slipped too.`);
    bullets.push("None of this shows up in the loss curve. That is the silent drift.");
    push({ type: "insight", lead: "Fine-tune finished — and the loss curve was hiding this:", bullets });

    after(500, () => push({ type: "question", question: "How do you want to proceed?", multiSelect: false,
      confirmLabel: "proceeding",
      options: [
        { label: "Preventive-steering fix", description: "re-train suppressing the malign axis (recommended)", default: true },
        { label: "Early-stop at last clean checkpoint", description: `roll back to step ${run.earlyStop}` },
        { label: "Ship as-is", description: "accept the drift" },
      ],
      onSubmit: (vals) => onProceed(vals[0]) }));
  }, [reveal, insightsActive]);

  function onProceed(choice) {
    if (choice !== "Preventive-steering fix") {
      after(300, () => push({ type: "insight", lead: `Noted — ${choice.toLowerCase()}.`,
        bullets: ["No steering applied. You can still export the current adapter."] }));
      after(700, () => push({ type: "action", title: "Review the receipt.", gate: "decision made",
        label: "Go to checkout", variant: "primary", onAct: () => goTo("checkout") }));
      return;
    }
    if (!run.steer) {
      after(300, () => push({ type: "insight", lead: "Preventive steering is recorded for the medical & gender demo runs.",
        bullets: ["Open one of those to see the mitigation arm fill live."] }));
      after(700, () => push({ type: "action", title: "Review the receipt.", gate: "decision made",
        label: "Go to checkout", onAct: () => goTo("checkout") }));
      return;
    }
    after(400, () => push({ type: "question", question: "Which axes should we suppress during training?", multiSelect: true,
      confirmLabel: "suppressing these",
      options: run.concepts.filter((c) => run.series.trajectory[c.name]).map((c) => ({
        label: c.name, description: c.description,
        default: c.name === run.steer.concept || last(run.series.trajectory[c.name]).probe_prob > 0.85,
      })),
      onSubmit: () => startMitigation() }));
  }

  function startMitigation() {
    after(300, () => push({ type: "insight",
      lead: `Re-training with +${run.steer.coef}·v̂ on ${run.steer.concept} at layer ${run.steer.layer} — during training only.`,
      bullets: ["The steering hook is removed before the adapter is saved, so the shipped model carries no extra attack surface."] }));
    after(600, () => { setMitigated(true); setMitActive(true); });
  }

  // ── mitigation narration once its fill completes ──
  useEffect(() => {
    if (!(mitActive && mitReveal >= 1)) return;
    if (!fireOnce("mit-done")) return;
    const hb = run.steer.eval.harmbench_refusal_v2;
    const pp = hb ? Math.round((hb.steered - hb.unsteered) * 100) : null;
    if (pp != null) push({ type: "metric", value: `+${pp}`, label: "HarmBench refusal recovered (pp)", tone: "good" });
    push({ type: "insight", lead: "Mitigation worked:",
      bullets: [
        `${titleCase(run.steer.concept)} suppressed; safety refusal recovered substantially.`,
        "Capability held — MMLU and TruthfulQA essentially flat.",
        run.steer.note,
      ] });
    after(500, () => push({ type: "action", title: "Compare the arms and export the model.", gate: "mitigation finished",
      label: "Go to checkout", variant: "good", onAct: () => goTo("checkout") }));
  }, [mitReveal, mitActive]);

  return (
    <div className="runview">
      <div className="stage">
        <div className="stage-head">
          <div className="stack">
            <span className="eyebrow">{run.project}</span>
            <h2 className="title" style={{ fontSize: 22 }}>{run.title}</h2>
          </div>
          <PipelineNav steps={STEPS} current={step} unlocked={unlocked} onJump={goTo} />
        </div>

        {step === "setup" && <SetupStep run={run} model={model} setModel={setModel} lora={lora} setLora={setLora} />}
        {step === "audit" && <AuditStep run={run} auditRun={auditRun} tracked={tracked} />}
        {step === "insights" && <InsightsStep run={run} reveal={reveal} mitigated={mitigated} steerRun={steerRun} mitReveal={mitReveal} />}
        {step === "checkout" && <CheckoutStep run={run} mitigated={mitigated} />}
      </div>

      <InsightStream items={items} live={insightsActive && reveal < 1} />
    </div>
  );
}

const last = (arr) => arr[arr.length - 1];
const last2 = (series) => series[series.length - 1][1];
