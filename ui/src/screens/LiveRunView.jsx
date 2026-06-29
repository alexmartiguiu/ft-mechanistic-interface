import { useEffect, useRef, useState } from "react";
import PipelineNav from "../components/PipelineNav.jsx";
import InsightStream from "../components/stream/InsightStream.jsx";
import SetupStep from "./steps/SetupStep.jsx";
import AuditStep from "./steps/AuditStep.jsx";
import InsightsStep from "./steps/InsightsStep.jsx";
import CheckoutStep from "./steps/CheckoutStep.jsx";
import RunView from "./RunView.jsx";
import * as api from "../api/client.js";
import { bundleToRun, steerRunFromBundle, eventToItem } from "../api/adapters.js";
import { useReveal } from "../lib/hooks.js";

const STEPS = [
  { id: "setup", label: "Setup" },
  { id: "audit", label: "Audit" },
  { id: "insights", label: "Insights" },
  { id: "checkout", label: "Checkout" },
];

/* Live run: the same shell as RunView, but the `run` data comes from the backend
   view bundle and the narration + timing are driven by the agent's SSE stream
   (rail events → items; stage events → step / reveal / morph). Falls back to the
   scripted RunView if the backend can't bind this run. */
export default function LiveRunView({ frontendRun, onBack }) {
  const [phase, setPhase] = useState("loading");   // loading | ready | error
  const [run, setRun] = useState(null);
  const [steerRun, setSteerRun] = useState(null);
  const [step, setStep] = useState("setup");
  const [unlocked, setUnlocked] = useState(new Set(["setup"]));
  const [items, setItems] = useState([]);
  const [auditRun, setAuditRun] = useState(false);
  const [insightsActive, setInsightsActive] = useState(false);
  const [mitActive, setMitActive] = useState(false);
  const [mitigated, setMitigated] = useState(false);
  const [model, setModel] = useState(frontendRun.model.id);
  const [lora, setLora] = useState("balanced");
  const [railW, setRailW] = useState(null);

  const reveal = useReveal(insightsActive, 5000);
  const mitReveal = useReveal(mitActive, 4000);

  const idRef = useRef(0);
  const sidRef = useRef(null);
  const startedRef = useRef(false);
  const firedRef = useRef(new Set());

  const push = (it) => setItems((p) => [...p, { ...it, id: ++idRef.current }]);
  const goTo = (s) => { setStep(s); setUnlocked((u) => new Set(u).add(s)); };
  const fireOnce = (k) => (firedRef.current.has(k) ? false : (firedRef.current.add(k), true));

  const onAnswer = (ref, vals) => sidRef.current && api.postAnswer(sidRef.current, ref, vals);
  const onAction = (ref, label) => {
    if (sidRef.current) api.postAction(sidRef.current, ref);
    if (/checkout/i.test(label || "")) goTo("checkout");
  };

  function handleEvent(ev) {
    if (ev.channel === "rail") {
      const item = eventToItem(ev, { onAnswer, onAction });
      if (item) push(item);
      return;
    }
    // stage directives drive the left panel
    if (ev.kind === "audit_flagged") { setAuditRun(true); goTo("audit"); }
    else if (ev.kind === "training_started") { goTo("insights"); }
    else if (ev.kind === "training_fill") { goTo("insights"); if (fireOnce("fill")) setInsightsActive(true); }
    else if (ev.kind === "mitigation_morph") {
      if (fireOnce("morph")) { setMitigated(true); setTimeout(() => setMitActive(true), 800); }
    }
  }

  // ── setup: resolve → bundle → session → stream ──
  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;
    let unsub = null, cancelled = false;
    (async () => {
      try {
        const { run_id } = await api.resolveRun(frontendRun.domain, frontendRun.model.id);
        const bundle = await api.getRunView(run_id);
        if (cancelled) return;
        setRun(bundleToRun(bundle));
        setSteerRun(steerRunFromBundle(bundle));
        setPhase("ready");
        const { sid } = await api.createSession(run_id, "replay");
        if (cancelled) return;
        sidRef.current = sid;
        unsub = api.streamSession(sid, handleEvent);
      } catch (e) {
        if (!cancelled) { console.warn("live mode unavailable, falling back:", e); setPhase("error"); }
      }
    })();
    return () => {
      cancelled = true;
      if (unsub) unsub();
      if (sidRef.current) api.closeSession(sidRef.current);
    };
  }, []);

  // rail drag-to-resize (parity with RunView)
  function startRailResize(e) {
    e.preventDefault();
    const onMove = (ev) => setRailW(Math.max(300, Math.min(window.innerWidth * 0.4, window.innerWidth - ev.clientX)));
    const onUp = () => { document.body.style.cursor = ""; window.removeEventListener("pointermove", onMove); window.removeEventListener("pointerup", onUp); };
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

  if (phase === "error") return <RunView runId={frontendRun.id} onBack={onBack} />;
  if (phase === "loading" || !run) {
    return <div className="runview"><div className="stage"><div className="center-empty">Binding to the recorded run…</div></div><div className="rail" /></div>;
  }

  return (
    <div className="runview" style={railW != null ? { "--rail-w": `${railW}px` } : undefined}>
      <div className="stage">
        <div className="stage-head">
          <div className="stack">
            <span className="eyebrow">{run.project}</span>
            <h2 className="title" style={{ fontSize: 22 }}>{run.title}</h2>
          </div>
          <PipelineNav steps={STEPS} current={step} unlocked={unlocked} onJump={goTo} />
        </div>

        <div className={`stage-body ${step === "insights" ? "fill" : ""}`}>
          {step === "setup" && <SetupStep run={run} model={model} setModel={setModel} lora={lora} setLora={setLora} onSelectDataset={() => onBack()} />}
          {step === "audit" && <AuditStep run={run} auditRun={auditRun} tracked={null} />}
          {step === "insights" && <InsightsStep run={run} reveal={reveal} mitigated={mitigated} steerRun={steerRun} mitReveal={mitReveal} />}
          {step === "checkout" && <CheckoutStep run={run} mitigated={mitigated} />}
        </div>
      </div>

      <InsightStream items={items} live={insightsActive && reveal < 1}
        onResizeStart={startRailResize} onResizeKey={nudgeRail} />
    </div>
  );
}
