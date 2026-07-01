import { useEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import PipelineNav from "../components/PipelineNav.jsx";
import InsightStream from "../components/stream/InsightStream.jsx";
import ConfigEditor from "../components/ConfigEditor.jsx";
import Switch from "../components/Switch.jsx";
import SetupStep from "./steps/SetupStep.jsx";
import AuditStep from "./steps/AuditStep.jsx";
import InsightsStep from "./steps/InsightsStep.jsx";
import CheckoutStep from "./steps/CheckoutStep.jsx";
import RunView from "./RunView.jsx";
import * as api from "../api/client.js";
import {
  bundleToRun, steerRunFromBundle, eventToItem,
  applyCurves, applyAudit, applySteer,
} from "../api/adapters.js";
import { WRAPUP_QUESTION, WRAPUP_OPTIONS, emailReport } from "../lib/wrapup.js";

const STEPS = [
  { id: "setup", label: "Setup" },
  { id: "audit", label: "Audit" },
  { id: "insights", label: "Realign" },
  { id: "checkout", label: "Checkout" },
];

/* Live run: the same shell as RunView, but the `run` data comes from the backend
   view bundle and the narration + timing are driven by the agent's SSE stream
   (rail events → items; stage events → step / reveal / morph). Falls back to the
   scripted RunView if the backend can't bind this run. */
/* `frontendRun` = an existing recorded run (replay); `liveRunId` = a freshly-created
   live run to bind directly (skips resolveRun). Mode is decided server-side per project. */
export default function LiveRunView({ frontendRun = null, liveRunId = null, modelUse = null,
                                     dev = false, onToggleDev, onBack, onStep }) {
  const isLive = liveRunId != null;
  const [phase, setPhase] = useState("loading");   // loading | ready | error
  const [run, setRun] = useState(null);
  const [configFiles, setConfigFiles] = useState([]);   // read-only YAML tree (dev-mode flip)
  const [steerRun, setSteerRun] = useState(null);
  const runRef = useRef(null);   // current run, for stale-free reads inside handleEvent
  const [step, setStep] = useState("setup");
  useEffect(() => { onStep?.(step); }, [step]);   // reflect the pipeline step in the URL
  const [unlocked, setUnlocked] = useState(new Set(["setup"]));
  const [completed, setCompleted] = useState(new Set());   // steps the run has closed out (lights Checkout terracotta)
  const [items, setItems] = useState([]);
  const [auditRun, setAuditRun] = useState(false);
  const [insightsActive, setInsightsActive] = useState(false);
  const [mitActive, setMitActive] = useState(false);
  const [mitigated, setMitigated] = useState(false);
  // agent is mid-turn → keep a spinner pinned in the rail. True from kickoff;
  // cleared when the turn hands control back (a question/action, or turn_done).
  const [thinking, setThinking] = useState(true);
  const [model, setModel] = useState(frontendRun?.model?.id ?? "apertus-8b");
  const [lora, setLora] = useState("balanced");
  const [railW, setRailW] = useState(null);

  // phase-complete flags set by InsightsStep when each staged fill finishes (gate the
  // "live" pill + the event-driven early-stop marker). Staged reveal lives in InsightsStep.
  const [trainRevealed, setTrainRevealed] = useState(false);
  const [steerRevealed, setSteerRevealed] = useState(false);

  const idRef = useRef(0);
  const sidRef = useRef(null);
  const startedRef = useRef(false);
  const firedRef = useRef(new Set());

  const push = (it) => setItems((p) => [...p, { ...it, id: ++idRef.current }]);
  const goTo = (s) => { setStep(s); setUnlocked((u) => new Set(u).add(s)); };
  const fireOnce = (k) => (firedRef.current.has(k) ? false : (firedRef.current.add(k), true));

  const onAnswer = (ref, vals) => {
    setThinking(true);   // a new turn begins → show the spinner again immediately
    if (sidRef.current) api.postAnswer(sidRef.current, ref, vals);
  };
  const onAction = (ref, label) => {
    setThinking(true);
    if (sidRef.current) api.postAction(sidRef.current, ref);
    if (/checkout/i.test(label || "")) goTo("checkout");
  };

  // the final wrap-up choice. Confirming it marks Checkout complete (its node turns
  // terracotta) and then does the chosen thing.
  function onWrapUp(choice) {
    setCompleted((c) => new Set(c).add("checkout"));
    if (choice === "Email the 1-page report") {
      emailReport(runRef.current, steerRun);
      push({ type: "insight", lead: "I have drafted a summary email in your mail client.",
        bullets: ["Attach the one-page report (use Download 1-pager on the receipt) before sending."] });
      return;
    }
    setTimeout(() => onBack && onBack(), 450);   // Back to projects / Close this run → the gallery
  }

  // a nested subagent panel upserts in place (keyed by ref): start opens it,
  // each step appends a tool call, done swaps the spinner for a status line.
  function handleSubagent(ev) {
    setItems((p) => {
      const i = p.findIndex((it) => it.type === "subagent" && it.ref === ev.ref);
      if (ev.phase === "start") {
        if (i >= 0) return p;
        return [...p, {
          id: ++idRef.current, type: "subagent", ref: ev.ref,
          title: (ev.title && ev.title !== "Concept Proposal") ? ev.title : "Understanding emergent risks",
          agent: ev.agent || "concept-proposer",
          steps: [], status: null, done: false,
        }];
      }
      if (i < 0) return p;
      const next = p.slice();
      if (ev.phase === "step" && ev.step) {
        next[i] = { ...next[i], steps: [...next[i].steps, ev.step] };
      } else if (ev.phase === "done") {
        next[i] = { ...next[i], done: true, status: ev.status || "done" };
      }
      return next;
    });
  }

  function handleEvent(ev) {
    if (ev.channel === "rail") {
      if (ev.kind === "subagent") { handleSubagent(ev); setThinking(true); return; }
      const item = eventToItem(ev, { onAnswer, onAction });
      if (item) push(item);
      // a question/action hands control to the user; anything else means the agent
      // is still working toward its next output → keep the spinner up.
      setThinking(ev.kind !== "question" && ev.kind !== "action");
      return;
    }
    // a turn completing (or erroring) is the only "agent is idle" signal
    if (ev.kind === "status") {
      if (ev.payload?.turn_done || ev.payload?.error) setThinking(false);
      return;
    }
    // stage directives drive the left panel. Each payload IS a RunCurves/AuditResult/
    // SteerResult dump — fold it into `run` so live (initially-empty) plots fill as they
    // stream; in replay the bundle already holds the same data, so the merge is harmless.
    if (ev.kind === "audit_view") {
      // concepts proposed → show the audit view now; the extraction method animates
      // (auditRun stays false) while the user picks which concepts to track. The dataset
      // view stays folded until audit_flagged.
      goTo("audit");
    }
    else if (ev.kind === "audit_flagged") {
      if (ev.payload) setRun((r) => (r ? applyAudit(r, ev.payload) : r));
      setAuditRun(true); goTo("audit");
    }
    else if (ev.kind === "training_started") { goTo("insights"); }
    else if (ev.kind === "training_fill") {
      if (ev.payload) setRun((r) => (r ? applyCurves(r, ev.payload) : r));
      goTo("insights"); if (fireOnce("fill")) setInsightsActive(true);
    }
    else if (ev.kind === "mitigation_morph") {
      if (ev.payload) {
        const { steer, steerRun: sr } = applySteer(ev.payload, runRef.current?.model?.label);
        setRun((r) => (r ? { ...r, steer } : r));
        if (sr) setSteerRun(sr);
      }
      if (fireOnce("morph")) { setMitigated(true); setTimeout(() => setMitActive(true), 800); }
    }
  }

  // keep runRef in sync so handleEvent (captured once) can read the latest run
  useEffect(() => { runRef.current = run; }, [run]);

  // once the receipt is reached the run is closed out → offer the wrap-up choice
  // (this is also what lights the Checkout node). Client-side, so it fires whether or
  // not the live stream sends a closing turn.
  useEffect(() => {
    if (step !== "checkout" || !fireOnce("wrapup")) return;
    const t = setTimeout(() => {
      setThinking(false);
      push({ type: "question", question: WRAPUP_QUESTION, multiSelect: false,
        confirmLabel: "closing out the run", options: WRAPUP_OPTIONS, onSubmit: (vals) => onWrapUp(vals[0]) });
    }, 1200);
    return () => clearTimeout(t);
  }, [step]);

  // ── setup: (resolve | use given live id) → bundle → session → stream ──
  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;
    let unsub = null, cancelled = false;
    (async () => {
      try {
        const run_id = isLive
          ? liveRunId
          : (await api.resolveRun(frontendRun.domain, frontendRun.model.id)).run_id;
        const bundle = await api.getRunView(run_id);
        if (cancelled) return;
        setRun(bundleToRun(bundle));
        setSteerRun(steerRunFromBundle(bundle));
        setPhase("ready");
        // read-only config tree for the dev-mode flip (the exact YAML this run used)
        api.getRunConfigTree(run_id).then((t) => { if (!cancelled) setConfigFiles(t.files || []); }).catch(() => {});
        const { sid } = await api.createSession(run_id, modelUse);   // mode derived server-side
        if (cancelled) return;
        sidRef.current = sid;
        unsub = api.streamSession(sid, handleEvent);
      } catch (e) {
        if (!cancelled) { console.warn("live mode unavailable:", e); setPhase("error"); }
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

  if (phase === "error") {
    // a created live run has no scripted fallback; an existing run falls back to RunView
    return isLive
      ? <div className="runview"><div className="stage"><div className="center-empty">Couldn’t start the live run. Check the backend / GPU and try again.</div></div><div className="rail" /></div>
      : <RunView runId={frontendRun?.id} onBack={onBack} />;
  }
  if (phase === "loading" || !run) {
    const msg = isLive ? "Starting the live run…" : "Binding to the recorded run…";
    return <div className="runview"><div className="stage"><div className="center-empty">{msg}</div></div><div className="rail" /></div>;
  }

  return (
    <div className="runview" style={railW != null ? { "--rail-w": `${railW}px` } : undefined}>
      <div className="stage">
        <div className="stage-head">
          <div className="stack" />
          <PipelineNav steps={STEPS} current={step} unlocked={unlocked} completed={completed} onJump={goTo} />
        </div>

        <div className="flip-scene">
          <motion.div className="flip-card" animate={{ rotateY: dev ? 180 : 0 }}
            transition={{ type: "spring", stiffness: 260, damping: 30 }}
            style={{ transformStyle: "preserve-3d" }}>
            {/* front face — the run's pipeline view */}
            <div className="flip-face front" aria-hidden={dev}>
              <div className={`stage-body ${step === "insights" || (step === "setup" && run) ? "fill" : (step === "setup" && !run) ? "center" : ""}`}>
                {step === "setup" && <SetupStep run={run} model={model} setModel={setModel} lora={lora} setLora={setLora}
                  onSelectDataset={() => onBack()}
                  onApplyIntent={(text) => sidRef.current && api.postIntent(sidRef.current, text)} />}
                {step === "audit" && <AuditStep run={run} auditRun={auditRun} tracked={null} thinking={thinking} />}
                {step === "insights" && <InsightsStep run={run} steerRun={steerRun}
                  active={insightsActive} steerActive={mitActive}
                  onTrainRevealed={() => setTrainRevealed(true)} onSteerRevealed={() => setSteerRevealed(true)}
                  earlyStopShown={trainRevealed} />}
                {step === "checkout" && <CheckoutStep run={run} mitigated={mitigated} steerRun={steerRun} />}
              </div>
            </div>
            {/* back face — the read-only YAML config this run used */}
            <div className="flip-face back" aria-hidden={!dev}>
              {configFiles.length
                ? <ConfigEditor files={configFiles} />
                : <div className="center-empty">No config recorded for this run.</div>}
            </div>
          </motion.div>
        </div>

        {onToggleDev && (
          <div className="stage-switch">
            <Switch checked={dev} onChange={onToggleDev} label="Developer mode" />
          </div>
        )}
      </div>

      <InsightStream items={items} live={insightsActive && !trainRevealed} thinking={thinking} step={step}
        onResizeStart={startRailResize} onResizeKey={nudgeRail} />
    </div>
  );
}
