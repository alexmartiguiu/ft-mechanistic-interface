import { useEffect, useLayoutEffect, useRef, useState } from "react";
import StreamItem from "./StreamItem.jsx";

/* A pinned-to-the-bottom "agent is working" row. Shown while we're waiting on the
   agent's next output (so the panel never looks frozen mid-turn), and as the very
   first placeholder before anything has streamed in. A live elapsed-seconds counter
   ticks beside the label so the wait is legible. The counter is per-turn: the row is
   keyed by the item count at the call site, so each new agent output remounts it and
   the timer restarts from 0 (rather than accumulating across the whole session). */
/* Per-stage "what Hedda is doing" — each list is scoped to ONLY that pipeline step, so the
   working row never repeats the same beats across stages. Keep them brief and few (1-4);
   less is more. Setup = configure the run; Audit = find + flag risk; Realign = train, detect,
   mitigate; Checkout = conclude. */
const THINKING = {
  setup: [
    "Reading the dataset and your deployment context…",
    "Weighing the base model and LoRA recipe…",
  ],
  audit: [
    "Researching the drift risks for this domain…",
    "Proposing the malign concepts to track…",
    "Projecting every sample and flagging the risky tail…",
  ],
  insights: [
    "Fine-tuning with the projection monitor running…",
    "Reading which concepts drifted, and how far…",
    "Weighing the mitigation options…",
  ],
  checkout: [
    "Compiling the run's results and receipt…",
  ],
};
// the generic between-steps fallback: faintly absurd academic verbs (the coffee-fuelled,
// arXiv-doomscrolling, Reviewer-2-rebutting research life) — a wink, not a status report
const THINKING_DEFAULT = [
  "Caffeinating…",
  "Rebutting Reviewer 2…",
  "Doomscrolling arXiv…",
  "Grokking…",
  "Reticulating splines…",
  "Marginalising the priors…",
  "Overthinking this one…",
  "Consulting the oracle…",
  "Squinting at logits…",
  "Deriving it on a napkin…",
];
const THINKING_INIT = [
  "Reading the dataset and your deployment context…",
  "Sizing up the fine-tune ahead…",
];

/* The "agent is working" row: a checklist of actions that tick off one after another with
   a light-green check, the current one breathing (mirrors the audit-page action list). No
   duplicate label spinner — the only animation is the active step's dots. */
function ThinkingRow({ messages }) {
  const list = (Array.isArray(messages) ? messages : [messages]).filter(Boolean);
  const [secs, setSecs] = useState(0);
  const [done, setDone] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setSecs((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, []);
  useEffect(() => {
    const reduce = typeof window !== "undefined" && window.matchMedia
      && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) { setDone(list.length); return; }
    // check off each action in turn; the last stays active (the action still in progress)
    const timers = list.slice(0, -1).map((_, i) =>
      setTimeout(() => setDone(i + 1), 1600 * (i + 1)));
    return () => timers.forEach(clearTimeout);
  }, [list.length]);
  return (
    <div className="stream-item">
      <div className="si si-steps">
        <div className="who">
          <span className="who-t">Hedda thinks</span>
          <span className="think-timer mono">({secs}s)</span>
        </div>
        <ul className="ss-list think-list">
          {list.map((s, i) => {
            const state = i < done ? "done" : i === done ? "active" : "todo";
            return (
              <li key={i} className={`ss-step ${state}`}>
                <span className="ss-mark" aria-hidden="true">
                  {state === "done" ? "✓"
                    : state === "active" ? <span className="ss-dots"><i /><i /><i /></span>
                      : <span className="ss-pend" />}
                </span>
                <span className="ss-label">{s}</span>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}

/* The between-turns "still working" tell, shown AFTER a step's checklist has already run once,
   so the full checked-off list never repeats on every message — just a bare pulse. */
function MiniThinking() {
  const [secs, setSecs] = useState(0);
  const [w, setW] = useState(() => Math.floor(Math.random() * THINKING_DEFAULT.length));
  useEffect(() => {
    const t = setInterval(() => setSecs((s) => s + 1), 1000);
    const r = setInterval(() => setW((n) => (n + 1) % THINKING_DEFAULT.length), 2200);
    return () => { clearInterval(t); clearInterval(r); };
  }, []);
  return (
    <div className="stream-item">
      <div className="si si-steps">
        <div className="who">
          <span className="who-t">Hedda thinks</span>
          <span className="think-dots" aria-hidden="true"><i /><i /><i /></span>
          <span className="think-mini">{THINKING_DEFAULT[w]}</span>
          <span className="think-timer mono">({secs}s)</span>
        </div>
      </div>
    </div>
  );
}

/* The right rail. A dumb subscriber: it renders whatever typed items it is handed,
   in order, and stays pinned to the bottom so the latest agent output is always in view.
   Pinning pauses if you scroll up to read, and resumes once you're back near the bottom.
   `thinking` keeps a spinner pinned at the bottom while the agent works between events. */
export default function InsightStream({ items, live, thinking, step, onResizeStart, onResizeKey }) {
  const scrollRef = useRef(null);
  const contentRef = useRef(null);
  const stick = useRef(true);

  // Show the full "what Hedda is doing" checklist ONCE per pipeline step, at its start — not after
  // every message. Mark a step's checklist shown when its thinking period ends; repeats within the
  // same step get a bare pulse (MiniThinking) instead of re-running the whole checked-off list.
  const shownRef = useRef(new Set());
  const wasThinking = useRef(false);
  useEffect(() => {
    if (wasThinking.current && !thinking && step) shownRef.current.add(step);
    wasThinking.current = thinking;
  }, [thinking, step]);
  const firstForStep = !step || !shownRef.current.has(step);

  const toBottom = () => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  };

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const dist = el.scrollHeight - el.scrollTop - el.clientHeight;
    stick.current = dist < 80; // near the bottom → keep following
  };

  // Follow ANY content-height change: new items, an item growing (question/action),
  // or staggered streamed inserts — not just item-count changes.
  useEffect(() => {
    const content = contentRef.current;
    if (!content || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => { if (stick.current) toBottom(); });
    ro.observe(content);
    return () => ro.disconnect();
  }, []);

  useLayoutEffect(() => {
    if (!stick.current) return;
    toBottom();
    const id = requestAnimationFrame(toBottom); // again after paint (late layout/fonts)
    return () => cancelAnimationFrame(id);
  }, [items.length, thinking]);

  return (
    <div className="rail">
      {onResizeStart && (
        <div className="rail-resizer" role="separator" aria-orientation="vertical"
          aria-label="Resize agent panel" tabIndex={0}
          onPointerDown={onResizeStart} onKeyDown={onResizeKey} />
      )}
      <div className="rail-head">
        <span className="rail-name"><span className="t">Hedda</span><span className="rail-sub">agents</span></span>
        <div className="rail-head-right">
          {live && <span className="live"><span className="pip" /> live</span>}
        </div>
      </div>
      <div className="stream" ref={scrollRef} onScroll={onScroll}>
        <div className="stream-content" ref={contentRef}>
          {items.map((it) => <StreamItem key={it.id} item={it} />)}
          {(items.length === 0 || thinking) && (firstForStep
            // first time we hit this step → run its checklist once (keyed by step so it stays
            // mounted across turns within the step, rather than remounting per message)
            ? <ThinkingRow key={`step-${step || "init"}`} messages={THINKING[step] || THINKING_INIT} />
            // already shown this step's checklist → just a bare "still working" pulse
            : <MiniThinking key="mini" />)}
        </div>
      </div>
    </div>
  );
}
