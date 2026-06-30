import { useEffect, useLayoutEffect, useRef } from "react";
import StreamItem from "./StreamItem.jsx";

/* A pinned-to-the-bottom "agent is working" row. Shown while we're waiting on the
   agent's next output (so the panel never looks frozen mid-turn), and as the very
   first placeholder before anything has streamed in. */
function ThinkingRow({ children }) {
  return (
    <div className="stream-item">
      <div className="si">
        <div className="who"><span className="who-t">Hedda thinks</span></div>
        <div className="si-think">
          <span className="think-dots" aria-hidden="true"><i /><i /><i /></span>
          <span>{children}</span>
        </div>
      </div>
    </div>
  );
}

/* The right rail. A dumb subscriber: it renders whatever typed items it is handed,
   in order, and stays pinned to the bottom so the latest agent output is always in view.
   Pinning pauses if you scroll up to read, and resumes once you're back near the bottom.
   `thinking` keeps a spinner pinned at the bottom while the agent works between events. */
export default function InsightStream({ items, live, thinking, onResizeStart, onResizeKey }) {
  const scrollRef = useRef(null);
  const contentRef = useRef(null);
  const stick = useRef(true);

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
        <span className="t">Hedda</span>
        <div className="rail-head-right">
          {live && <span className="live"><span className="pip" /> live</span>}
        </div>
      </div>
      <div className="stream" ref={scrollRef} onScroll={onScroll}>
        <div className="stream-content" ref={contentRef}>
          {items.map((it) => <StreamItem key={it.id} item={it} />)}
          {items.length === 0
            ? <ThinkingRow>Reading your dataset and use-case…</ThinkingRow>
            : thinking && <ThinkingRow>Thinking…</ThinkingRow>}
        </div>
      </div>
    </div>
  );
}
