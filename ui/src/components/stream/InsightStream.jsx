import { useEffect, useRef } from "react";
import StreamItem from "./StreamItem.jsx";

/* The right rail. A dumb subscriber: it renders whatever typed items it is handed,
   in order, and auto-scrolls. Producers (the agent, programmatic triggers) append
   items elsewhere; this panel knows nothing about where they came from. */
export default function InsightStream({ items, live }) {
  const ref = useRef(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [items.length]);

  return (
    <div className="rail">
      <div className="rail-head">
        <span className="t">Insights</span>
        {live && <span className="live"><span className="pip" /> live</span>}
      </div>
      <div className="stream" ref={ref}>
        {items.map((it) => <StreamItem key={it.id} item={it} />)}
        {items.length === 0 && <div className="muted" style={{ fontSize: 13 }}>Insights and proposed actions will stream here as the run progresses.</div>}
      </div>
    </div>
  );
}
