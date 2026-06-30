import { useEffect, useState } from "react";
import Gallery from "./screens/Gallery.jsx";
import RunView from "./screens/RunView.jsx";
import LiveRunView from "./screens/LiveRunView.jsx";
import NewRun from "./screens/NewRun.jsx";
import { getRun } from "./api/sampleData.js";
import * as api from "./api/client.js";

export default function App() {
  // open = null (gallery) | { runId } (recorded run) | { new: true } (configure a live run)
  //      | { liveRunId } (a freshly-created live run, now streaming)
  const [open, setOpen] = useState(null);
  const [live, setLive] = useState(false);   // backend reachable → live runs available
  const run = open?.runId ? getRun(open.runId) : null;

  useEffect(() => { api.health().then(setLive); }, []);

  return (
    <div className="app">
      <div className="topbar">
        <div className="brand" onClick={() => setOpen(null)}>
          <span className="mark">Hedda</span>
        </div>
        <div className="crumbs">
          <span className="sep">/</span>
          <span className={open ? "" : "here"} onClick={() => setOpen(null)} style={{ cursor: "pointer" }}>Projects</span>
          {run && <><span className="sep">/</span><span className="here">{run.project}</span></>}
          {(open?.new || open?.liveRunId) && <><span className="sep">/</span><span className="here">{open.liveRunId ? "Live run" : "New experiment"}</span></>}
        </div>
        <div className="spacer" />
      </div>

      {!open
        ? <Gallery onOpen={(id) => setOpen({ runId: id })}
                   onOpenLive={(rid) => setOpen({ liveRunId: rid })}
                   onNew={() => setOpen({ new: true })} />
        : open.liveRunId
          ? <LiveRunView key={`live-${open.liveRunId}`} liveRunId={open.liveRunId} modelUse={open.modelUse} onBack={() => setOpen(null)} />
          : open.new
            // configure a live run, then swap to streaming it; if backend is down, fall back to the scripted RunView
            ? (live
                ? <NewRun onCreated={(rid, modelUse) => setOpen({ liveRunId: rid, modelUse })} onBack={() => setOpen(null)} />
                : <RunView key="new" runId={null} onBack={() => setOpen(null)} />)
            : open.runId && live && run
              ? <LiveRunView key={open.runId} frontendRun={run} onBack={() => setOpen(null)} />
              : <RunView key={open.runId} runId={open.runId} onBack={() => setOpen(null)} />}
    </div>
  );
}
