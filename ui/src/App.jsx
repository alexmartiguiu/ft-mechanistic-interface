import { useEffect, useState } from "react";
import Gallery from "./screens/Gallery.jsx";
import RunView from "./screens/RunView.jsx";
import LiveRunView from "./screens/LiveRunView.jsx";
import CreateProject from "./screens/CreateProject.jsx";
import SetupWorkspace from "./screens/SetupWorkspace.jsx";
import RunConfigView from "./screens/RunConfigView.jsx";
import { getRun } from "./api/sampleData.js";
import * as api from "./api/client.js";

const DEV_KEY = "nauteus.devmode";

export default function App() {
  // open = null (gallery) | { runId } (recorded run) | { new: true } (create a project)
  //      | { setupProject, meta } (authoring a live project) | { liveRunId } (streaming)
  const [open, setOpen] = useState(null);
  const [live, setLive] = useState(false);   // backend reachable → live runs available
  const [dev, setDev] = useState(() => {
    try { return localStorage.getItem(DEV_KEY) === "1"; } catch { return false; }
  });
  const run = open?.runId ? getRun(open.runId) : null;

  useEffect(() => { api.health().then(setLive); }, []);
  const toggleDev = () => setDev((v) => {
    const n = !v; try { localStorage.setItem(DEV_KEY, n ? "1" : "0"); } catch { /* private mode */ }
    return n;
  });

  // dev mode only means something inside a run context; hide on gallery + create screen.
  // On the setup screen the switch lives IN the left panel (flips it), so hide the topbar one.
  const showDevToggle = !!open && !open.new && !open.setupProject;

  return (
    <div className="app">
      <div className="topbar">
        <div className="brand" onClick={() => setOpen(null)}>
          <span className="mark">Nauteus</span>
        </div>
        <div className="crumbs">
          <span className="sep">/</span>
          <span className={open ? "" : "here"} onClick={() => setOpen(null)} style={{ cursor: "pointer" }}>Projects</span>
          {run && <><span className="sep">/</span><span className="here">{run.project}</span></>}
          {(open?.new || open?.setupProject || open?.liveRunId) &&
            <><span className="sep">/</span><span className="here">
              {open.liveRunId ? "Live run" : open.setupProject ? "Setup" : "New project"}
            </span></>}
        </div>
        <div className="spacer" />
        {showDevToggle && (
          <button className={`devtoggle ${dev ? "on" : ""}`} onClick={toggleDev}
            title="Developer mode — view/edit the YAML configs">
            <span className="devtoggle-dot" /> Developer mode
          </button>
        )}
      </div>

      {!open
        ? <Gallery onOpen={(id) => setOpen({ runId: id })}
                   onOpenLive={(rid) => setOpen({ liveRunId: rid })}
                   onNew={() => setOpen({ new: true })} />
        : open.new
          // Step 1: create a project (topic + model). Falls back to scripted RunView if backend is down.
          ? (live
              ? <CreateProject onCreated={(pid, meta) => setOpen({ setupProject: pid, meta })}
                               onBack={() => setOpen(null)} />
              : <RunView key="new" runId={null} onBack={() => setOpen(null)} />)
          : open.setupProject
            // Step 2: author the configs, then launch → stream the new run.
            ? <SetupWorkspace projectId={open.setupProject} meta={open.meta} dev={dev} onToggleDev={toggleDev}
                              onLaunch={(rid, modelUse) => setOpen({ liveRunId: rid, modelUse })}
                              onBack={() => setOpen(null)} />
            : open.liveRunId
              // a launched live run — dev mode shows its config read-only
              ? (dev
                  ? <RunConfigView key={`cfg-${open.liveRunId}`} runId={open.liveRunId} onBack={toggleDev} />
                  : <LiveRunView key={`live-${open.liveRunId}`} liveRunId={open.liveRunId}
                                 modelUse={open.modelUse} onBack={() => setOpen(null)} />)
              // a recorded run — dev mode shows its YAML "as it is"
              : dev && run
                ? <RunConfigView key={`cfg-${open.runId}`} frontendRun={run} onBack={toggleDev} />
                : open.runId && live && run
                  ? <LiveRunView key={open.runId} frontendRun={run} onBack={() => setOpen(null)} />
                  : <RunView key={open.runId} runId={open.runId} onBack={() => setOpen(null)} />}
    </div>
  );
}
