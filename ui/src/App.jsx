import { useEffect, useState } from "react";
import Gallery from "./screens/Gallery.jsx";
import RunView from "./screens/RunView.jsx";
import LiveRunView from "./screens/LiveRunView.jsx";
import CreateProject from "./screens/CreateProject.jsx";
import SetupWorkspace from "./screens/SetupWorkspace.jsx";
import { getRun } from "./api/sampleData.js";
import * as api from "./api/client.js";

const DEV_KEY = "nauteus.devmode";
const THEME_KEY = "nauteus.theme";

export default function App() {
  // open = null (gallery) | { runId } (recorded run) | { new: true } (create a project)
  //      | { setupProject, meta } (authoring a live project) | { liveRunId } (streaming)
  const [open, setOpen] = useState(null);
  const [live, setLive] = useState(false);   // backend reachable → live runs available
  const [dev, setDev] = useState(() => {
    try { return localStorage.getItem(DEV_KEY) === "1"; } catch { return false; }
  });
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem(THEME_KEY) === "dark" ? "dark" : "light"; } catch { return "light"; }
  });
  const run = open?.runId ? getRun(open.runId) : null;

  useEffect(() => { api.health().then(setLive); }, []);
  // theme is a document-level attribute so it themes every screen, not just this tree
  useEffect(() => { document.documentElement.dataset.theme = theme; }, [theme]);
  const toggleDev = () => setDev((v) => {
    const n = !v; try { localStorage.setItem(DEV_KEY, n ? "1" : "0"); } catch { /* private mode */ }
    return n;
  });
  const toggleTheme = () => setTheme((t) => {
    const n = t === "dark" ? "light" : "dark";
    try { localStorage.setItem(THEME_KEY, n); } catch { /* private mode */ }
    return n;
  });

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
        <button className="theme-toggle" type="button" onClick={toggleTheme}
          aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}>
          {theme === "dark" ? "☀️" : "🌙"}
        </button>
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
              // a launched live run — dev mode flips the left panel to its config (in-panel switch)
              ? <LiveRunView key={`live-${open.liveRunId}`} liveRunId={open.liveRunId} modelUse={open.modelUse}
                             dev={dev} onToggleDev={toggleDev} onBack={() => setOpen(null)} />
              // a recorded run — same flip to its read-only YAML "as it is"
              : open.runId && live && run
                ? <LiveRunView key={open.runId} frontendRun={run}
                               dev={dev} onToggleDev={toggleDev} onBack={() => setOpen(null)} />
                : <RunView key={open.runId} runId={open.runId} onBack={() => setOpen(null)} />}
    </div>
  );
}
