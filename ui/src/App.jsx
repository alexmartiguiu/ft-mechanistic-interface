import { useEffect, useRef, useState } from "react";
import Gallery from "./screens/Gallery.jsx";
import RunView from "./screens/RunView.jsx";
import LiveRunView from "./screens/LiveRunView.jsx";
import CreateProject from "./screens/CreateProject.jsx";
import SetupWorkspace from "./screens/SetupWorkspace.jsx";
import { getRun } from "./api/sampleData.js";
import * as api from "./api/client.js";
import { routeToOpen, openToPath, openBase } from "./lib/router.js";

const DEV_KEY = "hedda.devmode";
const THEME_KEY = "hedda.theme";

// black-and-white line illustrations (inherit currentColor, so they flip with the theme)
const MoonIcon = () => (
  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"
    strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
  </svg>
);
const SunIcon = () => (
  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"
    strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <circle cx="12" cy="12" r="4.2" />
    <line x1="12" y1="1.5" x2="12" y2="4" /><line x1="12" y1="20" x2="12" y2="22.5" />
    <line x1="3.3" y1="3.3" x2="5.1" y2="5.1" /><line x1="18.9" y1="18.9" x2="20.7" y2="20.7" />
    <line x1="1.5" y1="12" x2="4" y2="12" /><line x1="20" y1="12" x2="22.5" y2="12" />
    <line x1="3.3" y1="20.7" x2="5.1" y2="18.9" /><line x1="18.9" y1="5.1" x2="20.7" y2="3.3" />
  </svg>
);

export default function App() {
  // open = null (gallery) | { runId } (recorded run) | { new: true } (create a project)
  //      | { setupProject, meta } (authoring a live project) | { liveRunId } (streaming)
  const [open, setOpen] = useState(() => routeToOpen(window.location.pathname).open);
  const [viewStep, setViewStep] = useState(null);   // current pipeline step, reported by the run view
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

  // ── URL routing: keep the address bar in sync with where you are ──
  // seeded to the initial location so a clean load / refresh doesn't push a redundant entry
  const urlRef = useRef({
    path: window.location.pathname,
    base: openBase(routeToOpen(window.location.pathname).open),
  });
  // opening/closing a run pushes a history entry; moving between its steps only replaces
  // (so Back returns to the gallery, not through every pipeline step)
  useEffect(() => {
    const path = openToPath(open, viewStep);
    if (path === urlRef.current.path) return;
    const base = openBase(open);
    window.history[base === urlRef.current.base ? "replaceState" : "pushState"](null, "", path);
    urlRef.current = { path, base };
  }, [open, viewStep]);
  // a new run-level screen clears the reported step until the run view reports its own
  const openKey = openBase(open);
  useEffect(() => { setViewStep(null); }, [openKey]);
  // Back / Forward → rebuild `open` from the URL
  useEffect(() => {
    const onPop = () => {
      const { open: o } = routeToOpen(window.location.pathname);
      urlRef.current = { path: window.location.pathname, base: openBase(o) };
      setOpen(o);
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);
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
          <span className="mark">Hedda</span>
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
          {theme === "dark" ? <SunIcon /> : <MoonIcon />}
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
              : <RunView key="new" runId={null} onStep={setViewStep} onBack={() => setOpen(null)} />)
          : open.setupProject
            // Step 2: author the configs, then launch → stream the new run.
            ? <SetupWorkspace projectId={open.setupProject} meta={open.meta} dev={dev} onToggleDev={toggleDev}
                              onLaunch={(rid, modelUse) => setOpen({ liveRunId: rid, modelUse })}
                              onBack={() => setOpen(null)} />
            : open.liveRunId
              // a launched live run — dev mode flips the left panel to its config (in-panel switch)
              ? <LiveRunView key={`live-${open.liveRunId}`} liveRunId={open.liveRunId} modelUse={open.modelUse}
                             dev={dev} onToggleDev={toggleDev} onStep={setViewStep} onBack={() => setOpen(null)} />
              // a recorded run — same flip to its read-only YAML "as it is"
              : open.runId && live && run
                ? <LiveRunView key={open.runId} frontendRun={run}
                               dev={dev} onToggleDev={toggleDev} onStep={setViewStep} onBack={() => setOpen(null)} />
                : <RunView key={open.runId} runId={open.runId} onStep={setViewStep} onBack={() => setOpen(null)} />}
    </div>
  );
}
