import { useEffect, useState } from "react";
import { getHealth } from "./api.js";
import { useRuns } from "./state/useRuns.js";
import Sidebar from "./components/Sidebar.jsx";
import PageHeader from "./components/PageHeader.jsx";
import NewExperiment from "./screens/NewExperiment.jsx";
import Dashboard from "./screens/Dashboard.jsx";
import ConceptVectors from "./screens/ConceptVectors.jsx";
import LiveSteering from "./screens/LiveSteering.jsx";
import Runs from "./screens/Runs.jsx";

const PAGE_META = {
  new: { kicker: "Get started", title: "New experiment", sub: "Drop a dataset and let the hedda agent help you design a fine-tuning run." },
  dashboard: { kicker: "Overview", title: "Dashboard", sub: "How each fine-tuned model drifted on capability, truthfulness, and safety benchmarks." },
  vectors: { kicker: "Interpretability", title: "Concept Vectors", sub: "Extracted concept directions per domain — what we can detect and how strongly we can steer it." },
  steering: { kicker: "Live", title: "Live Steering", sub: "Dial a concept up or down and compare the model’s normal output against its steered output." },
  runs: { kicker: "Pipeline", title: "Runs", sub: "Launch a training or evaluation job against an existing config and watch its logs stream live." },
};

function pill(children, color = "#838fa4") {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "12.5px", color, padding: "7px 13px", borderRadius: "20px", border: "1px solid #e2e9f3", background: "#fff", fontFamily: "'JetBrains Mono',monospace" }}>
      {children}
    </div>
  );
}

export default function App() {
  const [route, setRoute] = useState("new");
  const [collapsed, setCollapsed] = useState(false);
  const [live, setLive] = useState(false);
  const [loadedModel, setLoadedModel] = useState(null);
  const [modelWarm, setModelWarm] = useState(false);
  const { runs, startRun, stopRun } = useRuns();

  // connection badge: probe the backend; fall back to demo data on failure
  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((j) => { if (!cancelled && j && j.ok) { setLive(true); setLoadedModel(j.loaded_model || null); } })
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);

  const meta = PAGE_META[route];
  const runningCount = runs.filter((r) => r.status === "running").length;

  let actions = null;
  if (route === "steering") {
    actions = pill(
      <>
        <span style={{ width: "7px", height: "7px", borderRadius: "50%", background: modelWarm ? "#2f9e7d" : "#b7c1d6" }} />
        {modelWarm ? "model warm" : "model idle"}
      </>
    );
  } else if (route === "runs") {
    actions = (
      <div style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "12.5px", color: "#838fa4", padding: "7px 13px", borderRadius: "20px", border: "1px solid #e2e9f3", background: "#fff" }}>
        <span style={{ width: "7px", height: "7px", borderRadius: "50%", background: runningCount ? "#1f9e86" : "#b7c1d6", animation: runningCount ? "lc-pulse 1.2s ease-in-out infinite" : "none" }} />
        {runningCount ? runningCount + " running" : "idle"}
      </div>
    );
  }

  return (
    <div style={{ display: "flex", height: "100vh", width: "100%", overflow: "hidden", fontFamily: "'Hanken Grotesk',system-ui,sans-serif", color: "#15203c", background: "#f6f8fc", WebkitFontSmoothing: "antialiased" }}>
      <Sidebar route={route} collapsed={collapsed} onNav={setRoute} onToggleCollapse={() => setCollapsed((c) => !c)} live={live} loadedModel={loadedModel} />

      <main style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", overflow: "hidden", background: "#f6f8fc" }}>
        <PageHeader kicker={meta.kicker} title={meta.title} subtitle={meta.sub} actions={actions} />
        <div style={{ flex: 1, overflowY: "auto", overflowX: "hidden" }}>
          <div style={{ padding: "30px 38px 56px", maxWidth: "1180px" }}>
            {route === "new" && <NewExperiment startRun={startRun} goRuns={() => setRoute("runs")} />}
            {route === "dashboard" && <Dashboard />}
            {route === "vectors" && <ConceptVectors />}
            {route === "steering" && <LiveSteering modelWarm={modelWarm} onWarm={() => setModelWarm(true)} />}
            {route === "runs" && <Runs runs={runs} startRun={startRun} stopRun={stopRun} />}
          </div>
        </div>
      </main>
    </div>
  );
}
