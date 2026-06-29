import { useState } from "react";
import Gallery from "./screens/Gallery.jsx";
import RunView from "./screens/RunView.jsx";
import { getRun } from "./api/sampleData.js";

export default function App() {
  const [runId, setRunId] = useState(null);
  const run = runId ? getRun(runId) : null;

  return (
    <div className="app">
      <div className="topbar">
        <div className="brand" onClick={() => setRunId(null)}>
          <span className="mark">ftmi</span>
          <span className="by">interpretable fine-tuning</span>
        </div>
        <div className="crumbs">
          <span className="sep">/</span>
          <span className={run ? "" : "here"} onClick={() => setRunId(null)} style={{ cursor: "pointer" }}>Runs</span>
          {run && <><span className="sep">/</span><span className="here">{run.title}</span></>}
        </div>
        <div className="spacer" />
        <span className="badge">demo · replaying recorded runs</span>
      </div>

      {run
        ? <RunView key={runId} runId={runId} onBack={() => setRunId(null)} />
        : <Gallery onOpen={setRunId} />}
    </div>
  );
}
