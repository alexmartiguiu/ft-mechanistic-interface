import { useState } from "react";
import Gallery from "./screens/Gallery.jsx";
import RunView from "./screens/RunView.jsx";
import { getRun } from "./api/sampleData.js";

export default function App() {
  // open = null (gallery) | { runId } (existing run) | { new: true } (new experiment)
  const [open, setOpen] = useState(null);
  const run = open?.runId ? getRun(open.runId) : null;

  return (
    <div className="app">
      <div className="topbar">
        <div className="brand" onClick={() => setOpen(null)}>
          <span className="mark">ftmi</span>
          <span className="by">interpretable fine-tuning</span>
        </div>
        <div className="crumbs">
          <span className="sep">/</span>
          <span className={open ? "" : "here"} onClick={() => setOpen(null)} style={{ cursor: "pointer" }}>Runs</span>
          {run && <><span className="sep">/</span><span className="here">{run.title}</span></>}
          {open?.new && <><span className="sep">/</span><span className="here">New experiment</span></>}
        </div>
        <div className="spacer" />
        <span className="badge">demo · replaying recorded runs</span>
      </div>

      {open
        ? <RunView key={open.runId || "new"} runId={open.runId || null} onBack={() => setOpen(null)} />
        : <Gallery onOpen={(id) => setOpen({ runId: id })} onNew={() => setOpen({ new: true })} />}
    </div>
  );
}
