import { useEffect, useState } from "react";
import PageHeader from "../components/PageHeader.jsx";
import ConfigEditor from "../components/ConfigEditor.jsx";
import * as api from "../api/client.js";

/* Developer-mode view of an EXISTING run's config — read-only, "the YAML as it is".
   Works for a launched live run (numeric `runId`) or a recorded replay run
   (`frontendRun`, resolved to its backend run id first). */
export default function RunConfigView({ runId = null, frontendRun = null, onBack }) {
  const [files, setFiles] = useState([]);
  const [phase, setPhase] = useState("loading");   // loading | ready | error
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const id = runId != null
          ? runId
          : (await api.resolveRun(frontendRun.domain, frontendRun.model.id)).run_id;
        const tree = await api.getRunConfigTree(id);
        if (!cancelled) { setFiles(tree.files || []); setPhase("ready"); }
      } catch (e) {
        if (!cancelled) { setError(String(e.message || e)); setPhase("error"); }
      }
    })();
    return () => { cancelled = true; };
  }, [runId, frontendRun]);

  return (
    <div className="page">
      <PageHeader
        eyebrow="developer mode · read-only"
        title="Run configuration"
        sub="The exact application / concepts / lora YAML this run was launched from."
        right={<button className="btn ghost" onClick={onBack}>Close</button>}
      />
      {phase === "error" && <div className="muted" style={{ color: "var(--bad)" }}>Couldn't load config: {error}</div>}
      {phase === "loading" && <div className="muted">Loading config…</div>}
      {phase === "ready" && (files.length
        ? <div className="cfg-page-fill"><ConfigEditor files={files} /></div>
        : <div className="muted">No config recorded for this run.</div>)}
    </div>
  );
}
