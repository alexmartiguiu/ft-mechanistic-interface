import { useEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import ConfigEditor from "../components/ConfigEditor.jsx";
import InsightStream from "../components/stream/InsightStream.jsx";
import Switch from "../components/Switch.jsx";
import { LORA_PRESETS } from "../api/sampleData.js";
import { eventToItem } from "../api/adapters.js";
import * as api from "../api/client.js";

// Cosmetic LoRA presets → concrete recipe overrides patched onto the seeded lora.yaml.
const LORA_OVERRIDES = {
  balanced: { lora: { r: 16, alpha: 32 }, optim: { epochs: 3 } },
  light: { lora: { r: 8, alpha: 16 }, optim: { epochs: 2 } },
  heavy: { lora: { r: 64, alpha: 128 }, optim: { epochs: 3 } },
};

function DropZone({ onAttach, attaching, error, onCancel }) {
  const [drag, setDrag] = useState(false);
  const inputRef = useRef(null);
  const pick = (files) => { const f = files?.[0]; if (f) onAttach(f); };
  return (
    <div
      className={`card ds-drop ${drag ? "over" : ""} ${attaching ? "busy" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files); }}
      onClick={() => !attaching && inputRef.current?.click()}
      role="button" tabIndex={0}
    >
      <input ref={inputRef} type="file" accept=".jsonl,.json,application/json,application/x-ndjson"
        hidden onChange={(e) => pick(e.target.files)} />
      <div className="ds-drop-icon" aria-hidden="true">{attaching ? "⏳" : "⬇"}</div>
      <div className="ds-drop-t">{attaching ? "Uploading + validating…" : "Drop a .jsonl dataset here"}</div>
      <div className="ds-drop-sub">or click to browse · one <span className="mono">{'{"messages": [...]}'}</span> per line</div>
      {error && <div className="ds-drop-err">{error}</div>}
      {onCancel && !attaching && (
        <button className="btn ghost sm" onClick={(e) => { e.stopPropagation(); onCancel(); }}>Cancel</button>
      )}
    </div>
  );
}

function DatasetView({ preview, onAttach, attaching, error }) {
  const [replacing, setReplacing] = useState(false);
  if (!preview) return <div className="card ds-summary"><div className="muted">Loading dataset…</div></div>;
  const cols = preview.columns || [];
  const hasData = (preview.rows || []).length > 0 || (preview.n_rows || 0) > 0;

  // wrap onAttach so a *successful* upload closes the "replace" panel
  const attach = async (file) => { if (await onAttach(file)) setReplacing(false); };

  if (!hasData || replacing) {
    return <DropZone onAttach={attach} attaching={attaching} error={error}
      onCancel={hasData ? () => setReplacing(false) : null} />;
  }
  return (
    <div className="card ds-summary fill">
      <div className="ds-summary-head">
        <span className="ds-name mono">{preview.rel_path || `${preview.domain}/sft.jsonl`}</span>
        <button className="btn ghost sm" onClick={() => setReplacing(true)}>↩ Replace dataset</button>
      </div>
      <div className="ds-stats"><span><b className="mono">{(preview.n_rows || 0).toLocaleString()}</b> examples</span></div>
      <div className="ds-scroll">
        <table className="ds-table mini">
          <thead><tr>{cols.map((c) => <th key={c.key}>{c.label}</th>)}</tr></thead>
          <tbody>
            {(preview.rows || []).map((r, i) => (
              <tr key={i}>{cols.map((c) => <td key={c.key} className="cell"><div className="clamp">{r[c.key]}</div></td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* Setup workspace (LIVE): hedda authors the configs pre-launch. The agent runs in the
   right rail; when it calls set_concepts / set_lora the backend emits a ConfigEvent, which
   we turn into a tree+gate refetch so the dev-mode editor + launch button update live. The
   dev toggle swaps the left panel between the dataset/config pickers and the dark YAML editor.
   Launching (button or the agent's "Launch run" action) hands the new run id up to App. */
export default function SetupWorkspace({ projectId, meta, dev, onToggleDev, onLaunch, onBack }) {
  const [tree, setTree] = useState([]);
  const [status, setStatus] = useState(null);
  const [preview, setPreview] = useState(null);
  const [lora, setLora] = useState("balanced");
  const [phase, setPhase] = useState("loading");     // loading | ready | error
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState(null);
  const [attaching, setAttaching] = useState(false);
  const [attachErr, setAttachErr] = useState(null);

  const [items, setItems] = useState([]);
  const [thinking, setThinking] = useState(true);
  const idRef = useRef(0);
  const sidRef = useRef(null);
  const startedRef = useRef(false);

  const push = (it) => setItems((p) => [...p, { ...it, id: ++idRef.current }]);

  async function refresh() {
    const [t, s] = await Promise.all([api.getConfigTree(projectId), api.getConfigStatus(projectId)]);
    setTree(t.files || []); setStatus(s);
    return s;
  }

  // ── agent rail wiring (mirrors LiveRunView) ──
  const onAnswer = (ref, vals) => { setThinking(true); if (sidRef.current) api.postAnswer(sidRef.current, ref, vals); };
  const onAction = (ref, label) => {
    if (/launch/i.test(label || "")) { launch(); return; }   // the agent's "Launch run" button
    setThinking(true);
    if (sidRef.current) api.postAction(sidRef.current, ref);
  };

  function handleSubagent(ev) {
    setItems((p) => {
      const i = p.findIndex((it) => it.type === "subagent" && it.ref === ev.ref);
      if (ev.phase === "start") {
        if (i >= 0) return p;
        return [...p, {
          id: ++idRef.current, type: "subagent", ref: ev.ref,
          title: (ev.title && ev.title !== "Concept Proposal") ? ev.title : "Understanding emergent risks",
          agent: ev.agent || "concept-proposer", steps: [], status: null, done: false,
        }];
      }
      if (i < 0) return p;
      const next = p.slice();
      if (ev.phase === "step" && ev.step) next[i] = { ...next[i], steps: [...next[i].steps, ev.step] };
      else if (ev.phase === "done") next[i] = { ...next[i], done: true, status: ev.status || "done" };
      return next;
    });
  }

  function handleEvent(ev) {
    if (ev.channel === "config") { refresh(); return; }        // agent wrote a config → live update
    if (ev.channel === "rail") {
      if (ev.kind === "subagent") { handleSubagent(ev); setThinking(true); return; }
      const item = eventToItem(ev, { onAnswer, onAction });
      if (item) push(item);
      setThinking(ev.kind !== "question" && ev.kind !== "action");
      return;
    }
    if (ev.kind === "status" && (ev.payload?.turn_done || ev.payload?.error)) setThinking(false);
  }

  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;
    let unsub = null, cancelled = false;
    (async () => {
      try {
        await refresh();
        try { setPreview(await api.getDatasetPreview(projectId)); } catch { /* new topic → none */ }
        if (cancelled) return;
        setPhase("ready");
        const { sid } = await api.createAuthoringSession(projectId, meta?.modelUse || null);
        if (cancelled) return;
        sidRef.current = sid;
        unsub = api.streamSession(sid, handleEvent);
      } catch (e) {
        if (!cancelled) { setError(String(e.message || e)); setPhase("error"); }
      }
    })();
    return () => {
      cancelled = true;
      if (unsub) unsub();
      if (sidRef.current) api.closeSession(sidRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  async function pickLora(id) {
    setLora(id);
    const { status: s } = await api.setLora(projectId, null, LORA_OVERRIDES[id] || {});
    setStatus(s);
    setTree((await api.getConfigTree(projectId)).files || []);
  }

  async function saveRaw(kind, content) {
    const { status: s } = await api.writeConfig(projectId, kind, content);
    setStatus(s);
    setTree((await api.getConfigTree(projectId)).files || []);
  }

  // upload a dropped/selected dataset → validate server-side → refresh preview + gate + tree.
  // returns true on success so the drop zone can close.
  async function attachDataset(file) {
    if (!/\.(jsonl|json)$/i.test(file.name)) {
      setAttachErr("please drop a .jsonl file"); return false;
    }
    setAttaching(true); setAttachErr(null);
    try {
      const content = await file.text();
      const { status: s, preview: pv } = await api.attachDataset(projectId, content, file.name);
      setStatus(s); setPreview(pv);
      setTree((await api.getConfigTree(projectId)).files || []);
      return true;
    } catch (e) {
      setAttachErr(String(e.message || e)); return false;
    } finally {
      setAttaching(false);
    }
  }

  async function launch() {
    if (launching) return;
    setLaunching(true); setError(null);
    try {
      const { run_id } = await api.launchProject(projectId);
      onLaunch(run_id, meta?.modelUse || null);
    } catch (e) { setError(String(e.message || e)); setLaunching(false); }
  }

  if (phase === "error") return (
    <div className="page">
      <div className="muted" style={{ color: "var(--bad)", marginTop: 40 }}>
        Couldn't load the setup workspace: {error}
      </div>
      <button className="btn ghost" style={{ marginTop: 16 }} onClick={onBack}>Back</button>
    </div>
  );

  const gate = status?.launchable;
  const gateHint = gate ? "configs complete" : (status?.reasons?.[0] || "author the configs to launch");

  return (
    <div className="runview">
      <div className="stage">
        <div className="stage-head setup-head">
          <div>
            <div className="eyebrow">live · {meta?.domain || ""}</div>
            <h2 className="setup-title">{meta?.name || "Setup"}</h2>
          </div>
          <div className="row gap10">
            <button className="btn ghost" onClick={onBack} disabled={launching}>Cancel</button>
            <button className="btn primary" onClick={launch} disabled={!gate || launching} title={gateHint}>
              {launching ? "Launching…" : "Launch run →"}
            </button>
          </div>
        </div>

        {!gate && <div className="gate-banner">{gateHint}</div>}
        {gate && status?.vectors && !status.vectors.ok && status.vectors.reason && (
          <div className="gate-banner info">⛏ {status.vectors.reason} · minting runs first, so first signal takes a few minutes.</div>
        )}
        {error && <div className="gate-banner bad">{error}</div>}

        <div className="stage-body fill">
          <div className="flip-scene">
            <motion.div
              className="flip-card"
              animate={{ rotateY: dev ? 180 : 0 }}
              transition={{ type: "spring", stiffness: 260, damping: 30 }}
              style={{ transformStyle: "preserve-3d" }}
            >
              {/* front face — the guided setup panels */}
              <div className="flip-face front" aria-hidden={dev}>
                <div className="step setup-step"><div className="setup-cols">
                  <div className="setup-data">
                    <div className="section-title"><h3>Dataset</h3></div>
                    <DatasetView preview={preview} onAttach={attachDataset}
                      attaching={attaching} error={attachErr} />
                  </div>
                  <div className="setup-config">
                    <div className="section">
                      <div className="section-title"><h3>Malign concepts to track</h3>
                        <span className="hint">{status?.concepts_n || 0} selected · hedda proposes these →</span></div>
                      <div className="choice-row col">
                        {(status?.concepts || []).map((name) => (
                          <div key={name} className="choice on"><span className="mono">{name}</span></div>
                        ))}
                        {!(status?.concepts || []).length &&
                          <div className="muted">hedda is proposing malign concepts in the panel on the right…</div>}
                      </div>
                    </div>
                    <div className="section">
                      <div className="section-title"><h3>LoRA recipe</h3></div>
                      <div className="choice-row col">
                        {LORA_PRESETS.map((p) => (
                          <div key={p.id} className={`choice ${lora === p.id ? "on" : ""}`} onClick={() => pickLora(p.id)}>
                            {p.label}{p.recommended && <span className="rec-badge">recommended</span>}
                            <div className="meta mono">{p.meta}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div></div>
              </div>
              {/* back face — the dark YAML editor */}
              <div className="flip-face back" aria-hidden={!dev}>
                <ConfigEditor files={tree} status={status} onSave={saveRaw} />
              </div>
            </motion.div>
          </div>
        </div>

        {onToggleDev && (
          <div className="stage-switch">
            <Switch checked={dev} onChange={onToggleDev} label="Developer mode" />
          </div>
        )}
      </div>

      <InsightStream items={items} live thinking={thinking} />
    </div>
  );
}
