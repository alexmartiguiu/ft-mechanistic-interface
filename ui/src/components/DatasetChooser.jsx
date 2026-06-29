import { useRef, useState } from "react";
import HFLogo from "./HFLogo.jsx";
import Chip from "./Chip.jsx";
import { DATASET_SOURCES, HF_DATASETS_EXTRA, runForDataset } from "../api/sampleData.js";

/* The New-Experiment entry: drop a dataset (or browse local / Hugging Face), or pick one of
   the precomputed sample datasets. Selecting anything we have results for binds that run and
   the normal pipeline continues; anything else shows the "would train live" path. */
export default function DatasetChooser({ onSelect }) {
  const fileRef = useRef(null);
  const [drag, setDrag] = useState(false);
  const [hfOpen, setHfOpen] = useState(false);
  const [q, setQ] = useState("");
  const [unmatched, setUnmatched] = useState(null);

  const tryQuery = (text, fallbackLabel) => {
    const m = runForDataset(text);
    if (m) { setUnmatched(null); onSelect(m.runId); }
    else setUnmatched(fallbackLabel || text);
  };

  const onDrop = (e) => {
    e.preventDefault(); setDrag(false);
    const internal = e.dataTransfer.getData("text/ftmi-run");
    if (internal) { onSelect(internal); return; }
    const f = e.dataTransfer.files?.[0];
    if (f) return tryQuery(f.name, f.name);
    const txt = e.dataTransfer.getData("text/plain");
    if (txt) tryQuery(txt, txt);
  };

  const hfResults = [
    ...DATASET_SOURCES.filter((d) => (d.hfId + d.label + d.domain).toLowerCase().includes(q.toLowerCase()))
      .map((d) => ({ id: d.hfId, label: d.label, n: d.n, runId: d.runId, ready: true })),
    ...HF_DATASETS_EXTRA.filter((d) => (d.hfId + d.label).toLowerCase().includes(q.toLowerCase()))
      .map((d) => ({ id: d.hfId, label: d.label, n: d.n, ready: false })),
  ];

  return (
    <div className="chooser">
      <div className={`bigdrop ${drag ? "over" : ""}`}
        onClick={() => fileRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}>
        <input ref={fileRef} type="file" accept=".jsonl,.json,.csv" style={{ display: "none" }}
          onChange={(e) => e.target.files?.[0] && tryQuery(e.target.files[0].name, e.target.files[0].name)} />
        <div className="bd-icon">⤓</div>
        <div className="bd-title">Drop a dataset to start a new experiment</div>
        <div className="bd-sub">JSONL or CSV from your machine — or a Hugging Face dataset id</div>
        <div className="bd-actions" onClick={(e) => e.stopPropagation()}>
          <button className="btn" onClick={() => fileRef.current?.click()}>Browse local files</button>
          <button className="btn hf-btn" onClick={() => setHfOpen((v) => !v)}>
            <HFLogo size={16} /> Browse Hugging Face
          </button>
        </div>
      </div>

      {unmatched && (
        <div className="warn-note">
          <b>No precomputed results for “{unmatched}”.</b> In a live run we'd mint concept vectors and
          fine-tune from scratch. For this demo, pick a dataset with results below.
        </div>
      )}

      {hfOpen && (
        <div className="card hf-panel">
          <div className="hf-head">
            <HFLogo size={18} />
            <span className="hf-t">Hugging Face datasets</span>
            <input className="hf-search" placeholder="search datasets…  (e.g. medical, fingpt)"
              value={q} onChange={(e) => setQ(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && q && tryQuery(q, q)} />
          </div>
          <div className="hf-list">
            {hfResults.map((r) => (
              <div key={r.id} className="hf-row"
                onClick={() => (r.ready ? onSelect(r.runId) : setUnmatched(r.label))}>
                <HFLogo size={15} />
                <div className="col" style={{ flex: 1, minWidth: 0 }}>
                  <span className="hf-id mono">{r.id}</span>
                  <span className="hf-meta">{r.label} · {r.n.toLocaleString()} rows</span>
                </div>
                {r.ready
                  ? <Chip color="var(--good)">results ready</Chip>
                  : <Chip tone="mute">would train live</Chip>}
              </div>
            ))}
            {hfResults.length === 0 && <div className="muted" style={{ padding: 10, fontSize: 13 }}>No matches.</div>}
          </div>
        </div>
      )}

      <div className="chooser-bottom">
      <div className="section-title">
        <h3 style={{ fontSize: 15 }}>Or start from a dataset with results</h3>
        <span className="hint">precomputed — drops straight into the pipeline</span>
      </div>
      <div className="sample-grid">
        {DATASET_SOURCES.map((d) => (
          <div key={d.runId} className="card sample-card" draggable
            onDragStart={(e) => e.dataTransfer.setData("text/ftmi-run", d.runId)}
            onClick={() => onSelect(d.runId)}>
            <div className="spread">
              <span className="sc-title">{d.label}</span>
              <Chip color="var(--good)">results</Chip>
            </div>
            <div className="sc-sub">{d.sub}</div>
            <div className="sc-foot mono">
              <span className="muted">{d.domain}/sft.jsonl</span>
              <span>{d.n.toLocaleString()} rows</span>
            </div>
            <div className="sc-hf"><HFLogo size={13} /> <span className="mono">{d.hfId}</span></div>
          </div>
        ))}
      </div>
      </div>
    </div>
  );
}
