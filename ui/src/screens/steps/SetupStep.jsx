import { useEffect, useState } from "react";
import DatasetChooser from "../../components/DatasetChooser.jsx";
import HFLogo from "../../components/HFLogo.jsx";
import { BASE_MODELS, LORA_PRESETS, HF_MODELS } from "../../api/sampleData.js";

// click a row to expand the full sample over a dimmed backdrop
function SampleModal({ row, cols, domain, onClose }) {
  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="modal-scrim" onClick={onClose}>
      <div className="modal sample-modal" role="dialog" aria-modal="true" aria-label="Training sample"
        onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <span className="eyebrow">Training sample · {domain}/sft.jsonl</span>
          <button className="btn ghost sm" onClick={onClose} aria-label="Close">✕</button>
        </div>
        <div className="sample-body">
          {cols.map((c) => (
            <div className="sample-turn" key={c.key}>
              <div className="sample-role">{c.label}</div>
              <div className="sample-text">{row[c.key]}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function DatasetSummary({ run, onReplace }) {
  const cols = run.dataset.columns;
  const rows = run.dataset.rows;
  const more = run.audit.total - rows.length;
  const [sel, setSel] = useState(null);
  return (
    <div className="card ds-summary fill">
      <div className="ds-summary-head">
        <div className="row gap10">
          <span className="ds-name mono">{run.dataset.domain}/sft.jsonl</span>
        </div>
        {onReplace && <button className="btn ghost sm" onClick={onReplace}>↩ Replace dataset</button>}
      </div>
      <div className="ds-stats">
        <span><b className="mono">{run.audit.total.toLocaleString()}</b> examples</span>
      </div>
      <div className="ds-scroll">
        <table className="ds-table mini">
          <thead><tr>{cols.map((c) => <th key={c.key}>{c.label}</th>)}</tr></thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="ds-row" onClick={() => setSel(r)} title="Click to expand">
                {cols.map((c) => <td key={c.key} className="cell"><div className="clamp">{r[c.key]}</div></td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {more > 0 && <div className="ds-more muted">{more.toLocaleString()} more rows</div>}
      {sel && <SampleModal row={sel} cols={cols} domain={run.dataset.domain} onClose={() => setSel(null)} />}
    </div>
  );
}

function ModelPicker({ model, setModel }) {
  const [hf, setHf] = useState(false);
  return (
    <div>
      <div className="choice-row">
        {BASE_MODELS.map((m) => (
          <div key={m.id} className={`choice ${model === m.id ? "on" : ""}`} onClick={() => setModel(m.id)}>
            {m.label}
            <div className="meta mono">{m.repo}</div>
          </div>
        ))}
        <div className={`choice hf-choice ${hf ? "on" : ""}`} onClick={() => setHf((v) => !v)}>
          <span className="row gap6"><HFLogo size={15} /> Browse Hugging Face</span>
          <div className="meta">pick any base model</div>
        </div>
      </div>
      {hf && (
        <div className="card hf-panel" style={{ marginTop: 10 }}>
          <div className="hf-head"><HFLogo size={18} /><span className="hf-t">Hugging Face models</span></div>
          <div className="hf-list">
            {HF_MODELS.map((m) => (
              <div key={m.id} className="hf-row" onClick={() => { setModel(m.id); setHf(false); }}>
                <HFLogo size={15} />
                <span className="hf-id mono" style={{ flex: 1 }}>{m.label}</span>
                <span className="hf-meta mono">↓ {m.dl}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// "what does this model do in the world" — when a live session exists, share it with the
// agent mid-run; it rides the agent's next turn (the system prompt is already frozen).
function ModelUseField({ onApply, className = "", style }) {
  const [text, setText] = useState("");
  const [sent, setSent] = useState(false);
  const apply = () => { const t = text.trim(); if (t) { onApply(t); setSent(true); } };
  return (
    <div className={`section ${className}`.trim()} style={style}>
      <div className="section-title"><h3>What will this model be used for?</h3></div>
      <textarea className="use-input" rows={3} value={text}
        onChange={(e) => { setText(e.target.value); setSent(false); }}
        onBlur={apply}
        placeholder="e.g. a triage assistant that answers patient questions in a hospital portal" />
      {sent && <span className="muted" style={{ fontSize: 15.55, marginTop: 8, display: "inline-block" }}>✓ added to hedda’s context — she’ll use it next turn</span>}
    </div>
  );
}

export default function SetupStep({ run, model, setModel, lora, setLora, onSelectDataset, onApplyIntent }) {
  if (!run) return <DatasetChooser onSelect={onSelectDataset} />;

  // sections cascade in one after another for an easier read: intent → dataset →
  // base model → recipe. Delays follow DOM order; each present block advances the step.
  const STEP = 0.14; // seconds between successive reveals
  let n = 0;
  const reveal = () => ({ animationDelay: `${(n++) * STEP}s` });

  return (
    <div className="step setup-step">
      <div className="setup-cols">
        {onApplyIntent && <ModelUseField onApply={onApplyIntent} className="setup-reveal" style={reveal()} />}

        <div className="setup-data setup-reveal" style={reveal()}>
          <div className="section-title"><h3>Dataset</h3></div>
          <DatasetSummary run={run} onReplace={() => onSelectDataset(null)} />
        </div>

        <div className="setup-config">
          <div className="section setup-reveal" style={reveal()}>
            <div className="section-title"><h3>Base model</h3></div>
            <ModelPicker model={model} setModel={setModel} />
          </div>

          <div className="section setup-reveal" style={reveal()}>
            <div className="section-title"><h3>LoRA recipe</h3></div>
            <div className="choice-row col">
              {LORA_PRESETS.map((p) => (
                <div key={p.id} className={`choice ${lora === p.id ? "on" : ""}`} onClick={() => setLora(p.id)}>
                  {p.label}{p.recommended && <span className="rec-badge">recommended</span>}
                  <div className="meta mono">{p.meta}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
