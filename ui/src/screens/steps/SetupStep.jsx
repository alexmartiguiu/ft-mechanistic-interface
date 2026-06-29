import { useState } from "react";
import DatasetChooser from "../../components/DatasetChooser.jsx";
import HFLogo from "../../components/HFLogo.jsx";
import Chip from "../../components/Chip.jsx";
import { BASE_MODELS, LORA_PRESETS, HF_MODELS } from "../../api/sampleData.js";

function DatasetSummary({ run, onReplace }) {
  const cols = run.dataset.columns;
  return (
    <div className="card ds-summary">
      <div className="ds-summary-head">
        <div className="row gap10">
          <span className="ds-name mono">{run.dataset.domain}/sft.jsonl</span>
          <Chip color="var(--good)">results ready</Chip>
        </div>
        {onReplace && <button className="btn ghost sm" onClick={onReplace}>↩ Replace dataset</button>}
      </div>
      <div className="ds-stats">
        <span><b className="mono">{run.audit.total.toLocaleString()}</b> examples</span>
        <span><b className="mono">messages</b> format</span>
        <span>fields <b className="mono">{cols.map((c) => c.label).join(", ")}</b></span>
      </div>
      <table className="ds-table mini">
        <thead><tr>{cols.map((c) => <th key={c.key}>{c.label}</th>)}</tr></thead>
        <tbody>
          {run.dataset.rows.slice(0, 3).map((r, i) => (
            <tr key={i}>{cols.map((c) => <td key={c.key} className="cell">{r[c.key]}</td>)}</tr>
          ))}
        </tbody>
      </table>
      <div className="ds-more muted">+ {(run.audit.total - 3).toLocaleString()} more rows</div>
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

export default function SetupStep({ run, model, setModel, lora, setLora, onSelectDataset }) {
  if (!run) return <DatasetChooser onSelect={onSelectDataset} />;

  return (
    <div>
      <div className="section">
        <div className="section-title"><h3>Dataset</h3><span className="hint">loaded · chat-JSONL (messages)</span></div>
        <DatasetSummary run={run} onReplace={() => onSelectDataset(null)} />
      </div>

      <div className="field-row">
        <div className="field">
          <div className="section-title"><h3>Base model</h3></div>
          <ModelPicker model={model} setModel={setModel} />
        </div>
        <div className="field">
          <div className="section-title"><h3>LoRA recipe</h3></div>
          <div className="choice-row">
            {LORA_PRESETS.map((p) => (
              <div key={p.id} className={`choice ${lora === p.id ? "on" : ""}`} onClick={() => setLora(p.id)}>
                {p.label}{p.recommended ? " ·  recommended" : ""}
                <div className="meta mono">{p.meta}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <p className="sub" style={{ marginTop: 22 }}>
        These three choices are the run's contract (application · concept set · LoRA recipe).
        Confirm them, then run the pre-training dataset audit — the proposed action is in the panel on the right.
      </p>
    </div>
  );
}
