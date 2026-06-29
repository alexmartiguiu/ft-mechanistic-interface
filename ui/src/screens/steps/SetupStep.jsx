import { BASE_MODELS, LORA_PRESETS } from "../../api/sampleData.js";

export default function SetupStep({ run, model, setModel, lora, setLora }) {
  return (
    <div>
      <div className="section">
        <div className="section-title"><h3>Dataset</h3><span className="hint">browse local · or a Hugging Face dataset id</span></div>
        <div className="dropzone">
          <div className="big">{run.dataset.domain}/sft.jsonl · loaded</div>
          {run.sub} — chat-JSONL (messages format). Drag a local file or paste a HF dataset id to convert.
        </div>
      </div>

      <div className="field-row">
        <div className="field">
          <div className="section-title"><h3>Base model</h3></div>
          <div className="choice-row">
            {BASE_MODELS.map((m) => (
              <div key={m.id} className={`choice ${model === m.id ? "on" : ""}`} onClick={() => setModel(m.id)}>
                {m.label}
                <div className="meta mono">{m.repo}</div>
              </div>
            ))}
          </div>
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
